#![cfg(mlx_native_has_metal_tensor_artifact)]

use mlx_native::{
    prepare_quantized_matmul_id_ggml_grouped_m8,
    quantized_matmul_id_ggml_grouped_m8_f16_tiles_probe_into, DType,
    GgmlQuantizedMatmulIdParams, GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
    PreparedGroupedM8Experts,
};
use serde_json::{json, Value};

const PATTERNS: usize = 6;
const PACKET_LIMIT: usize = 16 * 1024 * 1024;
const RAW_CANARY: u8 = 0xa7;
const OUTPUT_CANARY: u16 = 0x7e55;
const TAIL_GUARD_BYTES: usize = 32;

#[derive(Clone, Copy)]
struct Plan {
    ty: GgmlType,
    experts: usize,
    n: usize,
    k: usize,
    raw_offset: usize,
    output_offset: usize,
    padding: usize,
}

impl Plan {
    fn format(self) -> &'static str {
        match self.ty {
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q6_K => "Q6_K",
            _ => unreachable!(),
        }
    }

    fn blocks(self) -> usize {
        self.k / 256
    }

    fn matrix_bytes(self) -> usize {
        self.n * self.blocks() * self.ty.block_bytes() as usize
    }

    fn stride(self) -> usize {
        self.matrix_bytes() + self.padding
    }

    fn raw_view_bytes(self) -> usize {
        (self.experts - 1) * self.stride() + self.matrix_bytes()
    }

    fn raw_root_bytes(self) -> usize {
        // Include the final expert's stride padding outside the logical view.
        self.raw_offset + self.experts * self.stride() + TAIL_GUARD_BYTES
    }

    fn elements(self) -> usize {
        self.experts * self.n * self.blocks() * 16 * 16
    }

    fn output_root_bytes(self) -> usize {
        self.output_offset + self.elements() * 2 + TAIL_GUARD_BYTES
    }

    fn prepared_bytes(self) -> usize {
        // Existing preparation requests 144 bytes/block for Q4 and 212 for Q6.
        let bytes_per_block = match self.ty {
            GgmlType::Q4_K => 144,
            GgmlType::Q6_K => 212,
            _ => unreachable!(),
        };
        self.experts * self.n * self.blocks() * bytes_per_block
    }

    fn params(self) -> GgmlQuantizedMatmulIdParams {
        GgmlQuantizedMatmulIdParams {
            n_tokens: 1,
            top_k: 1,
            n: self.n as u32,
            k: self.k as u32,
            n_experts: self.experts as u32,
            expert_stride: self.stride() as u64,
            ggml_type: self.ty,
        }
    }

    fn evidence(self) -> Value {
        let offsets = json!({
            "raw_u8_view_bytes": self.raw_offset,
            "output_f16_view_bytes": self.output_offset,
        });
        let extents = json!({
            "matrix_bytes": self.matrix_bytes(),
            "raw_expert_stride_bytes": self.stride(),
            "raw_expert_padding_bytes": self.padding,
            "raw_logical_view_bytes": self.raw_view_bytes(),
            "raw_root_requested_bytes": self.raw_root_bytes(),
            "each_output_logical_view_bytes": self.elements() * 2,
            "each_output_root_requested_bytes": self.output_root_bytes(),
            "each_preparation_requested_bytes": self.prepared_bytes(),
            "all_six_preparations_requested_bytes": PATTERNS * self.prepared_bytes(),
            "tail_guard_bytes": TAIL_GUARD_BYTES,
        });
        json!({
            "format": self.format(), "experts": self.experts, "n": self.n, "k": self.k,
            "elements_per_arm_per_case": self.elements(),
            "offsets": offsets, "extents": extents,
        })
    }
}

struct Fixture {
    plan: Plan,
    raw_root: MlxBuffer,
    raw: MlxBuffer,
    reference_root: MlxBuffer,
    reference: MlxBuffer,
    candidate_root: MlxBuffer,
    candidate: MlxBuffer,
    // Retain every prepared allocation until all checked completions finish.
    // This avoids retiring resident buffers between patterns, not a claim about
    // physical memory or hidden allocator behavior. Raw roots are also retained.
    prepared: Vec<PreparedGroupedM8Experts>,
}

impl Fixture {
    fn new(device: &MlxDevice, plan: Plan) -> Self {
        let raw_root = device
            .alloc_buffer(plan.raw_root_bytes(), DType::U8, vec![plan.raw_root_bytes()])
            .expect("raw fixture root");
        let raw = raw_root.slice_view(plan.raw_offset as u64, plan.raw_view_bytes());
        let output = || {
            let root = device
                .alloc_buffer(
                    plan.output_root_bytes(),
                    DType::F16,
                    vec![plan.output_root_bytes() / 2],
                )
                .expect("F16 fixture root");
            let view = root.slice_view(plan.output_offset as u64, plan.elements());
            (root, view)
        };
        let (reference_root, reference) = output();
        let (candidate_root, candidate) = output();
        Self {
            plan,
            raw_root,
            raw,
            reference_root,
            reference,
            candidate_root,
            candidate,
            prepared: Vec::with_capacity(PATTERNS),
        }
    }

    fn fill_source(&mut self, pattern: usize) {
        let p = self.plan;
        let data = self.raw_root.as_mut_slice::<u8>().expect("raw CPU view");
        data.fill(RAW_CANARY);
        for expert in 0..p.experts {
            for row in 0..p.n {
                for block in 0..p.blocks() {
                    let ordinal = (expert * p.n + row) * p.blocks() + block;
                    let offset = p.raw_offset + expert * p.stride()
                        + (row * p.blocks() + block) * p.ty.block_bytes() as usize;
                    fill_block(p.ty, pattern, ordinal, &mut data[offset..offset + p.ty.block_bytes() as usize]);
                }
            }
        }
    }

    fn reset_outputs(&mut self) {
        self.reference_root.as_mut_slice::<u16>().expect("reference CPU view").fill(OUTPUT_CANARY);
        self.candidate_root.as_mut_slice::<u16>().expect("candidate CPU view").fill(OUTPUT_CANARY);
    }

    fn source_differences(&self, pattern: usize) -> (usize, usize) {
        let p = self.plan;
        let data = self.raw_root.as_slice::<u8>().expect("raw CPU view");
        let mut payload = 0;
        let mut guards = 0;
        let mut expected = [0u8; 210];
        let block_bytes = p.ty.block_bytes() as usize;
        guards += data[..p.raw_offset].iter().filter(|&&b| b != RAW_CANARY).count();
        for expert in 0..p.experts {
            let base = p.raw_offset + expert * p.stride();
            for row in 0..p.n {
                for block in 0..p.blocks() {
                    let ordinal = (expert * p.n + row) * p.blocks() + block;
                    fill_block(p.ty, pattern, ordinal, &mut expected[..block_bytes]);
                    let offset = base + (row * p.blocks() + block) * block_bytes;
                    payload += data[offset..offset + block_bytes].iter()
                        .zip(&expected[..block_bytes]).filter(|(a, b)| a != b).count();
                }
            }
            guards += data[base + p.matrix_bytes()..base + p.stride()].iter()
                .filter(|&&b| b != RAW_CANARY).count();
        }
        guards += data[p.raw_offset + p.experts * p.stride()..].iter()
            .filter(|&&b| b != RAW_CANARY).count();
        (payload, guards)
    }

    fn output_guard_differences(&self) -> usize {
        let p = self.plan;
        [&self.reference_root, &self.candidate_root].iter().map(|root| {
            let words = root.as_slice::<u16>().expect("output root CPU view");
            let start = p.output_offset / 2;
            words[..start].iter().chain(&words[start + p.elements()..])
                .filter(|&&word| word != OUTPUT_CANARY).count()
        }).sum()
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn put_half(bytes: &mut [u8], offset: usize, bits: u16) {
    bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

// Fixture encoding only: no CPU dequantization/reference arithmetic.
fn q4_scales(bytes: &mut [u8], scale: [u8; 8], min: [u8; 8]) {
    for j in 0..4 {
        bytes[4 + j] = scale[j] | ((scale[j + 4] >> 4) << 6);
        bytes[8 + j] = min[j] | ((min[j + 4] >> 4) << 6);
        bytes[12 + j] = (scale[j + 4] & 15) | ((min[j + 4] & 15) << 4);
    }
}

fn pattern_name(ty: GgmlType, pattern: usize) -> &'static str {
    match ty {
        GgmlType::Q4_K => [
            "minimum_normal_f32_division", "minimum_subnormal_no_flush_assumption",
            "signed_zeros", "cancellation_and_half_rounding", "packed_scale_min_extremes",
            "seeded_mixed",
        ][pattern],
        GgmlType::Q6_K => [
            "codes_31_32_33", "signed_scales_minus128_plus127", "signed_zeros",
            "cancellation_and_half_rounding", "centered_zero_signed_scales", "seeded_mixed",
        ][pattern],
        _ => unreachable!(),
    }
}

fn fill_block(ty: GgmlType, pattern: usize, ordinal: usize, b: &mut [u8]) {
    b.fill(0);
    let mut rng = Rng(0xf160_4b6b_7a11_0001 ^ (ordinal as u64).wrapping_mul(0xd134_2543_de82_ef95)
        ^ ((pattern as u64) << 48));
    match ty {
        GgmlType::Q4_K => {
            let (d, dmin) = match pattern {
                0 => (0x0401, 0),
                1 => (0x0001, 0),
                2 => if ordinal & 1 == 0 { (0x8000, 0) } else { (0, 0x8000) },
                // Odd blocks: 1.0009765625 - 0.00048828125 is a half-way value.
                // Even blocks include exact cancellation when the code is one.
                3 => (0x3c01, if ordinal & 1 == 0 { 0x3c01 } else { 0x1000 }),
                4 => (if ordinal & 1 == 0 { 0x2401 } else { 0xa401 }, 0x2001),
                5 => (0x2400 | (rng.next() as u16 & 0x8fff), 0x2000 | (rng.next() as u16 & 0x0fff)),
                _ => unreachable!(),
            };
            put_half(b, 0, d);
            put_half(b, 2, dmin);
            match pattern {
                0 | 1 => q4_scales(b, [1; 8], [0; 8]),
                2 | 3 => q4_scales(b, [1; 8], [1; 8]),
                4 => q4_scales(b, [0, 1, 15, 16, 31, 32, 62, 63], [63, 62, 32, 31, 16, 15, 1, 0]),
                5 => {
                    for byte in &mut b[4..16] { *byte = rng.next() as u8; }
                }
                _ => unreachable!(),
            }
            for (i, byte) in b[16..].iter_mut().enumerate() {
                *byte = match pattern {
                    0 | 1 => 0x11,
                    2 => if i & 1 == 0 { 0x10 } else { 0xf1 },
                    3 => [0x11, 0x33, 0xff, 0x10][i % 4],
                    4 => ((i + ordinal) as u8 & 15) | (((i * 7 + ordinal) as u8 & 15) << 4),
                    5 => rng.next() as u8,
                    _ => unreachable!(),
                };
            }
        }
        GgmlType::Q6_K => {
            let d = match pattern {
                0 => 0x3c00,
                1 => 0x2001,
                2 => if ordinal & 1 == 0 { 0 } else { 0x8000 },
                3 => if ordinal & 1 == 0 { 0x3c01 } else { 0xbc01 },
                4 => 0x3a01,
                5 => 0x2400 | (rng.next() as u16 & 0x8fff),
                _ => unreachable!(),
            };
            put_half(b, 208, d);
            for (i, scale) in b[192..208].iter_mut().enumerate() {
                *scale = match pattern {
                    0 => 1,
                    1 | 2 | 4 => if (i + ordinal) & 1 == 0 { (-128i8) as u8 } else { 127 },
                    3 => [1, 3, 5, 7, 11, 17, 31, 63][i % 8],
                    5 => rng.next() as u8,
                    _ => unreachable!(),
                };
            }
            for group in 0..8 {
                for e in 0..32 {
                    let index = group * 32 + e;
                    let code: u8 = match pattern {
                        0 => [31, 32, 33][(index + ordinal) % 3],
                        1 => [0, 31, 32, 33, 63][(index + ordinal) % 5],
                        2 => [31, 32, 33][index % 3],
                        // Code 35, scale 1, d=0x3c01 lands exactly half-way
                        // between adjacent F16 numbers near three; 32 cancels.
                        3 => [31, 32, 33, 35][e % 4],
                        4 => 32,
                        5 => rng.next() as u8 & 63,
                        _ => unreachable!(),
                    };
                    let low = 64 * (group / 4) + 32 * (group & 1) + e;
                    let high = 128 + 32 * (group / 4) + e;
                    b[low] |= (code & 15) << (4 * ((group % 4) / 2));
                    b[high] |= (code >> 4) << (2 * (group % 4));
                }
            }
        }
        _ => unreachable!(),
    }
}

fn inspect_case(f: &Fixture, pattern: usize, tensor: bool, fixture_index: usize) -> (Value, usize) {
    let p = f.plan;
    let reference = f.reference.as_slice::<u16>().expect("reference bits");
    let candidate = f.candidate.as_slice::<u16>().expect("candidate bits");
    let mut differences = 0;
    let mut first_difference = None;
    let mut nonfinite = [0usize; 2];
    let mut nonzero = [0usize; 2];
    let mut negative_zero = [0usize; 2];
    let mut differences_by_il = [0usize; 16];
    let mut nonzero_by_il = [0usize; 16];
    let mut normal_edge_checks = 0;
    let mut normal_edge_failures = 0;
    for (index, (&expected, &actual)) in reference.iter().zip(candidate).enumerate() {
        let il = (index / 16) % 16;
        if actual != expected {
            differences += 1;
            differences_by_il[il] += 1;
            if first_difference.is_none() {
                let block_row = index / 256;
                first_difference = Some(json!({
                    "expert": block_row / (p.n * p.blocks()),
                    "row": (block_row / p.blocks()) % p.n,
                    "block": block_row % p.blocks(), "il": il,
                    "fragment_column": (index % 16) / 4,
                    "fragment_row": index % 4,
                    "reference_half_bits": expected, "candidate_half_bits": actual,
                }));
            }
        }
        for (arm, bits) in [expected, actual].into_iter().enumerate() {
            nonfinite[arm] += usize::from(bits & 0x7c00 == 0x7c00);
            nonzero[arm] += usize::from(bits & 0x7fff != 0);
            negative_zero[arm] += usize::from(bits == 0x8000);
            if p.ty == GgmlType::Q4_K && pattern == 0 && il % 4 >= 2 {
                normal_edge_checks += 1;
                normal_edge_failures += usize::from(bits != 0x0401);
            }
        }
        nonzero_by_il[il] += usize::from(expected & 0x7fff != 0);
    }
    let must_be_zero = pattern == 2 || (p.ty == GgmlType::Q6_K && pattern == 4);
    let may_flush_subnormal = p.ty == GgmlType::Q4_K && pattern == 1;
    let meaningful_failures = if must_be_zero {
        usize::from(nonzero != [0, 0])
    } else if may_flush_subnormal {
        0 // Exact GPU-arm equality is required; no hardware denormal assumption.
    } else {
        usize::from(nonzero.contains(&0)) + nonzero_by_il.iter().filter(|&&n| n == 0).count()
    };
    let (source_payload_differences, source_guard_differences) = f.source_differences(pattern);
    let output_guard_differences = f.output_guard_differences();
    let failures = differences + nonfinite.iter().sum::<usize>() + normal_edge_failures
        + meaningful_failures + source_payload_differences + source_guard_differences
        + output_guard_differences;
    let bits = json!({
        "exact_differences": differences, "first_difference": first_difference,
        "differences_by_il": differences_by_il, "reference_nonzero_by_il": nonzero_by_il,
        "reference_candidate_nonfinite": nonfinite, "reference_candidate_nonzero": nonzero,
        "reference_candidate_negative_zero": negative_zero,
        "normal_edge_checked_values_both_arms": normal_edge_checks,
        "normal_edge_failures": normal_edge_failures, "meaningful_value_failures": meaningful_failures,
        "all_values_required_zero": must_be_zero,
        "subnormal_flush_behavior_unspecified": may_flush_subnormal,
    });
    let bounds = json!({
        "source_payload_differences": source_payload_differences,
        "source_guard_or_padding_differences": source_guard_differences,
        "output_guard_differences": output_guard_differences,
    });
    (json!({
        "fixture": fixture_index, "pattern": pattern_name(p.ty, pattern),
        "reference_mode": if tensor { "tensor" } else { "classic" },
        "compared_elements": p.elements(), "bits": bits, "bounds": bounds,
    }), failures)
}

/// Native-only decoder parity experiment, not a production selector or fit test.
#[test]
#[ignore = "preregistered bounded native-only F16 decoder-tile spike"]
fn grouped_m8_prepared_f16_tiles_spike() {
    let plans = [GgmlType::Q4_K, GgmlType::Q6_K].map(|ty| [
        Plan { ty, experts: 1, n: 64, k: 256, raw_offset: 0, output_offset: 0, padding: 0 },
        Plan { ty, experts: 3, n: 128, k: 512, raw_offset: 32, output_offset: 16, padding: 32 },
    ]);
    // Plan the entire packet BEFORE the first fixture allocation, including
    // every preparation retained across all patterns, formats and geometries.
    let raw_requested: usize = plans.iter().flatten().map(|p| p.raw_root_bytes()).sum();
    let output_requested: usize = plans.iter().flatten().map(|p| 2 * p.output_root_bytes()).sum();
    let prepared_requested: usize = plans.iter().flatten().map(|p| PATTERNS * p.prepared_bytes()).sum();
    let total_requested = raw_requested + output_requested + prepared_requested;
    assert!(total_requested < PACKET_LIMIT, "complete fixture packet exceeds preregistered bound");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut fixtures: Vec<_> = plans.into_iter().flatten().map(|p| Fixture::new(&device, p)).collect();
    let mut cases = Vec::with_capacity(48);
    let mut failures = 0;
    let mut compared_elements = 0;
    let mut exact_differences = 0;
    let mut preparations = 0;
    for (fixture_index, f) in fixtures.iter_mut().enumerate() {
        for pattern in 0..PATTERNS {
            f.fill_source(pattern);
            let params = f.plan.params();
            let mut encoder = device.command_encoder().expect("prepare/classic encoder");
            let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(
                &mut encoder, &mut registry, &device, &f.raw, &params,
            ).expect("prepare expert planes once per pattern");
            preparations += 1;
            // Preparation's own barrier orders its async writes before the
            // classic candidate on the same encoder. No source owners drop.
            for tensor in [false, true] {
                if tensor {
                    encoder = device.command_encoder().expect("tensor encoder");
                }
                f.reset_outputs();
                quantized_matmul_id_ggml_grouped_m8_f16_tiles_probe_into(
                    &mut encoder, &mut registry, &f.raw, &prepared,
                    &f.reference, &f.candidate, &params, tensor,
                ).expect("encode actual production decoder and prepared decoder");
                encoder.commit_and_wait_labeled("test.grouped_m8.prepared_f16_tiles")
                    .expect("checked GPU completion before CPU access or fixture reuse");
                let (evidence, case_failures) = inspect_case(f, pattern, tensor, fixture_index);
                exact_differences += evidence["bits"]["exact_differences"].as_u64().expect("difference count");
                compared_elements += f.plan.elements();
                failures += case_failures;
                cases.push(evidence);
            }
            // All use has completed; retain instead of retiring allocations
            // in the loop. Raw/output roots are reused only after completion.
            f.prepared.push(prepared);
        }
    }
    let geometries: Vec<_> = fixtures.iter().map(|f| f.plan.evidence()).collect();
    let allocation = json!({
        "planned_requested_bytes": {
            "raw_roots": raw_requested, "both_output_roots": output_requested,
            "all_24_prepared_allocations": prepared_requested, "total_fixture_packet": total_requested,
        },
        "fixture_packet_upper_bytes": PACKET_LIMIT,
        "planning_basis": "all requested Metal fixture extents, including all retained preparations; excludes shader/runtime infrastructure",
        "raw_and_output_roots_reused_across_patterns_and_reference_modes": true,
        "all_prepared_owners_retained_through_final_checked_completion": true,
        "source_owner_release": false,
        "physical_bytes_observed": null, "allocator_bytes_observed": null,
        "hidden_allocator_growth_observed": null,
    });
    let instrumentation = json!({
        "MTL_SHADER_VALIDATION": std::env::var("MTL_SHADER_VALIDATION").ok(),
        "MTL_DEBUG_LAYER": std::env::var("MTL_DEBUG_LAYER").ok(),
    });
    let counts = json!({
        "formats": 2, "geometries_per_format": 2, "patterns_per_format": PATTERNS,
        "reference_modes": 2, "cases": cases.len(), "preparations": preparations,
        "il_positions_per_block": 16, "fragment_elements_per_il": 16,
        "compared_elements": compared_elements, "exact_differences": exact_differences,
        "gate_failures": failures,
    });
    // Construct nested values separately: no macro recursion-limit override.
    let summary = json!({
        "schema": "mlx-native.prepared-f16-tiles-spike.v1",
        "test": "grouped_m8_prepared_f16_tiles_spike",
        "counts": counts, "allocation": allocation, "instrumentation": instrumentation,
        "geometries": geometries, "cases": cases,
        "production_cutover": false, "release_qualification": false,
    });
    let line = serde_json::to_string(&summary).expect("bounded JSON summary");
    assert!(line.len() < 64 * 1024, "summary exceeds bounded evidence size");
    // Keep stdout's exact test-name/result line intact for the strict runner.
    eprintln!("{line}");
    assert_eq!(preparations, 24);
    assert_eq!(summary["counts"]["cases"], 48);
    assert_eq!(failures, 0, "F16 tile parity/edge/bounds gate failed; see STDERR JSON");
}
