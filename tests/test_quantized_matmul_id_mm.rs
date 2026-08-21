//! Unit tests for the GGML block-format `_id` matrix-matrix (mm) quantized
//! GPU kernels.
//!
//! ADR-011 Phase 3 Wave P3a port of the reference
//! `kernel_mul_mm_id_map0_ne20_<N>` + `kernel_mul_mm_id_<qtype>_f32`.
//!
//! Verification strategy: for a random MoE setup (weights, tokens, expert
//! ids), we run both the mv `_id` kernel (known-good via
//! `test_quantized_matmul_id_ggml.rs`) and the new mm `_id` kernel, then
//! compare outputs.  Both kernels produce `[n_tokens, top_k, N]` row-major
//! output; tolerance covers simdgroup-MMA vs scalar-simd_sum reduction
//! ordering differences (O(K * eps * max|v|) per element).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    CapturedNode, DType, GgmlIdMmDispatchParams, GgmlQuantizedMatmulIdParams, GgmlType,
    KernelRegistry, MlxDevice,
};

#[test]
fn mm_id_capture_annotates_both_dispatches() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let input = device
        .alloc_buffer(32 * 4, DType::F32, vec![1, 32])
        .unwrap();
    let weight = device
        .alloc_buffer(32 * 34, DType::U8, vec![32 * 34])
        .unwrap();
    let ids = device.alloc_buffer(4, DType::U32, vec![1, 1]).unwrap();
    let htpe = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
    let hids = device.alloc_buffer(4, DType::U32, vec![1, 1]).unwrap();
    let output = device
        .alloc_buffer(32 * 4, DType::F32, vec![1, 1, 32])
        .unwrap();
    let params = GgmlIdMmDispatchParams {
        n_tokens: 1,
        top_k: 1,
        n: 32,
        k: 32,
        n_experts: 1,
        expert_stride: (32 * 34) as u64,
        ggml_type: GgmlType::Q8_0,
    };
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    mlx_native::dispatch_id_mm_for_test(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &htpe,
        &hids,
        &output,
        &params,
    )
    .unwrap();

    let captured = encoder.take_capture().unwrap();
    assert_eq!(captured.len(), 3);
    match &captured[0] {
        CapturedNode::Dispatch { reads, writes, .. } => {
            assert_eq!(reads.len(), 1);
            assert_eq!(writes.len(), 2);
        }
        CapturedNode::Barrier => panic!("expected map dispatch"),
    }
    assert!(matches!(captured[1], CapturedNode::Barrier));
    match &captured[2] {
        CapturedNode::Dispatch { reads, writes, .. } => {
            assert_eq!(reads.len(), 4);
            assert_eq!(writes.len(), 1);
        }
        CapturedNode::Barrier => panic!("expected expert matmul dispatch"),
    }
}

// --------------------------------------------------------------------------
// PRNG + block pack helpers (copied from existing test files to keep this
// test self-contained — same exact CPU helpers, same bit-for-bit packing).
// --------------------------------------------------------------------------

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let v0 = (block[i] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            let v1 = (block[i + 16] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            nibbles[i] = v0 | (v1 << 4);
        }
        buf.extend_from_slice(&nibbles);
    }
    buf
}

fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            buf.push(q as u8);
        }
    }
    buf
}

fn pack_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(256) {
        let mut sub_scales = [0.0f32; 16];
        let mut sub_scale_int = [0i8; 16];
        let mut max_scale: f32 = 0.0;
        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            sub_scales[s] = amax;
            if amax > max_scale { max_scale = amax; }
        }
        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else { 0 };
        }
        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                q6[s * 16 + i] = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = l0_base + l;
                let v0 = q6[l0_base + l];
                let v2 = q6[l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);
                let v1 = q6[l0_base + l + 32];
                let v3 = q6[l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);
                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[ql_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                let v0 = q6[128 + l0_base + l];
                let v2 = q6[128 + l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);
                let v1 = q6[128 + l0_base + l + 32];
                let v3 = q6[128 + l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);
                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[qh_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }
        buf.extend_from_slice(&ql);
        buf.extend_from_slice(&qh);
        buf.extend_from_slice(
            &sub_scale_int.iter().map(|&s| s as u8).collect::<Vec<_>>(),
        );
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
    }
    buf
}

fn stack_expert_weights(expert_packed: &[Vec<u8>]) -> (Vec<u8>, usize) {
    let per_expert = expert_packed[0].len();
    for ep in expert_packed {
        assert_eq!(ep.len(), per_expert);
    }
    let mut stacked = Vec::with_capacity(per_expert * expert_packed.len());
    for ep in expert_packed {
        stacked.extend_from_slice(ep);
    }
    (stacked, per_expert)
}

// --------------------------------------------------------------------------
// Verification: run both mv_id (reference) and mm_id (under test), compare.
// --------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_mm_id_vs_mv_id_test(
    ggml_type: GgmlType,
    pack_fn: fn(&[f32]) -> Vec<u8>,
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    n: usize,
    k: usize,
    tolerance: f32,
    label: &str,
) {
    assert_eq!(top_k, 8, "only top_k=8 is instantiated in the shader");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    // ---- Inputs ----
    let input_data = pseudo_random_f32(0x5E5EED, n_tokens * k);
    let mut expert_packed = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let w = pseudo_random_f32(1000 + e as u64, n * k);
        expert_packed.push(pack_fn(&w));
    }
    let (stacked_bytes, per_expert_bytes) = stack_expert_weights(&expert_packed);

    // Random-looking but seed-deterministic routing
    let mut ids: Vec<u32> = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids.push(((t * 17 + s * 13 + 7) % n_experts) as u32);
        }
    }

    let total_rows = n_tokens * top_k;

    // ---- GPU buffers (shared across both runs) ----
    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens, k])
        .unwrap();
    input_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U8, vec![stacked_bytes.len()])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked_bytes);

    let mut ids_buf = device
        .alloc_buffer(ids.len() * u32_sz, DType::U32, vec![ids.len()])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    // ---- Run mv_id (reference) ----
    let mut mv_output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows, n])
        .unwrap();
    for v in mv_output_buf.as_mut_slice::<f32>().unwrap().iter_mut() { *v = 0.0; }

    {
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: top_k as u32,
            n: n as u32,
            k: k as u32,
            n_experts: n_experts as u32,
            expert_stride: per_expert_bytes as u64,
            ggml_type,
        };
        let mut enc = device.command_encoder().unwrap();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut enc, &mut registry, &device,
            &input_buf, &weight_buf, &ids_buf, &mut mv_output_buf,
            &params,
        ).unwrap();
        enc.commit_and_wait().unwrap();
    }

    // ---- Run mm_id (under test) ----
    let mut mm_output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows, n])
        .unwrap();
    for v in mm_output_buf.as_mut_slice::<f32>().unwrap().iter_mut() { *v = 0.0; }

    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type,
    };
    let mut htpe = device
        .alloc_buffer(dispatch.htpe_bytes(), DType::U32, vec![n_experts])
        .unwrap();
    let mut hids = device
        .alloc_buffer(dispatch.hids_bytes(), DType::U32, vec![n_experts, n_tokens])
        .unwrap();
    // Zero-init htpe + hids is not strictly required (kernels write all slots
    // the kernel will read), but defensive clearing makes ULP errors from
    // leftover scratch memory easier to spot.
    for v in htpe.as_mut_slice::<u32>().unwrap().iter_mut() { *v = 0; }
    for v in hids.as_mut_slice::<u32>().unwrap().iter_mut() { *v = 0; }

    {
        let mut enc = device.command_encoder().unwrap();
        mlx_native::dispatch_id_mm_for_test(
            &mut enc, &mut registry, &device,
            &input_buf, &weight_buf, &ids_buf,
            &mut htpe, &mut hids, &mut mm_output_buf,
            &dispatch,
        ).unwrap();
        enc.commit_and_wait().unwrap();
    }

    // ---- Compare ----
    let mv: &[f32] = mv_output_buf.as_slice().unwrap();
    let mm: &[f32] = mm_output_buf.as_slice().unwrap();

    let mut max_err: f32 = 0.0;
    let mut err_count = 0usize;
    let mut first_err = None;
    for i in 0..mv.len() {
        let err = (mm[i] - mv[i]).abs();
        if err > max_err {
            max_err = err;
        }
        if err > tolerance {
            if first_err.is_none() { first_err = Some((i, mm[i], mv[i], err)); }
            err_count += 1;
        }
    }

    if err_count > 0 {
        let (i, mm_v, mv_v, err) = first_err.unwrap();
        let row = i / n;
        let col = i % n;
        let tok = row / top_k;
        let slot = row % top_k;
        let expert = ids[row];
        panic!(
            "{}: mm_id vs mv_id: {} mismatches (max_err={}, first at tok={} slot={} expert={} col={}: mm={} mv={} err={})",
            label, err_count, max_err, tok, slot, expert, col, mm_v, mv_v, err,
        );
    }

    eprintln!(
        "{}: PASS (max_err={:.6}, tol={}, n_tokens={}, top_k={}, n={}, k={})",
        label, max_err, tolerance, n_tokens, top_k, n, k,
    );
}

// ==========================================================================
// Debug: verify map0 produces sane htpe and hids
// ==========================================================================

// ==========================================================================
// Diagnostic: map0 produces exactly the expected htpe + hids
// ==========================================================================
//
// This test caught a subtle bug during development: mlx-native's compute
// encoder runs dispatches concurrently by default, so without a
// `memory_barrier()` between map0 and mm_id the mm kernel read htpe as
// all-zeros and early-exited every threadgroup.  The peer engine emits
// the same barrier at its concurrency-reset point.
//
// The test verifies (a) map0 produces the expected counts and routed-ids,
// and (b) the full two-stage dispatch of `dispatch_id_mm_for_test` leaves
// htpe/hids with correct values afterward (i.e. the dispatcher doesn't
// clobber them).

#[test]
fn test_map0_produces_sane_htpe_and_hids() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let n_tokens = 16usize;
    let n_experts = 8usize;
    let top_k = 8usize;

    // Same routing as run_mm_id_vs_mv_id_test.
    let mut ids: Vec<u32> = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids.push(((t * 17 + s * 13 + 7) % n_experts) as u32);
        }
    }

    let mut ids_buf = device
        .alloc_buffer(ids.len() * 4, DType::U32, vec![ids.len()])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    // CPU reference: count per expert, build routed lists.
    let mut cpu_counts = vec![0u32; n_experts];
    let mut cpu_hids = vec![vec![]; n_experts];
    for t in 0..n_tokens {
        for s in 0..top_k {
            let e = ids[t * top_k + s] as usize;
            cpu_counts[e] += 1;
            cpu_hids[e].push((t * top_k + s) as i32);
        }
    }

    // Allocate + zero scratch.
    let mut htpe = device
        .alloc_buffer(n_experts * 4, DType::U32, vec![n_experts])
        .unwrap();
    let mut hids = device
        .alloc_buffer(n_experts * n_tokens * 4, DType::U32, vec![n_experts, n_tokens])
        .unwrap();
    for v in htpe.as_mut_slice::<u32>().unwrap().iter_mut() { *v = 0xDEAD_BEEF; }
    for v in hids.as_mut_slice::<u32>().unwrap().iter_mut() { *v = 0xDEAD_BEEF; }

    // Dispatch just the map0 step by reusing dispatch_id_mm_for_test's internals
    // via a minimal GgmlIdMmDispatchParams + a one-shot map0 call.  We'll do
    // this by running the full dispatch and reading htpe afterward — the mm
    // kernel reads htpe but doesn't modify it, so post-dispatch htpe reflects
    // map0's output.

    // Fabricate weights + input + output for a minimal mm launch.
    let k = 128usize;
    let n = 64usize;
    let weights: Vec<u8> = pack_q4_0(&vec![0.0f32; n * k]);
    let stacked: Vec<u8> = (0..n_experts).flat_map(|_| weights.clone()).collect();

    let mut weight_buf = device.alloc_buffer(stacked.len(), DType::U8, vec![stacked.len()]).unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked);
    let mut input_buf = device.alloc_buffer(n_tokens * k * 4, DType::F32, vec![n_tokens, k]).unwrap();
    for v in input_buf.as_mut_slice::<f32>().unwrap().iter_mut() { *v = 0.0; }
    let mut output_buf = device.alloc_buffer(n_tokens * top_k * n * 4, DType::F32, vec![n_tokens * top_k, n]).unwrap();

    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: weights.len() as u64,
        ggml_type: GgmlType::Q4_0,
    };

    let mut enc = device.command_encoder().unwrap();
    mlx_native::dispatch_id_mm_for_test(
        &mut enc, &mut registry, &device,
        &input_buf, &weight_buf, &ids_buf,
        &mut htpe, &mut hids, &mut output_buf, &dispatch,
    ).unwrap();
    enc.commit_and_wait().unwrap();

    let htpe_slice = htpe.as_slice::<u32>().unwrap();
    let hids_slice = hids.as_slice::<i32>().unwrap();

    eprintln!("cpu_counts = {:?}", cpu_counts);
    eprintln!("gpu htpe   = {:?}", htpe_slice);

    for e in 0..n_experts {
        assert_eq!(
            htpe_slice[e], cpu_counts[e],
            "expert {}: htpe_gpu={} != htpe_cpu={}",
            e, htpe_slice[e], cpu_counts[e],
        );
    }

    // Check that each expert's hids row starts with the right packed ids
    // (order doesn't need to match CPU exactly — map0 walks tokens in order
    // so ours is in increasing (token*top_k + slot) order, CPU did the same).
    for e in 0..n_experts {
        let count = cpu_counts[e] as usize;
        eprintln!(
            "expert {}: cpu_hids[0..{}] = {:?}  gpu_hids[0..{}] = {:?}",
            e, count, &cpu_hids[e],
            count, &hids_slice[e * n_tokens..e * n_tokens + count],
        );
        for i in 0..count {
            let cpu = cpu_hids[e][i];
            let gpu = hids_slice[e * n_tokens + i];
            assert_eq!(cpu, gpu, "expert {} entry {}: cpu={} gpu={}", e, i, cpu, gpu);
        }
    }
}

// ==========================================================================
// Q4_0 mm_id tests
// ==========================================================================

#[test]
fn test_q4_0_mm_id_matches_mv_id_small() {
    // n_tokens=16 (> mm threshold), 8 experts, top_k=8 — every token picks
    // all 8 experts, so expert routing is trivially uniform.  Small K for
    // CPU-side speed.
    run_mm_id_vs_mv_id_test(
        GgmlType::Q4_0, pack_q4_0,
        16, 8, 8,    // n_tokens, n_experts, top_k
        64, 128,     // n, k
        5e-3,
        "Q4_0 mm_id matches mv_id, n_tokens=16 top_k=8 N=64 K=128",
    );
}

#[test]
fn test_q4_0_mm_id_matches_mv_id_prefill_shape() {
    // Mid-sized: tokens like a short prefill, more experts, larger K.
    run_mm_id_vs_mv_id_test(
        GgmlType::Q4_0, pack_q4_0,
        64, 32, 8,
        128, 512,
        5e-2,
        "Q4_0 mm_id matches mv_id, n_tokens=64 n_experts=32 N=128 K=512",
    );
}

// ==========================================================================
// Q8_0 mm_id tests
// ==========================================================================

#[test]
fn test_q8_0_mm_id_matches_mv_id_small() {
    run_mm_id_vs_mv_id_test(
        GgmlType::Q8_0, pack_q8_0,
        16, 8, 8,
        64, 128,
        2e-3,
        "Q8_0 mm_id matches mv_id, n_tokens=16 N=64 K=128",
    );
}

#[test]
fn test_q8_0_mm_id_matches_mv_id_prefill_shape() {
    run_mm_id_vs_mv_id_test(
        GgmlType::Q8_0, pack_q8_0,
        64, 32, 8,
        128, 512,
        2e-2,
        "Q8_0 mm_id matches mv_id, n_tokens=64 n_experts=32 N=128 K=512",
    );
}

// ==========================================================================
// Q6_K mm_id tests
// ==========================================================================

#[test]
fn test_q6_k_mm_id_matches_mv_id_small() {
    run_mm_id_vs_mv_id_test(
        GgmlType::Q6_K, pack_q6_k,
        16, 8, 8,
        32, 256,
        5e-3,
        "Q6_K mm_id matches mv_id, n_tokens=16 N=32 K=256",
    );
}

#[test]
fn test_q6_k_mm_id_matches_mv_id_prefill_shape() {
    run_mm_id_vs_mv_id_test(
        GgmlType::Q6_K, pack_q6_k,
        64, 32, 8,
        128, 512,
        5e-2,
        "Q6_K mm_id matches mv_id, n_tokens=64 n_experts=32 N=128 K=512",
    );
}

// ==========================================================================
// Irregular shapes — partial tiles, sparse expert routing
// ==========================================================================

#[test]
fn test_q4_0_mm_id_partial_tiles() {
    // n_tokens=17 → one token past the 16-row MM tile boundary;
    // n=72 → 72/64 = 1 full tile + 8-col tail.
    // This exercises both the partial-M and partial-N write-back paths in
    // mm_id.
    run_mm_id_vs_mv_id_test(
        GgmlType::Q4_0, pack_q4_0,
        17, 16, 8,
        72, 256,
        5e-3,
        "Q4_0 mm_id partial tiles, n_tokens=17 N=72 K=256",
    );
}

#[test]
fn test_q4_0_mm_id_sparse_experts() {
    // Many experts, few tokens → average per-expert batch < 1, so most
    // expert-tiles early-exit via `r1 >= neh1`.  Correctness stressor.
    run_mm_id_vs_mv_id_test(
        GgmlType::Q4_0, pack_q4_0,
        16, 64, 8,
        64, 128,
        5e-3,
        "Q4_0 mm_id sparse experts, 16 tokens x 64 experts top_k=8",
    );
}

// ==========================================================================
// ADR-028 iter-94 — per-call timing harness for Q6_K mv_id at gemma decode
// shape (n_tokens=1, top_k=8). Establishes baseline for kernel-level
// optimization A/B testing in iter-95+.
//
// Hypothesis: per iter-90, hf2q decode = 15.86 µs/dispatch, peer-engine
// decode (FA=1, gemma APEX-Q5_K_M) = 11.22 µs/dispatch — 30% slower per
// dispatch. Per-call instrumentation here measures Q6_K mv_id specifically
// (the dominant gemma-MoE matmul kernel).
//
// Shape derivation (from /opt/hf2q/src/serve/config.rs:100):
//   gemma-4-26b-A4B-IT defaults: hidden=2816, moe_intermediate=704,
//   n_experts=128, top_k=8.
//
// Run with:
//   cargo test --release --test test_quantized_matmul_id_mm \
//     -- --ignored bench_q6_k_mv_id_gemma_decode --nocapture
//
// Reports median + p10/p50/p90 ms/call over N=200 trials with 20 warmup.
// Uses commit_and_wait for hard sync per-iteration (matches per-dispatch
// overhead measurement; not amortized over a session).

#[test]
#[ignore]
fn bench_q6_k_mv_id_gemma_decode_gate_up() {
    bench_q6_k_mv_id_one_shape(
        /* label = */ "Q6_K mv_id gemma decode gate_up: n=704, k=2816, n_experts=128, top_k=8",
        /* n     = */ 704,
        /* k     = */ 2816,
    );
}

// Note: gemma's down projection has K=moe_intermediate=704 which is NOT
// divisible by Q6_K block size (QK_K=256). Real gemma down_exps may be
// stored as Q5_1/Q4_K with different block size, or padded — needs
// model inspection. Iter-95+ candidate.

fn bench_q6_k_mv_id_one_shape(label: &str, n: usize, k: usize) {
    use std::time::Instant;

    let n_tokens: usize = 1;
    let top_k: usize = 8;
    let n_experts: usize = 128;

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    // Inputs.
    let input_data = pseudo_random_f32(0x5E5EED, n_tokens * k);
    let mut expert_packed = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let w = pseudo_random_f32(1000 + e as u64, n * k);
        expert_packed.push(pack_q6_k(&w));
    }
    let (stacked_bytes, per_expert_bytes) = stack_expert_weights(&expert_packed);

    // Routing: top_k=8 distinct experts.
    let mut ids: Vec<u32> = Vec::with_capacity(n_tokens * top_k);
    for s in 0..top_k {
        ids.push(((s * 17 + 7) % n_experts) as u32);
    }

    let total_rows = n_tokens * top_k;

    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens, k])
        .unwrap();
    input_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U8, vec![stacked_bytes.len()])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked_bytes);

    let mut ids_buf = device
        .alloc_buffer(ids.len() * u32_sz, DType::U32, vec![ids.len()])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    let mut output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows, n])
        .unwrap();

    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::Q6_K,
    };

    // Warmup (compile pipeline + warm caches).
    const WARMUP: usize = 20;
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().unwrap();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut enc, &mut registry, &device,
            &input_buf, &weight_buf, &ids_buf, &mut output_buf,
            &params,
        ).unwrap();
        enc.commit_and_wait().unwrap();
    }

    // Measurement: capture BOTH CPU wall-clock (encode + commit + wait) AND
    // GPU-pure (MTLCommandBuffer.GPUStartTime/GPUEndTime) per call. The
    // delta isolates encoder + commit overhead from actual kernel compute.
    // ADR-028 iter-96: separate dispatch overhead from kernel time to test
    // the "encode-bound" hypothesis.
    const N_TRIALS: usize = 200;
    let mut cpu_us: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut gpu_us: Vec<f64> = Vec::with_capacity(N_TRIALS);
    for _ in 0..N_TRIALS {
        let mut enc = device.command_encoder().unwrap();
        let t0 = Instant::now();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut enc, &mut registry, &device,
            &input_buf, &weight_buf, &ids_buf, &mut output_buf,
            &params,
        ).unwrap();
        let (gpu_start_s, gpu_end_s) = enc.commit_wait_with_gpu_time().unwrap();
        let dt = t0.elapsed();
        cpu_us.push(dt.as_secs_f64() * 1e6);
        gpu_us.push((gpu_end_s - gpu_start_s) * 1e6);
    }

    let pct = |xs: &mut Vec<f64>, q: usize| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[q]
    };

    let cpu_p10 = pct(&mut cpu_us, N_TRIALS / 10);
    let cpu_p50 = pct(&mut cpu_us, N_TRIALS / 2);
    let cpu_p90 = pct(&mut cpu_us, (N_TRIALS * 9) / 10);

    let gpu_p10 = pct(&mut gpu_us, N_TRIALS / 10);
    let gpu_p50 = pct(&mut gpu_us, N_TRIALS / 2);
    let gpu_p90 = pct(&mut gpu_us, (N_TRIALS * 9) / 10);

    let overhead_p50 = cpu_p50 - gpu_p50;
    let overhead_pct = overhead_p50 / cpu_p50 * 100.0;

    eprintln!("[BENCH] {}", label);
    eprintln!("[BENCH]   CPU wall-clock  p10 = {:7.2} µs  p50 = {:7.2} µs  p90 = {:7.2} µs",
              cpu_p10, cpu_p50, cpu_p90);
    eprintln!("[BENCH]   GPU pure-kernel p10 = {:7.2} µs  p50 = {:7.2} µs  p90 = {:7.2} µs",
              gpu_p10, gpu_p50, gpu_p90);
    eprintln!("[BENCH]   Encode+commit overhead p50 = {:6.2} µs ({:.1}% of CPU)",
              overhead_p50, overhead_pct);
    eprintln!("[BENCH]   Per-token (×60 calls):");
    eprintln!("[BENCH]     CPU estimate = {:6.2} ms (production single-session amortizes overhead)",
              cpu_p50 * 60.0 / 1000.0);
    eprintln!("[BENCH]     GPU estimate = {:6.2} ms (kernel-only lower bound)",
              gpu_p50 * 60.0 / 1000.0);
}

// ADR-028 iter-97 — multi-dispatch session bench. Mirrors PRODUCTION
// single-session amortization: 60 mv_id Q6_K dispatches in ONE encoder,
// ONE commit_wait_with_gpu_time at the end. Per-call cost = total / 60.
//
// Compares against the per-call isolated bench above to quantify how
// much encode overhead is amortized in production.
//
// Run:
//   cargo test --release --test test_quantized_matmul_id_mm \
//     bench_q6_k_mv_id_session60 -- --ignored --nocapture
#[test]
#[ignore]
fn bench_q6_k_mv_id_session60_gemma_decode() {
    use std::time::Instant;

    let n_tokens: usize = 1;
    let top_k: usize = 8;
    let n_experts: usize = 128;
    let n: usize = 704;
    let k: usize = 2816;

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    let input_data = pseudo_random_f32(0x5E5EED, n_tokens * k);
    let mut expert_packed = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let w = pseudo_random_f32(1000 + e as u64, n * k);
        expert_packed.push(pack_q6_k(&w));
    }
    let (stacked_bytes, per_expert_bytes) = stack_expert_weights(&expert_packed);

    let mut ids: Vec<u32> = Vec::with_capacity(n_tokens * top_k);
    for s in 0..top_k {
        ids.push(((s * 17 + 7) % n_experts) as u32);
    }

    let total_rows = n_tokens * top_k;

    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens, k])
        .unwrap();
    input_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U8, vec![stacked_bytes.len()])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked_bytes);

    let mut ids_buf = device
        .alloc_buffer(ids.len() * u32_sz, DType::U32, vec![ids.len()])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    // Allocate 60 distinct output buffers so dispatches don't false-share
    // in a way the dependency tracker would block.
    const N_DISPATCHES_PER_TOKEN: usize = 60;
    let mut output_bufs: Vec<_> = (0..N_DISPATCHES_PER_TOKEN)
        .map(|_| {
            device
                .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows, n])
                .unwrap()
        })
        .collect();

    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::Q6_K,
    };

    // Warmup.
    const WARMUP: usize = 3;
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().unwrap();
        for d in 0..N_DISPATCHES_PER_TOKEN {
            mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
                &mut enc, &mut registry, &device,
                &input_buf, &weight_buf, &ids_buf, &mut output_bufs[d],
                &params,
            ).unwrap();
        }
        enc.commit_and_wait().unwrap();
    }

    // Measurement: one encoder per trial, 60 dispatches, one wait.
    const N_TRIALS: usize = 30;
    let mut cpu_total_us: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut gpu_total_us: Vec<f64> = Vec::with_capacity(N_TRIALS);
    for _ in 0..N_TRIALS {
        let mut enc = device.command_encoder().unwrap();
        let t0 = Instant::now();
        for d in 0..N_DISPATCHES_PER_TOKEN {
            mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
                &mut enc, &mut registry, &device,
                &input_buf, &weight_buf, &ids_buf, &mut output_bufs[d],
                &params,
            ).unwrap();
        }
        let (gpu_start_s, gpu_end_s) = enc.commit_wait_with_gpu_time().unwrap();
        let cpu_dt = t0.elapsed();
        cpu_total_us.push(cpu_dt.as_secs_f64() * 1e6);
        gpu_total_us.push((gpu_end_s - gpu_start_s) * 1e6);
    }

    cpu_total_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gpu_total_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_p50 = cpu_total_us[N_TRIALS / 2];
    let gpu_p50 = gpu_total_us[N_TRIALS / 2];

    let cpu_per_call = cpu_p50 / N_DISPATCHES_PER_TOKEN as f64;
    let gpu_per_call = gpu_p50 / N_DISPATCHES_PER_TOKEN as f64;
    let overhead_per_call = cpu_per_call - gpu_per_call;

    eprintln!("[BENCH] Q6_K mv_id session60 (60 dispatches in 1 encoder, gemma decode shape)");
    eprintln!("[BENCH]   per-token total: CPU = {:6.2} ms, GPU = {:6.2} ms",
              cpu_p50 / 1000.0, gpu_p50 / 1000.0);
    eprintln!("[BENCH]   per-call avg:    CPU = {:6.2} µs, GPU = {:6.2} µs",
              cpu_per_call, gpu_per_call);
    eprintln!("[BENCH]   amortized encode overhead per call = {:5.2} µs ({:.1}% of CPU)",
              overhead_per_call, overhead_per_call / cpu_per_call * 100.0);
}
