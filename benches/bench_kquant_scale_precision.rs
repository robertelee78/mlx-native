//! Cost of evaluating the tile dequantizers' high-nibble scale in F32
//! (`float(xb->d) / 16.f`) instead of F16 (`xb->d / 16.h`).
//!
//! One process holds three registries whose affected shader files are
//! compiled from runtime source with identical options:
//!
//! * `half` — the shipped source with the F16 division put back (the text
//!   substitution is counted and must hit every site);
//! * `f32`  — the shipped source;
//! * `f32b` — the shipped source again, as the A/A control: the ratio
//!   `f32b / f32` is the resolution of the comparison.
//!
//! Every sample is committed right behind a filler command buffer holding
//! six repeats of the same product (filler pipelines come from the `f32b`
//! registry for all three variants), so the GPU is at a serving clock. The
//! variants alternate sample by sample in a rotating order, over a ring of
//! distinct weight buffers. Block scales are in the real-file range
//! (`|d|` of 2e-5 to 3e-4; `tests/kquant_real_scale_f64_reference.rs`).
//!
//! Usage: `bench_kquant_scale_precision [--repeat 15] [--json PATH]
//!         [--only SUBSTRING]`

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use half::f16;
use mlx_native::ops::dequant_to_f16::dispatch_dequant_to_f16;
use mlx_native::{
    dispatch_mm_q4_0_tensor_64x32_for_test, quantized_matmul_ggml_column_invariant_mma,
    quantized_matmul_ggml_with_policy, quantized_matmul_id_ggml_with_policy,
    quantized_matmul_mm_tensor_perm021, CommandEncoder, DType, GgmlQuantizedMatmulIdParams,
    GgmlQuantizedMatmulParams, GgmlQuantizedMatmulPerm021Params, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

use std::sync::LazyLock;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const FILLER: usize = 6;
const WARMUP: usize = 3;
const VARIANTS: [&str; 3] = ["half", "f32", "f32b"];

/// Affected shader files and the number of scale sites in each.
const SHADERS: [(&str, usize); 7] = [
    (include_str!("../src/shaders/quantized_matmul_mm.metal"), 3),
    (
        include_str!("../src/shaders/quantized_matmul_mm_tensor.metal"),
        3,
    ),
    (
        include_str!("../src/shaders/quantized_matmul_id_mm.metal"),
        3,
    ),
    (
        include_str!("../src/shaders/quantized_matmul_id_mm_tensor.metal"),
        3,
    ),
    (include_str!("../src/shaders/mul_mv_ext.metal"), 3),
    (include_str!("../src/shaders/mul_mv_mma.metal"), 1),
    (include_str!("../src/shaders/dequant_to_f16.metal"), 1),
];

/// (fixed text, shipped-before text) of the two forms of the expression.
const SITES: [(&str, &str); 2] = [
    (
        "il < 2 ? float(xb->d) : float(xb->d) / 16.f;",
        "il < 2 ? xb->d : xb->d / 16.h;",
    ),
    (
        "? (float(xb->d) / 16.f) : float(xb->d);",
        "? (xb->d / 16.h) : xb->d;",
    ),
];

/// Every kernel entry point a shader file defines.
fn kernel_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for marker in ["host_name(\"", "kernel void "] {
        let mut rest = source;
        while let Some(at) = rest.find(marker) {
            rest = &rest[at + marker.len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            if end > 0 && !names.iter().any(|n| n == &rest[..end]) {
                names.push(rest[..end].to_string());
            }
        }
    }
    names
}

/// The affected shader files as each variant compiles them: `[variant][file]`.
static SOURCES: LazyLock<Vec<Vec<String>>> = LazyLock::new(|| {
    VARIANTS
        .iter()
        .map(|variant| {
            SHADERS
                .iter()
                .map(|&(source, sites)| {
                    let mut text = source.to_string();
                    let found: usize = SITES
                        .iter()
                        .map(|(fixed, _)| text.matches(fixed).count())
                        .sum();
                    assert_eq!(found, sites, "scale sites in a shader file changed");
                    if *variant == "half" {
                        for (fixed, before) in SITES {
                            text = text.replace(fixed, before);
                        }
                    }
                    // A distinct trailing comment makes every variant a
                    // runtime-source override, so none of them resolves from
                    // the embedded metallib.
                    text.push_str(&format!("\n// bench variant: {variant}\n"));
                    text
                })
                .collect()
        })
        .collect()
});

/// A registry whose affected kernels compile from the variant's source.
fn registry(variant: usize) -> KernelRegistry {
    let mut registry = KernelRegistry::new();
    for (file, &(source, _)) in SHADERS.iter().enumerate() {
        let text: &'static str = SOURCES[variant][file].as_str();
        for name in kernel_names(source) {
            registry.register_source(name, text);
        }
    }
    registry
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

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn weight(
    dev: &MlxDevice,
    ggml_type: GgmlType,
    rows: usize,
    k: usize,
    rng: &mut Rng,
) -> R<MlxBuffer> {
    let block = ggml_type.block_bytes() as usize;
    let bytes = rows * k / ggml_type.block_values() as usize * block;
    let mut buffer = dev.alloc_buffer(bytes, DType::U8, vec![bytes])?;
    let (low, high) = if ggml_type == GgmlType::Q4_0 {
        (6.0e-5, 9.0e-4)
    } else {
        (2.0e-5f64, 3.0e-4f64)
    };
    for block in buffer.as_mut_slice::<u8>()?.chunks_exact_mut(block) {
        for chunk in block.chunks_mut(8) {
            let word = rng.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        let d = f16::from_f64(low * (high / low).powf(rng.unit()));
        block[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        if ggml_type != GgmlType::Q4_0 {
            let dmin = f16::from_f64(d.to_f64() * (5.0 + 5.0 * rng.unit()));
            block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        }
    }
    Ok(buffer)
}

fn activations(dev: &MlxDevice, rows: usize, k: usize, rng: &mut Rng) -> R<MlxBuffer> {
    let mut buffer = dev.alloc_buffer(rows * k * 4, DType::F32, vec![rows, k])?;
    for value in buffer.as_mut_slice::<f32>()? {
        *value = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
    }
    Ok(buffer)
}

struct Bench {
    registries: Vec<KernelRegistry>,
    repeat: usize,
    only: Option<String>,
    records: Vec<String>,
}

impl Bench {
    /// Times `encode(encoder, registry, ring index)` under the three
    /// variants and records one row.
    fn cell(
        &mut self,
        dev: &MlxDevice,
        label: &str,
        ring: usize,
        mut encode: impl FnMut(&mut CommandEncoder, &mut KernelRegistry, &MlxDevice, usize) -> R<()>,
    ) -> R<()> {
        if self
            .only
            .as_ref()
            .is_some_and(|only| !label.contains(only.as_str()))
        {
            return Ok(());
        }
        let mut samples = vec![Vec::new(); VARIANTS.len()];
        for round in 0..self.repeat + WARMUP {
            for turn in 0..VARIANTS.len() {
                let variant = (round + turn) % VARIANTS.len();
                let at = (round * VARIANTS.len() + turn) % ring;
                let mut filler = dev.command_encoder()?;
                for step in 0..FILLER {
                    filler.memory_barrier();
                    encode(
                        &mut filler,
                        &mut self.registries[2],
                        dev,
                        (at + 1 + step) % ring,
                    )?;
                }
                filler.commit();
                let mut encoder = dev.command_encoder()?;
                encoder.memory_barrier();
                encode(&mut encoder, &mut self.registries[variant], dev, at)?;
                let (start, end) = encoder.commit_wait_with_gpu_time()?;
                filler.wait_until_completed()?;
                if round >= WARMUP {
                    samples[variant].push((end - start) * 1e3);
                }
            }
        }
        let paired = |a: usize, b: usize| {
            let mut ratios: Vec<f64> = samples[a]
                .iter()
                .zip(&samples[b])
                .map(|(x, y)| x / y)
                .collect();
            ratios.sort_by(|x, y| x.partial_cmp(y).unwrap());
            ratios[ratios.len() / 2]
        };
        let (fix_ratio, aa_ratio) = (paired(1, 0), paired(2, 1));
        let quantiles: Vec<[f64; 3]> = samples
            .iter_mut()
            .map(|values| {
                values.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let q = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
                [q(0.5), q(0.1), q(0.9)]
            })
            .collect();
        println!(
            "{label:<44} half {:.4} [{:.4}, {:.4}]  f32 {:.4} [{:.4}, {:.4}]  f32b {:.4} [{:.4}, {:.4}] ms  f32/half {:.3} (paired {fix_ratio:.3})  f32b/f32 {:.3} (paired {aa_ratio:.3})",
            quantiles[0][0], quantiles[0][1], quantiles[0][2],
            quantiles[1][0], quantiles[1][1], quantiles[1][2],
            quantiles[2][0], quantiles[2][1], quantiles[2][2],
            quantiles[1][0] / quantiles[0][0],
            quantiles[2][0] / quantiles[1][0],
        );
        let field = |q: &[f64; 3]| {
            format!(
                "{{\"median_ms\":{:.5},\"p10_ms\":{:.5},\"p90_ms\":{:.5}}}",
                q[0], q[1], q[2]
            )
        };
        self.records.push(format!(
            "{{\"cell\":\"{label}\",\"half\":{},\"f32\":{},\"f32b\":{},\"paired_f32_over_half\":{fix_ratio:.4},\"paired_f32b_over_f32\":{aa_ratio:.4}}}",
            field(&quantiles[0]), field(&quantiles[1]), field(&quantiles[2]),
        ));
        Ok(())
    }
}

fn tag(ggml_type: GgmlType) -> &'static str {
    match ggml_type {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_K => "q5_K",
        _ => "q4_0",
    }
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let mut bench = Bench {
        registries: (0..VARIANTS.len()).map(registry).collect(),
        repeat: arg("--repeat").and_then(|v| v.parse().ok()).unwrap_or(15),
        only: arg("--only"),
        records: Vec::new(),
    };
    let dev = MlxDevice::new()?;
    let mut rng = Rng(0x5ca1_e16);
    let codecs = [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q4_0];

    let default_policy = GgmlRoutingPolicy::default();
    let mv_ext = GgmlRoutingPolicy {
        dense_decode_mvn: false,
        dense_decode_mv_ext: true,
        ..default_policy
    };
    let simd = GgmlRoutingPolicy {
        dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..default_policy
    };

    // Dense products: FFN gate of the 27B dense model, DeltaNet qkv of the
    // 35B MoE model.
    const RING: usize = 4;
    for (shape, n, k) in [
        ("17408x5120", 17408usize, 5120usize),
        ("8192x2048", 8192, 2048),
    ] {
        for ggml_type in codecs {
            let weights = (0..RING)
                .map(|_| weight(&dev, ggml_type, n, k, &mut rng))
                .collect::<R<Vec<_>>>()?;
            let input = activations(&dev, 720, k, &mut rng)?;
            let output = dev.alloc_buffer(720 * n * 4, DType::F32, vec![720, n])?;
            let mut routes: Vec<(&str, GgmlRoutingPolicy, Vec<u32>)> = vec![
                ("tensor_v1", default_policy, vec![32]),
                ("tensor_v2", default_policy, vec![160, 720]),
                ("simdgroup", simd, vec![32, 160, 720]),
            ];
            routes.insert(
                0,
                (
                    "mul_mv_ext",
                    mv_ext,
                    match ggml_type {
                        GgmlType::Q5_K => vec![1, 4, 8],
                        _ => vec![4, 8],
                    },
                ),
            );
            for (route, policy, rows) in routes {
                for m in rows {
                    let params = GgmlQuantizedMatmulParams {
                        m,
                        n: n as u32,
                        k: k as u32,
                        ggml_type,
                    };
                    let view = input.slice_view(0, m as usize * k);
                    bench.cell(
                        &dev,
                        &format!("{route} {} {shape} rows={m}", tag(ggml_type)),
                        RING,
                        |encoder, registry, dev, at| {
                            Ok(quantized_matmul_ggml_with_policy(
                                encoder,
                                registry,
                                dev,
                                &view,
                                &weights[at],
                                &output,
                                &params,
                                &policy,
                            )?)
                        },
                    )?;
                }
            }
            if ggml_type == GgmlType::Q4_K {
                for m in [1u32, 4, 8] {
                    let params = GgmlQuantizedMatmulParams {
                        m,
                        n: n as u32,
                        k: k as u32,
                        ggml_type,
                    };
                    let view = input.slice_view(0, m as usize * k);
                    bench.cell(
                        &dev,
                        &format!("mul_mv_mma q4_K {shape} rows={m}"),
                        RING,
                        |encoder, registry, dev, at| {
                            Ok(quantized_matmul_ggml_column_invariant_mma(
                                encoder,
                                registry,
                                dev,
                                &view,
                                &weights[at],
                                &output,
                                &params,
                            )?)
                        },
                    )?;
                }
            }
            if ggml_type == GgmlType::Q4_0 {
                let params = GgmlQuantizedMatmulParams {
                    m: 160,
                    n: n as u32,
                    k: k as u32,
                    ggml_type,
                };
                let view = input.slice_view(0, 160 * k);
                bench.cell(
                    &dev,
                    &format!("tensor_64x32 q4_0 {shape} rows=160"),
                    RING,
                    |encoder, registry, dev, at| {
                        Ok(dispatch_mm_q4_0_tensor_64x32_for_test(
                            encoder,
                            registry,
                            dev,
                            &view,
                            &weights[at],
                            &output,
                            &params,
                        )?)
                    },
                )?;
            }
        }
    }

    // Q4_0 only: the F16 shadow and the BF16 head-major tensor kernel.
    {
        let (n, k) = (8192usize, 2048usize);
        let weights = (0..RING)
            .map(|_| weight(&dev, GgmlType::Q4_0, n, k, &mut rng))
            .collect::<R<Vec<_>>>()?;
        let shadow = dev.alloc_buffer(n * k * 2, DType::F16, vec![n, k])?;
        bench.cell(
            &dev,
            "dequant_to_f16 q4_0 8192x2048",
            RING,
            |encoder, registry, dev, at| {
                Ok(dispatch_dequant_to_f16(
                    encoder,
                    registry,
                    dev.metal_device(),
                    &weights[at],
                    &shadow,
                    n as u32,
                    k as u32,
                    GgmlType::Q4_0,
                )?)
            },
        )?;
        const HEAD_DIM: usize = 256;
        for m in [160usize, 720] {
            let mut input =
                dev.alloc_buffer(m * k * 2, DType::BF16, vec![k / HEAD_DIM, m, HEAD_DIM])?;
            for value in input.as_mut_slice::<u16>()? {
                let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                *value = half::bf16::from_f32(unit).to_bits();
            }
            let output = dev.alloc_buffer(m * n * 4, DType::F32, vec![m, n])?;
            let params = GgmlQuantizedMatmulPerm021Params {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                head_dim: HEAD_DIM as u32,
                ggml_type: GgmlType::Q4_0,
            };
            bench.cell(
                &dev,
                &format!("tensor_bf16_perm021 q4_0 8192x2048 rows={m}"),
                RING,
                |encoder, registry, dev, at| {
                    Ok(quantized_matmul_mm_tensor_perm021(
                        encoder,
                        registry,
                        dev,
                        &input,
                        &weights[at],
                        &output,
                        &params,
                    )?)
                },
            )?;
        }
    }

    // Expert products: the 35B MoE model's gate (256 experts, top-8,
    // 2048 -> 512).
    const EXPERTS: usize = 256;
    const TOP_K: usize = 8;
    const EXPERT_RING: usize = 2;
    let (n, k) = (512usize, 2048usize);
    for ggml_type in codecs {
        let weights = (0..EXPERT_RING)
            .map(|_| weight(&dev, ggml_type, EXPERTS * n, k, &mut rng))
            .collect::<R<Vec<_>>>()?;
        let input = activations(&dev, 720, k, &mut rng)?;
        let output = dev.alloc_buffer(720 * TOP_K * n * 4, DType::F32, vec![720 * TOP_K, n])?;
        let mut ids = dev.alloc_buffer(720 * TOP_K * 4, DType::U32, vec![720, TOP_K])?;
        for token in ids.as_mut_slice::<u32>()?.chunks_exact_mut(TOP_K) {
            let mut filled = 0;
            while filled < TOP_K {
                let expert = (rng.next() % EXPERTS as u64) as u32;
                if !token[..filled].contains(&expert) {
                    token[filled] = expert;
                    filled += 1;
                }
            }
        }
        let expert_stride =
            (n * k / ggml_type.block_values() as usize * ggml_type.block_bytes() as usize) as u64;
        for (route, preference) in [
            ("mm_id_tensor_c", GgmlTensorMmPreference::AutoProbe),
            ("mm_id_simdgroup_c", GgmlTensorMmPreference::ForceSimd),
        ] {
            let policy = GgmlRoutingPolicy {
                expert_tensor_mm: preference,
                ..default_policy
            };
            for tokens in [40usize, 160, 720] {
                let params = GgmlQuantizedMatmulIdParams {
                    n_tokens: tokens as u32,
                    top_k: TOP_K as u32,
                    n: n as u32,
                    k: k as u32,
                    n_experts: EXPERTS as u32,
                    expert_stride,
                    ggml_type,
                };
                let input_view = input.slice_view(0, tokens * k);
                let ids_view = ids.slice_view(0, tokens * TOP_K);
                let output_view = output.slice_view(0, tokens * TOP_K * n);
                bench.cell(
                    &dev,
                    &format!(
                        "{route} {} 256x512x2048 top8 tokens={tokens}",
                        tag(ggml_type)
                    ),
                    EXPERT_RING,
                    |encoder, registry, dev, at| {
                        Ok(quantized_matmul_id_ggml_with_policy(
                            encoder,
                            registry,
                            dev,
                            &input_view,
                            &weights[at],
                            &ids_view,
                            &output_view,
                            &params,
                            &policy,
                        )?)
                    },
                )?;
            }
        }
    }

    if let Some(path) = arg("--json") {
        std::fs::write(path, format!("[\n{}\n]\n", bench.records.join(",\n")))?;
    }
    Ok(())
}
