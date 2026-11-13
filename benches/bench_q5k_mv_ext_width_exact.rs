//! Cost of the width-exact Q5_K `mul_mv_ext` accumulation (explicit fma
//! lanes) against the implicit `dot + dot + dot + dot` form it replaced.
//!
//! One process, three registries whose `mul_mv_ext.metal` kernels compile
//! from runtime source with identical options: `base` (a baseline copy of
//! the shader file given on the command line), `exact` (the shipped source)
//! and `exact_b` (the shipped source again, the A/A control). Samples
//! alternate in a rotating order, each right behind a filler command buffer
//! of six repeats of the same product, over a ring of distinct weights with
//! real-file block scales.
//!
//! ```text
//! git show <baseline-rev>:src/shaders/mul_mv_ext.metal > /tmp/mul_mv_ext.base.metal
//! bench_q5k_mv_ext_width_exact --baseline-shader /tmp/mul_mv_ext.base.metal [--repeat 60] [--json PATH]
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use half::f16;
use mlx_native::{
    quantized_matmul_ggml_with_policy, CommandEncoder, DType, GgmlQuantizedMatmulParams,
    GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};
use std::sync::OnceLock;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const FILLER: usize = 6;
const WARMUP: usize = 3;
const VARIANTS: [&str; 3] = ["base", "exact", "exact_b"];
const SHIPPED: &str = include_str!("../src/shaders/mul_mv_ext.metal");

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

/// The shader text each variant compiles.
static SOURCES: OnceLock<Vec<String>> = OnceLock::new();

/// A registry whose `mul_mv_ext` kernels compile from the variant's source.
fn registry(variant: usize) -> KernelRegistry {
    let mut registry = KernelRegistry::new();
    let text: &'static str = SOURCES.get().expect("sources")[variant].as_str();
    for name in kernel_names(SHIPPED) {
        registry.register_source(name, text);
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
            "{label:<44} base {:.4} [{:.4}, {:.4}]  exact {:.4} [{:.4}, {:.4}]  exact_b {:.4} [{:.4}, {:.4}] ms  exact/base {:.3} (paired {fix_ratio:.3})  exact_b/exact {:.3} (paired {aa_ratio:.3})",
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
            "{{\"cell\":\"{label}\",\"base\":{},\"exact\":{},\"exact_b\":{},\"paired_exact_over_base\":{fix_ratio:.4},\"paired_exact_b_over_exact\":{aa_ratio:.4}}}",
            field(&quantiles[0]), field(&quantiles[1]), field(&quantiles[2]),
        ));
        Ok(())
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
    let baseline = std::fs::read_to_string(
        arg("--baseline-shader").ok_or("--baseline-shader PATH is required")?,
    )?;
    assert_ne!(
        baseline, SHIPPED,
        "the baseline shader equals the shipped one"
    );
    // A distinct trailing comment makes every variant a runtime-source
    // override, so none of them resolves from the embedded metallib.
    let sources = [baseline.as_str(), SHIPPED, SHIPPED]
        .iter()
        .zip(VARIANTS)
        .map(|(text, variant)| format!("{text}\n// bench variant: {variant}\n"))
        .collect();
    SOURCES.set(sources).map_err(|_| "sources set twice")?;
    let mut bench = Bench {
        registries: (0..VARIANTS.len()).map(registry).collect(),
        repeat: arg("--repeat").and_then(|v| v.parse().ok()).unwrap_or(60),
        only: arg("--only"),
        records: Vec::new(),
    };
    let dev = MlxDevice::new()?;
    let mut rng = Rng(0x5ca1_e16);
    let policy = GgmlRoutingPolicy::default();
    const RING: usize = 4;
    for (shape, n, k) in [
        ("17408x5120", 17408usize, 5120usize),
        ("8192x2048", 8192, 2048),
    ] {
        let ggml_type = GgmlType::Q5_K;
        let weights = (0..RING)
            .map(|_| weight(&dev, ggml_type, n, k, &mut rng))
            .collect::<R<Vec<_>>>()?;
        let input = activations(&dev, 8, k, &mut rng)?;
        let output = dev.alloc_buffer(8 * n * 4, DType::F32, vec![8, n])?;
        for m in [1u32, 2, 3, 4, 5, 8] {
            let params = GgmlQuantizedMatmulParams {
                m,
                n: n as u32,
                k: k as u32,
                ggml_type,
            };
            let view = input.slice_view(0, m as usize * k);
            bench.cell(
                &dev,
                &format!("mul_mv_ext q5_K {shape} rows={m}"),
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
    if let Some(path) = arg("--json") {
        std::fs::write(path, format!("[\n{}\n]\n", bench.records.join(",\n")))?;
    }
    Ok(())
}
