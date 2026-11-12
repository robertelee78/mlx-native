//! Target-side projections of one joint DFlash2 verify layer, model-free, at
//! the served Qwen shapes (hf2q Splash-gap program, DeltaProj stage).
//!
//! A verify layer multiplies one normalized activation tile (8-32 rows) by
//! several prepared Q4_K / Q6_K weights: a DeltaNet layer by alpha/beta, qkv
//! and the z gate, a full-attention layer by Q/gate, K and V. This bench
//! times those products of one layer three ways:
//!
//! - `grid`: one default prepared product per weight, as issued today. A
//!   weight with several K splits is a split-grid pass, a memory barrier and
//!   a reduce pass, so the products of a layer run in serialized stages.
//! - `wave`: one `dispatch_repacked_gguf_mm_bf16_input_one_pass` per weight
//!   and no barrier: the products overlap and fill the GPU jointly. Bitwise
//!   the `grid` outputs.
//! - `fused`: one prepared weight holding every product's rows (only when all
//!   share one GGML type), default schedule. Its K-split count is that of
//!   the fused shape, so its bits differ from the separate products'.
//!
//! Weights are kept DRAM-cold as in a serving cycle, where each layer's
//! weights are read once per cycle: a ring of layers of at least
//! `RING_BYTES` is walked, the sample reading the layer after the filler's.
//! Every sample is the `GPUEndTime - GPUStartTime` of a command buffer
//! committed right behind a filler command buffer of `FILLER_LAYERS` layers
//! of the same route, so the GPU is at the clocks of a serving cycle.
//!
//! `--sample-layers N` (1-4) puts N consecutive layers in the sampled command
//! buffer and reports the time per layer, which removes the command buffer's
//! own start and end from a layer's share. `fused1` is `fused` on the
//! one-pass schedule.
//!
//! Usage: `cargo bench --bench bench_qwen_delta_proj -- [--sets
//! qwen38-dn-q4,...] [--rows 8,16,32] [--routes grid,wave,fused,fused1]
//! [--repeat 20] [--sample-layers 1] [--json PATH]`

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::ops::repacked_gguf_mm::{
    dispatch_repacked_gguf_mm_bf16_input, dispatch_repacked_gguf_mm_bf16_input_one_pass,
    prepare_repacked_gguf_weight, PreparedGgufWeight,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

type R<T> = Result<T, Box<dyn std::error::Error>>;

const WARMUP_MS: u128 = 300;
/// Layers of GPU work queued ahead of every sample.
const FILLER_LAYERS: usize = 6;
/// Least prepared-weight bytes the ring holds (and at least
/// `FILLER_LAYERS + 5` layers).
const RING_BYTES: usize = 384 << 20;

const Q4: GgmlType = GgmlType::Q4_K;
const Q6: GgmlType = GgmlType::Q6_K;

/// The products of one layer that read one activation: `(name, N, type)`.
struct Set {
    name: &'static str,
    k: u32,
    products: &'static [(&'static str, u32, GgmlType)],
}

const SETS: &[Set] = &[
    Set {
        name: "qwen38-dn-q4",
        k: 5120,
        products: &[("ab", 128, Q4), ("qkv", 10240, Q4), ("gate", 6144, Q4)],
    },
    Set {
        name: "qwen38-dn-q6",
        k: 5120,
        products: &[("ab", 128, Q4), ("qkv", 10240, Q6), ("gate", 6144, Q4)],
    },
    Set {
        name: "qwen38-fa-q4",
        k: 5120,
        products: &[("qgate", 12288, Q4), ("k", 1024, Q4), ("v", 1024, Q4)],
    },
    Set {
        name: "qwen38-fa-q6",
        k: 5120,
        products: &[("qgate", 12288, Q4), ("k", 1024, Q4), ("v", 1024, Q6)],
    },
    Set {
        name: "qwen38-out",
        k: 6144,
        products: &[("out", 5120, Q4)],
    },
    Set {
        name: "qwen36-dn-q4",
        k: 2048,
        products: &[("ab", 64, Q4), ("qkv", 8192, Q4), ("gate", 4096, Q4)],
    },
    Set {
        name: "qwen36-dn-q6",
        k: 2048,
        products: &[("ab", 64, Q4), ("qkv", 8192, Q6), ("gate", 4096, Q4)],
    },
    Set {
        name: "qwen36-fa-q4",
        k: 2048,
        products: &[("qgate", 8192, Q4), ("k", 512, Q4), ("v", 512, Q4)],
    },
    Set {
        name: "qwen36-fa-q6",
        k: 2048,
        products: &[("qgate", 8192, Q4), ("k", 512, Q4), ("v", 512, Q6)],
    },
    Set {
        name: "qwen36-out",
        k: 4096,
        products: &[("out", 2048, Q4)],
    },
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn block_bytes(ggml_type: GgmlType) -> usize {
    ggml_type.block_bytes() as usize
}

/// Random finite GGML blocks `[n, k]`.
fn raw_weight(dev: &MlxDevice, n: u32, k: u32, ggml_type: GgmlType, rng: &mut Rng) -> R<MlxBuffer> {
    let bytes = n as usize * (k / 256) as usize * block_bytes(ggml_type);
    let mut weight = dev.alloc_buffer(bytes, DType::U8, vec![bytes])?;
    let scale = half::f16::from_f32(0.015625).to_le_bytes();
    for block in weight
        .as_mut_slice::<u8>()?
        .chunks_exact_mut(block_bytes(ggml_type))
    {
        for chunk in block.chunks_mut(8) {
            let word = rng.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        if ggml_type == Q4 {
            block[0..2].copy_from_slice(&scale);
            block[2..4].copy_from_slice(&half::f16::from_f32(0.0078125).to_le_bytes());
            for byte in &mut block[4..16] {
                *byte &= 0x3f;
            }
        } else {
            for byte in &mut block[192..208] {
                *byte = (*byte as i8).clamp(-31, 31) as u8;
            }
            block[208..210].copy_from_slice(&scale);
        }
    }
    Ok(weight)
}

struct Product {
    name: &'static str,
    params: GgmlQuantizedMatmulParams,
    prepared: PreparedGgufWeight,
    output: MlxBuffer,
}

/// One layer of the ring: the separate products and, when every product has
/// one type, the fused one.
struct Layer {
    products: Vec<Product>,
    fused: Option<Product>,
}

fn product(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    name: &'static str,
    raw: &MlxBuffer,
    rows: u32,
    n: u32,
    k: u32,
    ggml_type: GgmlType,
    output: &MlxBuffer,
) -> R<Product> {
    let params = GgmlQuantizedMatmulParams {
        m: rows,
        n,
        k,
        ggml_type,
    };
    let mut enc = dev.command_encoder()?;
    let prepared = prepare_repacked_gguf_weight(&mut enc, reg, dev, raw, &params)?;
    enc.commit_and_wait()?;
    Ok(Product {
        name,
        params,
        prepared,
        output: output.clone(),
    })
}

fn prepared_bytes(n: u32, k: u32, ggml_type: GgmlType) -> usize {
    // Planes of 16 (Q4_K) or 24 (Q6_K) bytes per 32 weights, plus metadata.
    n as usize * (k / 32) as usize * if ggml_type == Q4 { 18 } else { 27 }
}

struct Case {
    rows: u32,
    input: MlxBuffer,
    layers: Vec<Layer>,
    fused_splits: Option<u32>,
}

impl Case {
    fn new(dev: &MlxDevice, reg: &mut KernelRegistry, set: &Set, rows: u32) -> R<Self> {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ u64::from(rows) << 20 ^ u64::from(set.k));
        let elements = (rows * set.k) as usize;
        let mut input = dev.alloc_buffer(elements * 2, DType::BF16, vec![elements])?;
        for value in input.as_mut_slice::<half::bf16>()? {
            let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
            *value = half::bf16::from_f32(unit * 2.0 - 1.0);
        }
        let layer_bytes: usize = set
            .products
            .iter()
            .map(|&(_, n, ggml_type)| prepared_bytes(n, set.k, ggml_type))
            .sum();
        let ring = RING_BYTES.div_ceil(layer_bytes).max(FILLER_LAYERS + 5);
        let output = |n: u32| -> R<MlxBuffer> {
            let elements = (rows * n) as usize;
            Ok(dev.alloc_buffer(elements * 4, DType::F32, vec![elements])?)
        };
        let outputs = set
            .products
            .iter()
            .map(|&(_, n, _)| output(n))
            .collect::<R<Vec<_>>>()?;
        let fused_n: u32 = set.products.iter().map(|&(_, n, _)| n).sum();
        let one_type = set.products.len() > 1
            && set
                .products
                .iter()
                .all(|product| product.2 == set.products[0].2);
        let fused_output = output(fused_n)?;
        let mut layers = Vec::with_capacity(ring);
        for _ in 0..ring {
            let mut fused_raw = Vec::new();
            let mut products = Vec::with_capacity(set.products.len());
            for (&(name, n, ggml_type), output) in set.products.iter().zip(&outputs) {
                let raw = raw_weight(dev, n, set.k, ggml_type, &mut rng)?;
                if one_type {
                    fused_raw.extend_from_slice(raw.as_slice::<u8>()?);
                }
                products.push(product(
                    dev, reg, name, &raw, rows, n, set.k, ggml_type, output,
                )?);
            }
            let fused = if one_type {
                let mut raw =
                    dev.alloc_buffer(fused_raw.len(), DType::U8, vec![fused_raw.len()])?;
                raw.as_mut_slice::<u8>()?.copy_from_slice(&fused_raw);
                Some(product(
                    dev,
                    reg,
                    "fused",
                    &raw,
                    rows,
                    fused_n,
                    set.k,
                    set.products[0].2,
                    &fused_output,
                )?)
            } else {
                None
            };
            layers.push(Layer { products, fused });
        }
        let fused_splits = layers[0]
            .fused
            .as_ref()
            .map(|fused| fused.prepared.splits());
        Ok(Self {
            rows,
            input,
            layers,
            fused_splits,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Grid,
    Wave,
    Fused,
    FusedOnePass,
}

impl Route {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "grid" => Some(Self::Grid),
            "wave" => Some(Self::Wave),
            "fused" => Some(Self::Fused),
            "fused1" => Some(Self::FusedOnePass),
            _ => None,
        }
    }
}

/// Encode `products` (all of a layer, or one of them) on `route`; returns
/// the dispatch count.
fn encode(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    case: &Case,
    products: &[&Product],
    route: Route,
) -> R<u32> {
    let mut dispatches = 0;
    for product in products {
        match route {
            Route::Wave | Route::FusedOnePass => {
                dispatch_repacked_gguf_mm_bf16_input_one_pass(
                    enc,
                    reg,
                    dev,
                    &case.input,
                    &product.prepared,
                    &product.output,
                    &product.params,
                )?;
                dispatches += if case.rows % 32 == 0 || case.rows < 32 {
                    1
                } else {
                    2
                };
            }
            Route::Grid | Route::Fused => {
                dispatch_repacked_gguf_mm_bf16_input(
                    enc,
                    reg,
                    dev,
                    &case.input,
                    &product.prepared,
                    &product.output,
                    &product.params,
                )?;
                dispatches += if product.prepared.splits() > 1 { 2 } else { 1 };
            }
        }
    }
    Ok(dispatches)
}

fn layer_products(layer: &Layer, route: Route) -> Vec<&Product> {
    match route {
        Route::Fused | Route::FusedOnePass => layer.fused.iter().collect(),
        _ => layer.products.iter().collect(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// GPU ms per layer of `sample_layers` consecutive ring layers from `at`
/// (all products, or product `pick` of each) in their own command buffer,
/// right behind a filler of the next `FILLER_LAYERS` whole layers; and the
/// dispatch count of one layer.
fn hot_gpu_ms(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    case: &Case,
    route: Route,
    at: usize,
    pick: Option<usize>,
    sample_layers: usize,
) -> R<(f64, u32)> {
    let ring = case.layers.len();
    let mut filler = dev.command_encoder()?;
    for step in 0..FILLER_LAYERS {
        filler.memory_barrier();
        let layer = &case.layers[(at + sample_layers + step) % ring];
        encode(
            &mut filler,
            reg,
            dev,
            case,
            &layer_products(layer, route),
            route,
        )?;
    }
    filler.commit();
    let mut enc = dev.command_encoder()?;
    let mut dispatches = 0;
    for step in 0..sample_layers {
        if step > 0 {
            enc.memory_barrier();
        }
        let layer = &case.layers[(at + step) % ring];
        let all = layer_products(layer, route);
        let products = match pick {
            Some(index) => vec![all[index]],
            None => all,
        };
        dispatches = encode(&mut enc, reg, dev, case, &products, route)?;
    }
    let (start, end) = enc.commit_wait_with_gpu_time()?;
    filler.wait_until_completed()?;
    Ok(((end - start) * 1e3 / sample_layers as f64, dispatches))
}

fn list<T: std::str::FromStr + Clone>(args: &[String], flag: &str, default: &[T]) -> Vec<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let all_sets: Vec<String> = SETS.iter().map(|set| set.name.to_string()).collect();
    let sets: Vec<String> = list(&args, "--sets", &all_sets);
    let rows: Vec<u32> = list(&args, "--rows", &[32]);
    let routes: Vec<String> = list(
        &args,
        "--routes",
        &["grid".to_string(), "wave".to_string(), "fused".to_string()],
    );
    let repeat = list(&args, "--repeat", &[20usize])[0];
    let sample_layers = list(&args, "--sample-layers", &[1usize])[0].clamp(1, 4);
    let json_path: Option<String> = list::<String>(&args, "--json", &[]).into_iter().next();

    let dev = MlxDevice::new()?;
    let mut reg = KernelRegistry::new();
    println!(
        "device {}, input projections of one verify layer, DRAM-cold ring of at least {} MiB, median of {repeat}",
        dev.name(),
        RING_BYTES >> 20
    );
    let mut records = Vec::new();
    for set in SETS
        .iter()
        .filter(|set| sets.iter().any(|name| name == set.name))
    {
        for &row_count in &rows {
            let case = Case::new(&dev, &mut reg, set, row_count)?;
            for route_name in &routes {
                let route = Route::parse(route_name)
                    .ok_or_else(|| format!("unknown route {route_name}"))?;
                if matches!(route, Route::Fused | Route::FusedOnePass)
                    && case.layers[0].fused.is_none()
                {
                    continue;
                }
                let mut at = 0usize;
                let mut sample = |reg: &mut KernelRegistry, pick| -> R<(f64, u32)> {
                    at += FILLER_LAYERS + sample_layers;
                    hot_gpu_ms(&dev, reg, &case, route, at, pick, sample_layers)
                };
                let warm = std::time::Instant::now();
                while warm.elapsed().as_millis() < WARMUP_MS {
                    sample(&mut reg, None)?;
                }
                let mut whole = Vec::with_capacity(repeat);
                let mut dispatches = 0;
                for _ in 0..repeat {
                    let (ms, count) = sample(&mut reg, None)?;
                    whole.push(ms);
                    dispatches = count;
                }
                let names: Vec<&str> = layer_products(&case.layers[0], route)
                    .iter()
                    .map(|product| product.name)
                    .collect();
                let mut alone = Vec::new();
                if names.len() > 1 {
                    for (index, name) in names.iter().enumerate() {
                        let mut values = Vec::with_capacity(repeat);
                        for _ in 0..repeat {
                            values.push(sample(&mut reg, Some(index))?.0);
                        }
                        alone.push((*name, median(&mut values)));
                    }
                }
                let min = whole.iter().copied().fold(f64::INFINITY, f64::min);
                let layer_ms = median(&mut whole);
                let splits: Vec<String> = layer_products(&case.layers[0], route)
                    .iter()
                    .map(|product| format!("{}:S{}", product.name, product.prepared.splits()))
                    .collect();
                let alone_text: Vec<String> = alone
                    .iter()
                    .map(|(name, ms)| format!("{name} {ms:.4}"))
                    .collect();
                println!(
                    "{:<13} rows={:<2} {:<5} layer={:.4} ms (min {:.4}) dispatches={} ring={} [{}]{}",
                    set.name,
                    row_count,
                    route_name,
                    layer_ms,
                    min,
                    dispatches,
                    case.layers.len(),
                    splits.join(" "),
                    if alone_text.is_empty() {
                        String::new()
                    } else {
                        format!(" alone: {}", alone_text.join(", "))
                    }
                );
                records.push(format!(
                    "{{\"set\":\"{}\",\"rows\":{},\"route\":\"{}\",\"layer_ms\":{:.6},\"min_ms\":{:.6},\"dispatches\":{},\"ring\":{},\"alone\":{{{}}}}}",
                    set.name,
                    row_count,
                    route_name,
                    layer_ms,
                    min,
                    dispatches,
                    case.layers.len(),
                    alone
                        .iter()
                        .map(|(name, ms)| format!("\"{name}\":{ms:.6}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            if let Some(splits) = case.fused_splits {
                println!(
                    "{:<13} rows={:<2} fused weight K splits: {splits}",
                    set.name, row_count
                );
            }
        }
    }
    if let Some(path) = json_path {
        std::fs::write(&path, format!("[\n{}\n]\n", records.join(",\n")))?;
        println!("wrote {path}");
    }
    Ok(())
}
