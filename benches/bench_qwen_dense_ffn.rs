//! Dense FFN layer and vocabulary head of a Qwen joint verify step over
//! prepared Q4_K / Q6_K weights, model-free, at the served shapes (hf2q
//! Splash-gap program).
//!
//! FFN (Qwen3.8-27B: hidden 5120, intermediate 17408, gate/up Q4_K, down Q4_K
//! or Q6_K) at `--rows` activation rows, as hf2q's prepared joint step encodes
//! it: F32 -> BF16 cast of the input, gate and up as two prepared
//! projections, `silu_mul_f32`, F32 -> BF16 cast of the product, the down
//! projection (split K and its reduce), the residual add.
//!
//! Head (`[248320, K]` Q6_K): the stock GGML matmul, the prepared projection
//! with its input cast, and the per-row argmax dispatches.
//!
//! Every sample is the `GPUEndTime - GPUStartTime` of a command buffer
//! queued between a filler command buffer of six FFN layers and a spin
//! command buffer of twelve, so the GPU queue never drains and the GPU stays
//! in its top performance state (as in Splash's benchmarks, whose backend
//! spins the GPU). A serving cycle with host waits runs below these clocks. Per route it prints the fused
//! layer (one command buffer, the layer's own barriers), the per-dispatch-kind
//! sums with every kind submitted as its own command buffer, and a fused
//! command buffer of 64 layers (the FFN of one Qwen3.8 verify cycle). FFN
//! weights are a ring of `--ring` prepared copies, each sample reading the
//! next, so they stream from DRAM as a model's layers do.
//!
//! A one-pass gate/up pair writing the BF16 product, a residual folded into
//! the down projection's reduce and a single row-batched argmax were measured
//! with this bench and rejected: none was faster on the GPU.
//!
//! Usage: `cargo bench --bench bench_qwen_dense_ffn -- [--rows 8,16,24,32]
//! [--repeat 20] [--ring 6] [--down q4k,q6k] [--head-k 5120,2048]
//! [--parts ffn,head]`

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::rc::Rc;

use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::ops::elementwise::{cast, elementwise_add, CastDirection};
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::ops::repacked_gguf_mm::{
    dispatch_repacked_gguf_mm_bf16_input, prepare_repacked_gguf_weight, PreparedGgufWeight,
};
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

type R<T> = Result<T, Box<dyn std::error::Error>>;
type Encode = Box<dyn Fn(&mut CommandEncoder, &mut KernelRegistry, &MlxDevice) -> R<()>>;

const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 17_408;
const VOCAB: usize = 248_320;
const FILLER_LAYERS: usize = 6;
const CYCLE_LAYERS: usize = 64;
const SEPARATE_REPEAT: usize = 5;

struct Step {
    kind: &'static str,
    /// A memory barrier precedes the step.
    barrier: bool,
    encode: Encode,
}

fn step(kind: &'static str, barrier: bool, encode: Encode) -> Step {
    Step {
        kind,
        barrier,
        encode,
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 32_768.0 - 1.0
    }
}

fn alloc(dev: &MlxDevice, dtype: DType, shape: &[usize]) -> R<MlxBuffer> {
    let elements: usize = shape.iter().product();
    Ok(dev.alloc_buffer(elements * dtype.size_of(), dtype, shape.to_vec())?)
}

fn f32_rows(dev: &MlxDevice, rows: usize, width: usize, rng: &mut Rng) -> R<MlxBuffer> {
    let mut buffer = alloc(dev, DType::F32, &[rows, width])?;
    for value in buffer.as_mut_slice::<f32>()? {
        *value = rng.unit();
    }
    Ok(buffer)
}

/// Synthetic `[n, k]` Q4_K or Q6_K blocks with small finite weights.
fn weight(dev: &MlxDevice, ggml_type: GgmlType, n: usize, k: usize, salt: usize) -> R<MlxBuffer> {
    let block_bytes = ggml_type.block_bytes() as usize;
    let bytes = n * (k / 256) * block_bytes;
    let mut buffer = dev.alloc_buffer(bytes, DType::U8, vec![bytes])?;
    for (block, bytes) in buffer
        .as_mut_slice::<u8>()?
        .chunks_exact_mut(block_bytes)
        .enumerate()
    {
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = ((block * 31 + i * 13 + salt) & 255) as u8;
        }
        match ggml_type {
            GgmlType::Q4_K => {
                bytes[0..2].copy_from_slice(&half::f16::from_f32(0.000_976_562_5).to_le_bytes());
                bytes[2..4].copy_from_slice(&half::f16::from_f32(0.000_488_281_25).to_le_bytes());
                for scale in &mut bytes[4..16] {
                    *scale &= 0x3f;
                }
            }
            GgmlType::Q6_K => {
                for scale in &mut bytes[192..208] {
                    *scale = (*scale as i8).clamp(-31, 31) as u8;
                }
                bytes[208..210].copy_from_slice(&half::f16::from_f32(0.000_061_035).to_le_bytes());
            }
            _ => unreachable!(),
        }
    }
    Ok(buffer)
}

fn params(rows: usize, n: usize, k: usize, ggml_type: GgmlType) -> GgmlQuantizedMatmulParams {
    GgmlQuantizedMatmulParams {
        m: rows as u32,
        n: n as u32,
        k: k as u32,
        ggml_type,
    }
}

fn prepare(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    source: &MlxBuffer,
    p: &GgmlQuantizedMatmulParams,
) -> R<PreparedGgufWeight> {
    let mut enc = dev.command_encoder()?;
    let prepared = prepare_repacked_gguf_weight(&mut enc, reg, dev, source, p)?;
    enc.commit_and_wait()?;
    Ok(prepared)
}

/// One FFN layer's prepared weights and the step's scratch.
struct Layer {
    gate: PreparedGgufWeight,
    up: PreparedGgufWeight,
    down: PreparedGgufWeight,
}

struct Scratch {
    x: MlxBuffer,
    residual: MlxBuffer,
    x_bf16: MlxBuffer,
    gate: MlxBuffer,
    up: MlxBuffer,
    hidden: MlxBuffer,
    hidden_bf16: MlxBuffer,
    down: MlxBuffer,
    sum: MlxBuffer,
    silu_params: MlxBuffer,
}

fn separate_route(
    layer: &Rc<Layer>,
    s: &Rc<Scratch>,
    rows: usize,
    down_type: GgmlType,
) -> Vec<Step> {
    let gate_up = params(rows, INTERMEDIATE, HIDDEN, GgmlType::Q4_K);
    let down = params(rows, HIDDEN, INTERMEDIATE, down_type);
    let projection = |kind, barrier, which: fn(&Layer) -> &PreparedGgufWeight,
                      input: fn(&Scratch) -> &MlxBuffer,
                      output: fn(&Scratch) -> &MlxBuffer,
                      p: GgmlQuantizedMatmulParams| {
        let (layer, s) = (layer.clone(), s.clone());
        step(
            kind,
            barrier,
            Box::new(move |enc, reg, dev| {
                Ok(dispatch_repacked_gguf_mm_bf16_input(
                    enc,
                    reg,
                    dev,
                    input(&s),
                    which(&layer),
                    output(&s),
                    &p,
                )?)
            }),
        )
    };
    let cast_step = |kind, input: fn(&Scratch) -> &MlxBuffer,
                     output: fn(&Scratch) -> &MlxBuffer,
                     elements: usize| {
        let s = s.clone();
        step(
            kind,
            true,
            Box::new(move |enc, reg, dev| {
                Ok(cast(
                    enc,
                    reg,
                    dev.metal_device(),
                    input(&s),
                    output(&s),
                    elements,
                    CastDirection::F32ToBF16,
                )?)
            }),
        )
    };
    let silu = {
        let s = s.clone();
        step(
            "silu_mul",
            true,
            Box::new(move |enc, reg, dev| {
                Ok(dispatch_silu_mul(
                    enc,
                    reg,
                    dev.metal_device(),
                    &s.gate,
                    &s.up,
                    &s.hidden,
                    &s.silu_params,
                    (rows * INTERMEDIATE) as u32,
                )?)
            }),
        )
    };
    let add = {
        let s = s.clone();
        step(
            "residual_add",
            true,
            Box::new(move |enc, reg, dev| {
                Ok(elementwise_add(
                    enc,
                    reg,
                    dev.metal_device(),
                    &s.down,
                    &s.residual,
                    &s.sum,
                    rows * HIDDEN,
                    DType::F32,
                )?)
            }),
        )
    };
    vec![
        cast_step("cast_input", |s| &s.x, |s| &s.x_bf16, rows * HIDDEN),
        projection("gate", true, |l| &l.gate, |s| &s.x_bf16, |s| &s.gate, gate_up),
        projection("up", false, |l| &l.up, |s| &s.x_bf16, |s| &s.up, gate_up),
        silu,
        cast_step(
            "cast_hidden",
            |s| &s.hidden,
            |s| &s.hidden_bf16,
            rows * INTERMEDIATE,
        ),
        projection(
            "down+reduce",
            true,
            |l| &l.down,
            |s| &s.hidden_bf16,
            |s| &s.down,
            down,
        ),
        add,
    ]
}

fn encode_steps(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    steps: &[Step],
) -> R<()> {
    for step in steps {
        if step.barrier {
            enc.memory_barrier();
        }
        (step.encode)(enc, reg, dev)?;
    }
    Ok(())
}

/// GPU time of `measure` in its own command buffer, queued between a filler
/// command buffer of `FILLER_LAYERS` layers and a spin command buffer of
/// twice as many. The spin keeps the queue from draining while the host
/// waits for the sample and encodes the next one: with idle gaps between
/// samples the GPU leaves its top performance state and the same projection
/// drifts from 0.118 ms to 0.18 ms over a few seconds.
fn sample(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    spin: &mut Option<CommandEncoder>,
    filler_layers: &[&[Step]],
    measure: &mut dyn FnMut(&mut CommandEncoder, &mut KernelRegistry) -> R<()>,
) -> R<f64> {
    let mut filler = dev.command_encoder()?;
    let mut next_spin = dev.command_encoder()?;
    for layer in filler_layers {
        filler.memory_barrier();
        encode_steps(&mut filler, reg, dev, layer)?;
        for _ in 0..2 {
            next_spin.memory_barrier();
            encode_steps(&mut next_spin, reg, dev, layer)?;
        }
    }
    let mut enc = dev.command_encoder()?;
    measure(&mut enc, reg)?;
    filler.commit();
    enc.commit();
    next_spin.commit();
    if let Some(previous) = spin.replace(next_spin) {
        previous.wait_until_completed()?;
    }
    filler.wait_until_completed()?;
    enc.wait_until_completed()?;
    Ok(enc.completed_gpu_interval_ns() as f64 * 1e-6)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn list<T: std::str::FromStr + Clone>(args: &[String], flag: &str, default: &[T]) -> Vec<T> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(|raw| {
            raw.split(',')
                .filter_map(|item| item.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn bench_ffn(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    rows: usize,
    down_type: GgmlType,
    ring: usize,
    repeat: usize,
    spin: &mut Option<CommandEncoder>,
) -> R<()> {
    let mut rng = Rng(0x5EED ^ rows as u64);
    let gate_source = weight(dev, GgmlType::Q4_K, INTERMEDIATE, HIDDEN, 7)?;
    let up_source = weight(dev, GgmlType::Q4_K, INTERMEDIATE, HIDDEN, 11)?;
    let down_source = weight(dev, down_type, HIDDEN, INTERMEDIATE, 13)?;
    let gate_up = params(rows, INTERMEDIATE, HIDDEN, GgmlType::Q4_K);
    let down = params(rows, HIDDEN, INTERMEDIATE, down_type);
    let layers = (0..ring)
        .map(|_| {
            Ok(Rc::new(Layer {
                gate: prepare(dev, reg, &gate_source, &gate_up)?,
                up: prepare(dev, reg, &up_source, &gate_up)?,
                down: prepare(dev, reg, &down_source, &down)?,
            }))
        })
        .collect::<R<Vec<_>>>()?;
    let mut silu_params = alloc(dev, DType::U32, &[1])?;
    silu_params.as_mut_slice::<u32>()?[0] = (rows * INTERMEDIATE) as u32;
    let scratch = Rc::new(Scratch {
        x: f32_rows(dev, rows, HIDDEN, &mut rng)?,
        residual: f32_rows(dev, rows, HIDDEN, &mut rng)?,
        x_bf16: alloc(dev, DType::BF16, &[rows, HIDDEN])?,
        gate: alloc(dev, DType::F32, &[rows, INTERMEDIATE])?,
        up: alloc(dev, DType::F32, &[rows, INTERMEDIATE])?,
        hidden: alloc(dev, DType::F32, &[rows, INTERMEDIATE])?,
        hidden_bf16: alloc(dev, DType::BF16, &[rows, INTERMEDIATE])?,
        down: alloc(dev, DType::F32, &[rows, HIDDEN])?,
        sum: alloc(dev, DType::F32, &[rows, HIDDEN])?,
        silu_params,
    });

    {
        let route = "separate";
        let build = separate_route;
        let routes: Vec<Vec<Step>> = layers
            .iter()
            .map(|layer| build(layer, &scratch, rows, down_type))
            .collect();
        let filler = |first: usize| -> Vec<&[Step]> {
            (0..FILLER_LAYERS)
                .map(|i| routes[(first + i) % ring].as_slice())
                .collect()
        };
        // Pipelines, page-in and clocks.
        for i in 0..2 * ring {
            sample(dev, reg, spin, &filler(i), &mut |enc, reg| {
                encode_steps(enc, reg, dev, &routes[i % ring])
            })?;
        }
        let fused = (0..repeat)
            .map(|i| {
                sample(dev, reg, spin, &filler(i + 1), &mut |enc, reg| {
                    encode_steps(enc, reg, dev, &routes[i % ring])
                })
            })
            .collect::<R<Vec<_>>>()?;
        let cycle = (0..repeat.min(7))
            .map(|i| {
                sample(dev, reg, spin, &filler(i), &mut |enc, reg| {
                    for layer in 0..CYCLE_LAYERS {
                        enc.memory_barrier();
                        encode_steps(enc, reg, dev, &routes[(i + layer) % ring])?;
                    }
                    Ok(())
                })
            })
            .collect::<R<Vec<_>>>()?;
        let mut kinds: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        for round in 0..SEPARATE_REPEAT.max(repeat / 2) {
            let steps = &routes[round % ring];
            for index in 0..steps.len() {
                // The steps before it run in the filler so the measured one
                // reads what the layer would hand it.
                let ms = sample(dev, reg, spin, &filler(round + 1), &mut |enc, reg| {
                    (steps[index].encode)(enc, reg, dev)
                })?;
                kinds.entry(steps[index].kind).or_default().push(ms);
            }
        }
        let fused_ms = median(fused.clone());
        let fused_min = fused.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "ffn rows={rows} down={down_type:?} route={route}: layer fused {fused_ms:.4} ms (min {fused_min:.4}), 64 layers fused {:.3} ms",
            median(cycle)
        );
        let mut total = 0.0;
        for step in &routes[0] {
            let ms = median(kinds[step.kind].clone());
            total += ms;
            println!("    {:<14} {ms:.4} ms", step.kind);
        }
        println!(
            "    {:<14} {total:.4} ms per layer, {:.3} ms per 64 layers",
            "sum separate",
            total * CYCLE_LAYERS as f64
        );
    }
    Ok(())
}

fn bench_head(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    rows: usize,
    k: usize,
    repeat: usize,
    filler_layer: &[Step],
    spin: &mut Option<CommandEncoder>,
) -> R<()> {
    let mut rng = Rng(0xEAD ^ k as u64);
    let source = weight(dev, GgmlType::Q6_K, VOCAB, k, 3)?;
    let p = params(rows, VOCAB, k, GgmlType::Q6_K);
    let prepared = prepare(dev, reg, &source, &p)?;
    let x = f32_rows(dev, rows, k, &mut rng)?;
    let x_bf16 = alloc(dev, DType::BF16, &[rows, k])?;
    let logits = alloc(dev, DType::F32, &[rows, VOCAB])?;
    let indices = alloc(dev, DType::U32, &[rows])?;
    let values = alloc(dev, DType::F32, &[rows])?;
    let mut argmax_params = alloc(dev, DType::U32, &[1])?;
    argmax_params.as_mut_slice::<u32>()?[0] = VOCAB as u32;

    let filler: Vec<&[Step]> = vec![filler_layer; FILLER_LAYERS];
    let mut measure = |reg: &mut KernelRegistry,
                   label: &str,
                   encode: &mut dyn FnMut(&mut CommandEncoder, &mut KernelRegistry) -> R<()>|
     -> R<f64> {
        for _ in 0..3 {
            sample(dev, reg, spin, &filler, encode)?;
        }
        let ms = median(
            (0..repeat)
                .map(|_| sample(dev, reg, spin, &filler, encode))
                .collect::<R<Vec<_>>>()?,
        );
        println!("head rows={rows} K={k} {label:<28} {ms:.4} ms");
        Ok(ms)
    };

    measure(reg, "stock quantized_matmul_ggml", &mut |enc, reg| {
        Ok(quantized_matmul_ggml(enc, reg, dev, &x, &source, &logits, &p)?)
    })?;
    measure(reg, "cast F32->BF16", &mut |enc, reg| {
        Ok(cast(
            enc,
            reg,
            dev.metal_device(),
            &x,
            &x_bf16,
            rows * k,
            CastDirection::F32ToBF16,
        )?)
    })?;
    measure(reg, "prepared BF16-input MM", &mut |enc, reg| {
        Ok(dispatch_repacked_gguf_mm_bf16_input(
            enc, reg, dev, &x_bf16, &prepared, &logits, &p,
        )?)
    })?;
    measure(reg, "argmax per row", &mut |enc, reg| {
        for row in 0..rows {
            dispatch_argmax_f32(
                enc,
                reg,
                dev.metal_device(),
                &logits.slice_view((row * VOCAB * 4) as u64, VOCAB),
                &indices.slice_view((row * 4) as u64, 1),
                &values.slice_view((row * 4) as u64, 1),
                &argmax_params,
                VOCAB as u32,
            )?;
        }
        Ok(())
    })?;
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let rows: Vec<usize> = list(&args, "--rows", &[8, 16, 24, 32]);
    let repeat = list(&args, "--repeat", &[20usize])[0];
    let ring = list(&args, "--ring", &[6usize])[0];
    let down: Vec<String> = list(&args, "--down", &["q4k".to_string(), "q6k".to_string()]);
    let head_k: Vec<usize> = list(&args, "--head-k", &[5_120, 2_048]);
    let parts: Vec<String> = list(&args, "--parts", &["ffn".to_string(), "head".to_string()]);

    let dev = MlxDevice::new()?;
    let mut reg = KernelRegistry::new();
    let mut spin = None;
    println!(
        "device {}, median of {repeat}, ring of {ring} weight copies, filler of {FILLER_LAYERS} layers",
        dev.name()
    );
    if parts.iter().any(|part| part == "ffn") {
        for down in &down {
            let down_type = match down.as_str() {
                "q4k" => GgmlType::Q4_K,
                "q6k" => GgmlType::Q6_K,
                other => return Err(format!("unknown down type {other}").into()),
            };
            for &rows in &rows {
                bench_ffn(&dev, &mut reg, rows, down_type, ring, repeat, &mut spin)?;
            }
        }
    }
    if parts.iter().any(|part| part == "head") {
        // A filler of real FFN work for the head's samples.
        let mut rng = Rng(1);
        let gate_up = params(32, INTERMEDIATE, HIDDEN, GgmlType::Q4_K);
        let down = params(32, HIDDEN, INTERMEDIATE, GgmlType::Q4_K);
        let layer = Rc::new(Layer {
            gate: prepare(
                &dev,
                &mut reg,
                &weight(&dev, GgmlType::Q4_K, INTERMEDIATE, HIDDEN, 7)?,
                &gate_up,
            )?,
            up: prepare(
                &dev,
                &mut reg,
                &weight(&dev, GgmlType::Q4_K, INTERMEDIATE, HIDDEN, 11)?,
                &gate_up,
            )?,
            down: prepare(
                &dev,
                &mut reg,
                &weight(&dev, GgmlType::Q4_K, HIDDEN, INTERMEDIATE, 13)?,
                &down,
            )?,
        });
        let mut silu_params = alloc(&dev, DType::U32, &[1])?;
        silu_params.as_mut_slice::<u32>()?[0] = (32 * INTERMEDIATE) as u32;
        let scratch = Rc::new(Scratch {
            x: f32_rows(&dev, 32, HIDDEN, &mut rng)?,
            residual: f32_rows(&dev, 32, HIDDEN, &mut rng)?,
            x_bf16: alloc(&dev, DType::BF16, &[32, HIDDEN])?,
            gate: alloc(&dev, DType::F32, &[32, INTERMEDIATE])?,
            up: alloc(&dev, DType::F32, &[32, INTERMEDIATE])?,
            hidden: alloc(&dev, DType::F32, &[32, INTERMEDIATE])?,
            hidden_bf16: alloc(&dev, DType::BF16, &[32, INTERMEDIATE])?,
            down: alloc(&dev, DType::F32, &[32, HIDDEN])?,
            sum: alloc(&dev, DType::F32, &[32, HIDDEN])?,
            silu_params,
        });
        let filler_layer = separate_route(&layer, &scratch, 32, GgmlType::Q4_K);
        for &k in &head_k {
            for &rows in &rows {
                bench_head(&dev, &mut reg, rows, k, repeat.min(9), &filler_layer, &mut spin)?;
            }
        }
    }
    if let Some(spin) = spin.take() {
        spin.wait_until_completed()?;
    }
    Ok(())
}
