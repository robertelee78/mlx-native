//! DeltaNet core of one joint DFlash2 verify layer, model-free, at the
//! served Qwen geometries (hf2q Splash-gap program, DeltaProj stage).
//!
//! A verify step carries 1-4 lanes of 8 rows. Between the input projections
//! and the output projection a DeltaNet layer runs the convolution, the Q/K
//! L2 norms, the query scale, the gates, the delta-rule recurrence against
//! each lane's slot state and the gated RMS norm. This bench times that span
//! two ways:
//!
//! - `chain`: the stock kernels as the joint decode core issues them: the
//!   alpha/beta split, `ssm_conv` and a raw-row copy per window | per row an
//!   `l2_norm` of Q, one of K and a V copy | per row `scalar_mul`, and
//!   `compute_g_beta` | `gated_delta_net_decode` per window |
//!   `ssm_norm_gate` | the F32 to BF16 cast of the gated rows.
//! - `fused`: `dispatch_gdn_window_fused`, one dispatch with a BF16 output.
//!   Bitwise the chain (`tests/test_gdn_window_fused.rs`).
//!
//! The recurrent state is kept DRAM-cold as in a serving cycle, where each
//! layer's state is read and written once per cycle: a ring of layers of at
//! least `RING_BYTES` of state is walked. Every sample is the GPU interval
//! of a command buffer of `--sample-layers` consecutive layers committed
//! right behind a filler command buffer of the same route (`FILLER_LAYERS`
//! layers at four lanes, proportionally more at fewer, so the GPU is at
//! serving clocks), divided by the layer count.
//!
//! Usage: `cargo bench --bench bench_qwen_delta_core -- [--shapes
//! qwen38,qwen36] [--lanes 1,2,4] [--routes chain,fused] [--repeat 20]
//! [--sample-layers 4] [--json PATH]`

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use mlx_native::ops::compute_g_beta::dispatch_compute_g_beta;
use mlx_native::ops::copy::{dispatch_copy_f32, dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::ops::elementwise::{cast, scalar_mul_f32, CastDirection};
use mlx_native::ops::gated_delta_net::GatedDeltaNetParams;
use mlx_native::ops::gated_delta_net_decode::dispatch_gated_delta_net_decode;
use mlx_native::ops::gdn_window_fused::{
    dispatch_gdn_window_fused, GdnWindowInputs, GdnWindowLane, GdnWindowRows, GdnWindowShape,
    GdnWindowState, GdnWindowTape, GdnWindowWeights, HEAD_DIM,
};
use mlx_native::ops::l2_norm::dispatch_l2_norm;
use mlx_native::ops::ssm_conv::{dispatch_ssm_conv, SsmConvParams};
use mlx_native::ops::ssm_norm_gate::dispatch_ssm_norm_gate;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

type R<T> = Result<T, Box<dyn std::error::Error>>;

const HD: usize = HEAD_DIM as usize;
const ROWS: usize = 8;
const SLOTS: usize = 4;
const TAPS: u32 = 4;
const EPS: f32 = 1e-6;
const WARMUP_MS: u128 = 300;
/// Filler layers at four lanes; fewer lanes queue proportionally more, so
/// the GPU stays at serving clocks behind a light layer too.
const FILLER_LAYERS: usize = 6;
const RING_BYTES: usize = 384 << 20;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    n_k: u32,
    n_v: u32,
    /// The dense target pads the verify tile to 32 rows.
    dense: bool,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "qwen38",
        n_k: 16,
        n_v: 48,
        dense: true,
    },
    Shape {
        name: "qwen36",
        n_k: 16,
        n_v: 32,
        dense: false,
    },
];

impl Shape {
    fn key(self) -> usize {
        self.n_k as usize * HD
    }
    fn value(self) -> usize {
        self.n_v as usize * HD
    }
    fn channels(self) -> usize {
        2 * self.key() + self.value()
    }
    fn nv(self) -> usize {
        self.n_v as usize
    }
    /// Columns of the fused alpha/beta product.
    fn ab_stride(self) -> usize {
        (2 * self.nv()).div_ceil(64) * 64
    }
}

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
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn f32_buf(dev: &MlxDevice, elements: usize, rng: &mut Rng, lo: f32, hi: f32) -> R<MlxBuffer> {
    let mut buffer = dev.alloc_buffer(elements * 4, DType::F32, vec![elements])?;
    for value in buffer.as_mut_slice::<f32>()? {
        *value = rng.range(lo, hi);
    }
    Ok(buffer)
}

fn scratch(dev: &MlxDevice, dtype: DType, elements: usize) -> R<MlxBuffer> {
    Ok(dev.alloc_buffer((elements * dtype.size_of()).max(16), dtype, vec![elements])?)
}

fn params_u32(dev: &MlxDevice, values: &[u32]) -> R<MlxBuffer> {
    let mut buffer = scratch(dev, DType::U32, values.len())?;
    buffer.as_mut_slice::<u32>()?[..values.len()].copy_from_slice(values);
    Ok(buffer)
}

fn params_f32(dev: &MlxDevice, values: &[f32]) -> R<MlxBuffer> {
    let mut buffer = scratch(dev, DType::F32, values.len())?;
    buffer.as_mut_slice::<f32>()?[..values.len()].copy_from_slice(values);
    Ok(buffer)
}

/// The state of one layer of the ring.
struct Layer {
    conv: [MlxBuffer; 2],
    rec: [MlxBuffer; 2],
}

struct Case {
    shape: Shape,
    lanes: usize,
    width: usize,
    qkv: MlxBuffer,
    z: MlxBuffer,
    /// Fused alpha/beta product rows `[width][ab_stride]`.
    ab: MlxBuffer,
    conv_w: MlxBuffer,
    dt_bias: MlxBuffer,
    ssm_a: MlxBuffer,
    norm: MlxBuffer,
    layers: Vec<Layer>,
    // Chain scratch.
    alpha: MlxBuffer,
    beta_logit: MlxBuffer,
    qkv_conv: MlxBuffer,
    q_l2: MlxBuffer,
    core: MlxBuffer,
    gated: MlxBuffer,
    gated_bf16: MlxBuffer,
    // The replay storage both routes write: per lane raw rows, q, k, v;
    // then g and beta of every row.
    tape: MlxBuffer,
    tapes: Vec<GdnWindowTape>,
    conv_params: MlxBuffer,
    l2_params: MlxBuffer,
    g_params: MlxBuffer,
    gdn_params: MlxBuffer,
    gate_params: MlxBuffer,
}

impl Case {
    fn new(dev: &MlxDevice, shape: Shape, lanes: usize) -> R<Self> {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (lanes as u64) << 8 ^ u64::from(shape.n_v));
        let real = lanes * ROWS;
        let width = if shape.dense {
            32
        } else {
            real.next_power_of_two()
        };
        let (channels, key, value, nv) = (shape.channels(), shape.key(), shape.value(), shape.nv());
        let conv_slot = channels * (TAPS as usize - 1);
        let rec_slot = value * HD;
        let layer_bytes = 2 * SLOTS * (conv_slot + rec_slot) * 4;
        let ring = RING_BYTES
            .div_ceil(layer_bytes)
            .max(FILLER_LAYERS * SLOTS / lanes + 5);
        let mut layers = Vec::with_capacity(ring);
        for _ in 0..ring {
            layers.push(Layer {
                conv: [
                    f32_buf(dev, SLOTS * conv_slot, &mut rng, -1.0, 1.0)?,
                    f32_buf(dev, SLOTS * conv_slot, &mut rng, -1.0, 1.0)?,
                ],
                rec: [
                    f32_buf(dev, SLOTS * rec_slot, &mut rng, -0.5, 0.5)?,
                    f32_buf(dev, SLOTS * rec_slot, &mut rng, -0.5, 0.5)?,
                ],
            });
        }
        let per_lane = (TAPS as usize - 1 + ROWS) * channels + 2 * ROWS * key + ROWS * value;
        let gates = lanes * per_lane;
        let tapes = (0..lanes)
            .map(|lane| {
                let base = lane * per_lane;
                let conv_rows = base + (TAPS as usize - 1) * channels;
                let q = base + (TAPS as usize - 1 + ROWS) * channels;
                GdnWindowTape {
                    conv_rows: conv_rows as u32,
                    q: q as u32,
                    k: (q + ROWS * key) as u32,
                    v: (q + 2 * ROWS * key) as u32,
                    g: (gates + lane * ROWS * nv) as u32,
                    beta: (gates + width * nv + lane * ROWS * nv) as u32,
                }
            })
            .collect();
        Ok(Self {
            shape,
            lanes,
            width,
            qkv: f32_buf(dev, width * channels, &mut rng, -2.0, 2.0)?,
            z: f32_buf(dev, width * value, &mut rng, -3.0, 3.0)?,
            ab: f32_buf(dev, width * shape.ab_stride(), &mut rng, -4.0, 4.0)?,
            conv_w: f32_buf(dev, channels * TAPS as usize, &mut rng, -0.6, 0.6)?,
            dt_bias: f32_buf(dev, nv, &mut rng, -1.0, 1.0)?,
            ssm_a: f32_buf(dev, nv, &mut rng, -1.5, -0.01)?,
            norm: f32_buf(dev, HD, &mut rng, 0.5, 1.5)?,
            layers,
            alpha: scratch(dev, DType::F32, width * nv)?,
            beta_logit: scratch(dev, DType::F32, width * nv)?,
            qkv_conv: scratch(dev, DType::F32, width * channels)?,
            q_l2: scratch(dev, DType::F32, width * key)?,
            core: scratch(dev, DType::F32, width * value)?,
            gated: scratch(dev, DType::F32, width * value)?,
            gated_bf16: scratch(dev, DType::BF16, width * value)?,
            tape: scratch(dev, DType::F32, gates + 2 * width * nv)?,
            tapes,
            conv_params: params_u32(dev, &[channels as u32, ROWS as u32, 1, TAPS])?,
            l2_params: params_f32(dev, &[EPS, HD as f32])?,
            g_params: params_u32(dev, &[shape.n_v, width as u32])?,
            gdn_params: params_u32(
                dev,
                &[
                    HEAD_DIM,
                    HEAD_DIM,
                    shape.n_k,
                    shape.n_v,
                    ROWS as u32,
                    1,
                    0,
                    0,
                    0,
                ],
            )?,
            gate_params: params_f32(dev, &[EPS, HD as f32])?,
        })
    }

    fn q_scale(&self) -> f32 {
        1.0 / (HD as f32).sqrt()
    }

    /// The stock chain over ring layer `layer`; returns the dispatch count.
    fn encode_chain(
        &self,
        enc: &mut CommandEncoder,
        reg: &mut KernelRegistry,
        dev: &MlxDevice,
        layer: &Layer,
    ) -> R<u32> {
        let shape = self.shape;
        let (channels, key, value, nv) = (shape.channels(), shape.key(), shape.value(), shape.nv());
        let metal = dev.metal_device();
        let conv_slot = channels * (TAPS as usize - 1);
        let rec_slot = value * HD;
        let mut dispatches = 0;
        // Alpha/beta split of the fused product.
        let split = StridedCopyParams {
            rows: self.width as u32,
            cols: shape.n_v,
            stride_row: shape.ab_stride() as u32,
            stride_col: 1,
        };
        dispatch_strided_copy_f32(enc, reg, metal, &self.ab, &self.alpha, &split)?;
        let beta_source = self
            .ab
            .slice_view((nv * 4) as u64, self.width * shape.ab_stride() - nv);
        dispatch_strided_copy_f32(enc, reg, metal, &beta_source, &self.beta_logit, &split)?;
        dispatches += 2;
        let conv = SsmConvParams {
            channels: channels as u32,
            n_tokens: ROWS as u32,
            n_seqs: 1,
            k_width: TAPS,
        };
        let rows = |buffer: &MlxBuffer, lane: usize, stride: usize| {
            buffer.slice_view((lane * ROWS * stride * 4) as u64, ROWS * stride)
        };
        let field =
            |start: u32, elements: usize| self.tape.slice_view(u64::from(start) * 4, elements);
        for lane in 0..self.lanes {
            dispatch_ssm_conv(
                enc,
                reg,
                metal,
                &rows(&self.qkv, lane, channels),
                &self.conv_w,
                &layer.conv[0].slice_view((lane * conv_slot * 4) as u64, conv_slot),
                &layer.conv[1].slice_view((lane * conv_slot * 4) as u64, conv_slot),
                &rows(&self.qkv_conv, lane, channels),
                &self.conv_params,
                conv,
            )?;
            dispatch_copy_f32(
                enc,
                reg,
                metal,
                &self.qkv,
                &self.tape,
                lane * ROWS * channels,
                self.tapes[lane].conv_rows as usize,
                ROWS * channels,
            )?;
            dispatches += 3;
        }
        enc.memory_barrier();
        for lane in 0..self.lanes {
            let tape = self.tapes[lane];
            for row in 0..ROWS {
                let base = (lane * ROWS + row) * channels;
                dispatch_l2_norm(
                    enc,
                    reg,
                    metal,
                    &self.qkv_conv.slice_view((base * 4) as u64, key),
                    &self
                        .q_l2
                        .slice_view(((lane * ROWS + row) * key * 4) as u64, key),
                    &self.l2_params,
                    shape.n_k,
                    HEAD_DIM,
                )?;
                dispatch_l2_norm(
                    enc,
                    reg,
                    metal,
                    &self.qkv_conv.slice_view(((base + key) * 4) as u64, key),
                    &field(tape.k + (row * key) as u32, key),
                    &self.l2_params,
                    shape.n_k,
                    HEAD_DIM,
                )?;
                dispatch_copy_f32(
                    enc,
                    reg,
                    metal,
                    &self.qkv_conv,
                    &self.tape,
                    base + 2 * key,
                    tape.v as usize + row * value,
                    value,
                )?;
                dispatches += 3;
            }
        }
        enc.memory_barrier();
        for lane in 0..self.lanes {
            for row in 0..ROWS {
                scalar_mul_f32(
                    enc,
                    reg,
                    metal,
                    &self
                        .q_l2
                        .slice_view(((lane * ROWS + row) * key * 4) as u64, key),
                    &field(self.tapes[lane].q + (row * key) as u32, key),
                    key,
                    self.q_scale(),
                )?;
                dispatches += 1;
            }
        }
        let g_all = field(self.tapes[0].g, self.width * nv);
        let beta_all = field(self.tapes[0].beta, self.width * nv);
        dispatch_compute_g_beta(
            enc,
            reg,
            metal,
            &self.alpha,
            &self.beta_logit,
            &self.dt_bias,
            &self.ssm_a,
            &g_all,
            &beta_all,
            &self.g_params,
            self.width as u32,
            shape.n_v,
        )?;
        dispatches += 1;
        enc.memory_barrier();
        let gdn = GatedDeltaNetParams {
            d_k: HEAD_DIM,
            d_v: HEAD_DIM,
            n_k_heads: shape.n_k,
            n_v_heads: shape.n_v,
            n_tokens: ROWS as u32,
            n_seqs: 1,
        };
        for lane in 0..self.lanes {
            let tape = self.tapes[lane];
            dispatch_gated_delta_net_decode(
                enc,
                reg,
                metal,
                &field(tape.q, ROWS * key),
                &field(tape.k, ROWS * key),
                &field(tape.v, ROWS * value),
                &field(tape.g, ROWS * nv),
                &field(tape.beta, ROWS * nv),
                &layer.rec[0].slice_view((lane * rec_slot * 4) as u64, rec_slot),
                &rows(&self.core, lane, value),
                &layer.rec[1].slice_view((lane * rec_slot * 4) as u64, rec_slot),
                &self.gdn_params,
                gdn,
            )?;
            dispatches += 1;
        }
        enc.memory_barrier();
        dispatch_ssm_norm_gate(
            enc,
            reg,
            metal,
            &self.core,
            &self.norm,
            &self.z,
            &self.gated,
            &self.gate_params,
            self.width as u32 * shape.n_v,
            HEAD_DIM,
        )?;
        enc.memory_barrier();
        cast(
            enc,
            reg,
            metal,
            &self.gated,
            &self.gated_bf16,
            self.width * value,
            CastDirection::F32ToBF16,
        )?;
        Ok(dispatches + 2)
    }

    /// The fused kernel over ring layer `layer`.
    fn encode_fused(
        &self,
        enc: &mut CommandEncoder,
        reg: &mut KernelRegistry,
        dev: &MlxDevice,
        layer: &Layer,
    ) -> R<u32> {
        let shape = self.shape;
        let nv = shape.nv();
        let mut lanes: Vec<GdnWindowLane> = (0..self.lanes)
            .map(|lane| GdnWindowLane::Window {
                first_row: (lane * ROWS) as u32,
                rows: ROWS as u32,
                slot: lane as u32,
                state_in: 0,
                tape: Some(self.tapes[lane]),
            })
            .collect();
        for first in (self.lanes * ROWS..self.width).step_by(ROWS) {
            lanes.push(GdnWindowLane::Padding {
                first_row: first as u32,
                rows: ROWS.min(self.width - first) as u32,
            });
        }
        let ab_stride = shape.ab_stride() as u32;
        let beta = self
            .ab
            .slice_view((nv * 4) as u64, self.width * shape.ab_stride() - nv);
        dispatch_gdn_window_fused(
            enc,
            reg,
            dev,
            &GdnWindowShape {
                n_k_heads: shape.n_k,
                n_v_heads: shape.n_v,
                head_dim: HEAD_DIM,
                conv_taps: TAPS,
                eps: EPS,
                q_scale: self.q_scale(),
                slots: SLOTS as u32,
            },
            &GdnWindowInputs {
                qkv: GdnWindowRows {
                    buffer: &self.qkv,
                    stride: shape.channels() as u32,
                },
                z: GdnWindowRows {
                    buffer: &self.z,
                    stride: shape.value() as u32,
                },
                alpha: GdnWindowRows {
                    buffer: &self.ab,
                    stride: ab_stride,
                },
                beta: GdnWindowRows {
                    buffer: &beta,
                    stride: ab_stride,
                },
            },
            &GdnWindowWeights {
                conv: &self.conv_w,
                dt_bias: &self.dt_bias,
                ssm_a: &self.ssm_a,
                norm: &self.norm,
            },
            &GdnWindowState {
                conv: [&layer.conv[0], &layer.conv[1]],
                recurrent: [&layer.rec[0], &layer.rec[1]],
            },
            &self.gated_bf16,
            Some(&self.tape),
            self.width as u32,
            &lanes,
        )?;
        Ok(1)
    }

    fn encode(
        &self,
        enc: &mut CommandEncoder,
        reg: &mut KernelRegistry,
        dev: &MlxDevice,
        route: &str,
        layer: usize,
    ) -> R<u32> {
        let layer = &self.layers[layer % self.layers.len()];
        match route {
            "chain" => self.encode_chain(enc, reg, dev, layer),
            "fused" => self.encode_fused(enc, reg, dev, layer),
            other => Err(format!("unknown route {other}").into()),
        }
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

/// GPU ms per layer of `sample_layers` ring layers from `at`, behind a
/// filler of the layers after them, and one layer's dispatch count.
fn hot_gpu_ms(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    case: &Case,
    route: &str,
    at: usize,
    sample_layers: usize,
) -> R<(f64, u32)> {
    let mut filler = dev.command_encoder()?;
    for step in 0..FILLER_LAYERS * SLOTS / case.lanes {
        filler.memory_barrier();
        case.encode(&mut filler, reg, dev, route, at + sample_layers + step)?;
    }
    filler.commit();
    let mut enc = dev.command_encoder()?;
    let mut dispatches = 0;
    for step in 0..sample_layers {
        if step > 0 {
            enc.memory_barrier();
        }
        dispatches = case.encode(&mut enc, reg, dev, route, at + step)?;
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
    let shapes: Vec<String> = list(
        &args,
        "--shapes",
        &["qwen38".to_string(), "qwen36".to_string()],
    );
    let lanes: Vec<usize> = list(&args, "--lanes", &[1, 2, 4]);
    let routes: Vec<String> = list(
        &args,
        "--routes",
        &["chain".to_string(), "fused".to_string()],
    );
    let repeat = list(&args, "--repeat", &[20usize])[0];
    let sample_layers = list(&args, "--sample-layers", &[4usize])[0].clamp(1, 4);
    let json_path: Option<String> = list::<String>(&args, "--json", &[]).into_iter().next();

    let dev = MlxDevice::new()?;
    let mut reg = KernelRegistry::new();
    println!(
        "device {}, DeltaNet core of one verify layer, {ROWS} rows per lane, DRAM-cold state ring of at least {} MiB, {sample_layers} layers per sample, median of {repeat}",
        dev.name(),
        RING_BYTES >> 20
    );
    let mut records = Vec::new();
    for shape in SHAPES
        .iter()
        .filter(|shape| shapes.iter().any(|name| name == shape.name))
    {
        for &lane_count in &lanes {
            let case = Case::new(&dev, *shape, lane_count.clamp(1, SLOTS))?;
            for route in &routes {
                let mut at = 0usize;
                let mut sample = |reg: &mut KernelRegistry| -> R<(f64, u32)> {
                    at += FILLER_LAYERS * SLOTS / case.lanes + sample_layers;
                    hot_gpu_ms(&dev, reg, &case, route, at, sample_layers)
                };
                let warm = std::time::Instant::now();
                while warm.elapsed().as_millis() < WARMUP_MS {
                    sample(&mut reg)?;
                }
                let mut values = Vec::with_capacity(repeat);
                let mut dispatches = 0;
                for _ in 0..repeat {
                    let (ms, count) = sample(&mut reg)?;
                    values.push(ms);
                    dispatches = count;
                }
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let layer_ms = median(&mut values);
                println!(
                    "{:<7} {}/{} heads lanes={} width={:<2} {:<5} layer={:.4} ms (min {:.4}) dispatches={:<3} ring={}",
                    shape.name,
                    shape.n_k,
                    shape.n_v,
                    case.lanes,
                    case.width,
                    route,
                    layer_ms,
                    min,
                    dispatches,
                    case.layers.len()
                );
                records.push(format!(
                    "{{\"shape\":\"{}\",\"lanes\":{},\"width\":{},\"route\":\"{}\",\"layer_ms\":{:.6},\"min_ms\":{:.6},\"dispatches\":{}}}",
                    shape.name, case.lanes, case.width, route, layer_ms, min, dispatches
                ));
            }
        }
    }
    if let Some(path) = json_path {
        std::fs::write(&path, format!("[\n{}\n]\n", records.join(",\n")))?;
        println!("wrote {path}");
    }
    Ok(())
}
