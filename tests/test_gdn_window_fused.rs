//! Gates of the fused DeltaNet window kernel (`ops::gdn_window_fused`).
//!
//! The kernel replaces a chain of stock kernels and must reproduce it bit for
//! bit (Class A), so the stock chain is the reference. At the served Qwen
//! geometries (16 key / 48 value heads and 16 / 32, head dimension 128, four
//! convolution taps, 1-4 lanes of 1-8 rows):
//!
//! 1. Bitwise equality with the stock chain as the joint decode core issues
//!    it per lane (`ssm_conv`, `l2_norm` of Q and of K per row, `scalar_mul`
//!    per row, `compute_g_beta`, `gated_delta_net_decode`, `ssm_norm_gate`):
//!    the gated output rows, the convolution and recurrent state after the
//!    window, and every retained row (scaled Q, normalized K, V, raw rows, g,
//!    beta). The BF16 output equals `cast_f32_to_bf16` of the F32 output.
//! 2. Containment: no other slot's state, neither buffer's source region, no
//!    uncovered output row and no tape element outside the lanes' ranges
//!    changes; padding rows are zero.
//! 3. Determinism: repeated runs are bitwise equal; a digest of every output
//!    is printed so runs under `MTL_SHADER_VALIDATION=1` and
//!    `MTL_DEBUG_LAYER=1` can be compared with a plain run.
//! 4. Lane independence, bitwise: a lane's outputs and state do not depend
//!    on the other lanes, their order, the packed rows it occupies, the
//!    layout of its inputs (separate buffers or columns of one packed
//!    buffer) or whether it retains its rows.
//! 5. Invalid steps are rejected before anything is encoded.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use mlx_native::ops::compute_g_beta::dispatch_compute_g_beta;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::ops::elementwise::{cast, scalar_mul_f32, CastDirection};
use mlx_native::ops::gated_delta_net::GatedDeltaNetParams;
use mlx_native::ops::gated_delta_net_decode::dispatch_gated_delta_net_decode;
use mlx_native::ops::gdn_window_fused::{
    dispatch_gdn_window_fused, gdn_window_fused_available, GdnWindowInputs, GdnWindowLane,
    GdnWindowRows, GdnWindowShape, GdnWindowState, GdnWindowTape, GdnWindowWeights, HEAD_DIM,
    WINDOW_ROWS,
};
use mlx_native::ops::l2_norm::dispatch_l2_norm;
use mlx_native::ops::ssm_conv::{dispatch_ssm_conv, SsmConvParams};
use mlx_native::ops::ssm_norm_gate::dispatch_ssm_norm_gate;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const HD: usize = HEAD_DIM as usize;
const SLOTS: u32 = 4;
const TAPS: u32 = 4;
const EPS: f32 = 1e-6;

struct Xor(u64);

impl Xor {
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
    fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    n_k: u32,
    n_v: u32,
}

const QWEN38: Geometry = Geometry { n_k: 16, n_v: 48 };
const QWEN36: Geometry = Geometry { n_k: 16, n_v: 32 };

impl Geometry {
    fn key(self) -> usize {
        self.n_k as usize * HD
    }
    fn value(self) -> usize {
        self.n_v as usize * HD
    }
    fn channels(self) -> usize {
        2 * self.key() + self.value()
    }
    fn conv_slot(self) -> usize {
        self.channels() * (TAPS as usize - 1)
    }
    fn rec_slot(self) -> usize {
        self.value() * HD
    }
    fn shape(self) -> GdnWindowShape {
        GdnWindowShape {
            n_k_heads: self.n_k,
            n_v_heads: self.n_v,
            head_dim: HEAD_DIM,
            conv_taps: TAPS,
            eps: EPS,
            q_scale: 1.0 / (HD as f32).sqrt(),
            slots: SLOTS,
        }
    }
}

/// One lane of a test step.
#[derive(Clone, Copy, Debug)]
struct Lane {
    first_row: u32,
    rows: u32,
    slot: u32,
    state_in: u32,
    retain: bool,
}

/// Host copy of one layer's inputs, weights and state.
#[derive(Clone)]
struct Step {
    geometry: Geometry,
    width: usize,
    qkv: Vec<f32>,
    z: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv_w: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    norm: Vec<f32>,
    conv: [Vec<f32>; 2],
    rec: [Vec<f32>; 2],
}

impl Step {
    fn random(geometry: Geometry, width: usize, rng: &mut Xor) -> Self {
        let nv = geometry.n_v as usize;
        let mut alpha = rng.fill(width * nv, -6.0, 6.0);
        // Both saturated branches of the softplus.
        alpha[0] = 30.0;
        alpha[1] = -30.0;
        Self {
            geometry,
            width,
            qkv: rng.fill(width * geometry.channels(), -2.0, 2.0),
            z: rng.fill(width * geometry.value(), -4.0, 4.0),
            alpha,
            beta: rng.fill(width * nv, -5.0, 5.0),
            conv_w: rng.fill(geometry.channels() * TAPS as usize, -0.6, 0.6),
            dt_bias: rng.fill(nv, -1.0, 1.0),
            ssm_a: rng.fill(nv, -1.5, -0.01),
            norm: rng.fill(HD, 0.5, 1.5),
            conv: [
                rng.fill(SLOTS as usize * geometry.conv_slot(), -2.0, 2.0),
                rng.fill(SLOTS as usize * geometry.conv_slot(), -2.0, 2.0),
            ],
            rec: [
                rng.fill(SLOTS as usize * geometry.rec_slot(), -1.0, 1.0),
                rng.fill(SLOTS as usize * geometry.rec_slot(), -1.0, 1.0),
            ],
        }
    }
}

fn buffer<T: bytemuck::Pod>(device: &MlxDevice, dtype: DType, values: &[T]) -> MlxBuffer {
    let bytes: &[u8] = bytemuck::cast_slice(values);
    let mut buffer = device
        .alloc_buffer(bytes.len().max(16), dtype, vec![values.len()])
        .expect("alloc");
    buffer.as_mut_slice::<u8>().expect("write")[..bytes.len()].copy_from_slice(bytes);
    buffer
}

fn zeros(device: &MlxDevice, elements: usize) -> MlxBuffer {
    buffer(device, DType::F32, &vec![0.0_f32; elements])
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

struct Gpu {
    device: MlxDevice,
    registry: KernelRegistry,
}

fn gpu() -> Option<Gpu> {
    let device = MlxDevice::new().ok()?;
    let mut registry = KernelRegistry::new();
    if !gdn_window_fused_available(&mut registry, &device).expect("availability") {
        eprintln!("[skip] gdn_window_fused: device limits");
        return None;
    }
    Some(Gpu { device, registry })
}

/// What a lane produced.
#[derive(Debug, PartialEq, Eq, Clone)]
struct LaneOutcome {
    gated: Vec<u32>,
    conv: Vec<u32>,
    rec: Vec<u32>,
    q: Vec<u32>,
    k: Vec<u32>,
    v: Vec<u32>,
    conv_rows: Vec<u32>,
    g: Vec<u32>,
    beta: Vec<u32>,
}

/// The stock kernels over one lane, issued as the joint decode core issues
/// them.
fn stock_lane(gpu: &mut Gpu, step: &Step, lane: &Lane) -> LaneOutcome {
    let geometry = step.geometry;
    let (channels, key, value) = (geometry.channels(), geometry.key(), geometry.value());
    let nv = geometry.n_v as usize;
    let n = lane.rows as usize;
    let first = lane.first_row as usize;
    let device = &gpu.device;
    let registry = &mut gpu.registry;
    let metal = device.metal_device();
    let rows_of =
        |values: &[f32], stride: usize| values[first * stride..(first + n) * stride].to_vec();
    let slot_of = |values: &[f32], per_slot: usize| {
        values[lane.slot as usize * per_slot..(lane.slot as usize + 1) * per_slot].to_vec()
    };
    let source = lane.state_in as usize;
    let x = buffer(device, DType::F32, &rows_of(&step.qkv, channels));
    let conv_in = buffer(
        device,
        DType::F32,
        &slot_of(&step.conv[source], geometry.conv_slot()),
    );
    let conv_out = zeros(device, geometry.conv_slot());
    let rec_in = buffer(
        device,
        DType::F32,
        &slot_of(&step.rec[source], geometry.rec_slot()),
    );
    let rec_out = zeros(device, geometry.rec_slot());
    let conv_w = buffer(device, DType::F32, &step.conv_w);
    let y = zeros(device, n * channels);
    let conv_params = SsmConvParams {
        channels: channels as u32,
        n_tokens: n as u32,
        n_seqs: 1,
        k_width: TAPS,
    };
    let conv_params_buf = buffer(device, DType::U32, &[channels as u32, n as u32, 1, TAPS]);
    let mut enc = device.command_encoder().expect("encoder");
    dispatch_ssm_conv(
        &mut enc,
        registry,
        metal,
        &x,
        &conv_w,
        &conv_in,
        &conv_out,
        &y,
        &conv_params_buf,
        conv_params,
    )
    .expect("ssm_conv");
    enc.memory_barrier();

    let l2_params = buffer(device, DType::F32, &[EPS, HD as f32]);
    let q_l2 = zeros(device, n * key);
    let k_l2 = zeros(device, n * key);
    let v = zeros(device, n * value);
    for t in 0..n {
        let base = t * channels;
        let q_view = y.slice_view((base * 4) as u64, key);
        let k_view = y.slice_view(((base + key) * 4) as u64, key);
        dispatch_l2_norm(
            &mut enc,
            registry,
            metal,
            &q_view,
            &q_l2.slice_view((t * key * 4) as u64, key),
            &l2_params,
            geometry.n_k,
            HEAD_DIM,
        )
        .expect("q l2");
        dispatch_l2_norm(
            &mut enc,
            registry,
            metal,
            &k_view,
            &k_l2.slice_view((t * key * 4) as u64, key),
            &l2_params,
            geometry.n_k,
            HEAD_DIM,
        )
        .expect("k l2");
        dispatch_copy_f32(
            &mut enc,
            registry,
            metal,
            &y,
            &v,
            base + 2 * key,
            t * value,
            value,
        )
        .expect("v gather");
    }
    enc.memory_barrier();

    let q = zeros(device, n * key);
    let q_scale = geometry.shape().q_scale;
    for t in 0..n {
        scalar_mul_f32(
            &mut enc,
            registry,
            metal,
            &q_l2.slice_view((t * key * 4) as u64, key),
            &q.slice_view((t * key * 4) as u64, key),
            key,
            q_scale,
        )
        .expect("q scale");
    }
    let alpha = buffer(device, DType::F32, &rows_of(&step.alpha, nv));
    let beta_logit = buffer(device, DType::F32, &rows_of(&step.beta, nv));
    let dt_bias = buffer(device, DType::F32, &step.dt_bias);
    let ssm_a = buffer(device, DType::F32, &step.ssm_a);
    let g = zeros(device, n * nv);
    let beta = zeros(device, n * nv);
    let g_params = buffer(device, DType::U32, &[geometry.n_v, n as u32]);
    dispatch_compute_g_beta(
        &mut enc,
        registry,
        metal,
        &alpha,
        &beta_logit,
        &dt_bias,
        &ssm_a,
        &g,
        &beta,
        &g_params,
        n as u32,
        geometry.n_v,
    )
    .expect("g beta");
    enc.memory_barrier();

    let gdn = GatedDeltaNetParams {
        d_k: HEAD_DIM,
        d_v: HEAD_DIM,
        n_k_heads: geometry.n_k,
        n_v_heads: geometry.n_v,
        n_tokens: n as u32,
        n_seqs: 1,
    };
    let gdn_params = buffer(
        device,
        DType::U32,
        &[
            HEAD_DIM,
            HEAD_DIM,
            geometry.n_k,
            geometry.n_v,
            n as u32,
            1,
            0,
            0,
            0,
        ],
    );
    let core = zeros(device, n * value);
    dispatch_gated_delta_net_decode(
        &mut enc,
        registry,
        metal,
        &q,
        &k_l2,
        &v,
        &g,
        &beta,
        &rec_in,
        &core,
        &rec_out,
        &gdn_params,
        gdn,
    )
    .expect("gdn");
    enc.memory_barrier();

    let z = buffer(device, DType::F32, &rows_of(&step.z, value));
    let norm = buffer(device, DType::F32, &step.norm);
    let gated = zeros(device, n * value);
    let gate_params = buffer(device, DType::F32, &[EPS, HD as f32]);
    dispatch_ssm_norm_gate(
        &mut enc,
        registry,
        metal,
        &core,
        &norm,
        &z,
        &gated,
        &gate_params,
        n as u32 * geometry.n_v,
        HEAD_DIM,
    )
    .expect("norm gate");
    enc.commit_and_wait().expect("stock chain");

    let read = |buffer: &MlxBuffer, elements: usize| {
        bits(&buffer.as_slice::<f32>().expect("read")[..elements])
    };
    LaneOutcome {
        gated: read(&gated, n * value),
        conv: read(&conv_out, geometry.conv_slot()),
        rec: read(&rec_out, geometry.rec_slot()),
        q: read(&q, n * key),
        k: read(&k_l2, n * key),
        v: read(&v, n * value),
        conv_rows: read(&x, n * channels),
        g: read(&g, n * nv),
        beta: read(&beta, n * nv),
    }
}

/// `cast_f32_to_bf16` of F32 bit patterns.
fn stock_bf16(gpu: &mut Gpu, values: &[u32]) -> Vec<u16> {
    let input = buffer(&gpu.device, DType::F32, values);
    let output = buffer(&gpu.device, DType::BF16, &vec![0_u16; values.len()]);
    let mut enc = gpu.device.command_encoder().expect("encoder");
    cast(
        &mut enc,
        &mut gpu.registry,
        gpu.device.metal_device(),
        &input,
        &output,
        values.len(),
        CastDirection::F32ToBF16,
    )
    .expect("cast");
    enc.commit_and_wait().expect("cast commit");
    output.as_slice::<u16>().expect("read")[..values.len()].to_vec()
}

const SENTINEL: f32 = -77.25;

/// Everything a fused dispatch wrote.
struct Fused {
    gated: Vec<u32>,
    gated_bf16: Vec<u16>,
    conv: [Vec<u32>; 2],
    rec: [Vec<u32>; 2],
    tape: Vec<u32>,
    tapes: Vec<Option<GdnWindowTape>>,
}

/// Run the fused kernel over `lanes` plus padding lanes for the rows in
/// `padding`. `packed` puts every input field in one row matrix.
fn run_fused(
    gpu: &mut Gpu,
    step: &Step,
    lanes: &[Lane],
    padding: &[(u32, u32)],
    packed: bool,
    dtype: DType,
) -> Fused {
    let geometry = step.geometry;
    let (channels, key, value) = (geometry.channels(), geometry.key(), geometry.value());
    let nv = geometry.n_v as usize;
    let width = step.width;
    let device = &gpu.device;
    let (qkv, z, alpha, beta, strides);
    let packed_buffer;
    if packed {
        let stride = channels + value + 2 * nv + 5;
        let mut rows = vec![SENTINEL; width * stride];
        for row in 0..width {
            let out = &mut rows[row * stride..(row + 1) * stride];
            out[..channels].copy_from_slice(&step.qkv[row * channels..(row + 1) * channels]);
            out[channels..channels + value]
                .copy_from_slice(&step.z[row * value..(row + 1) * value]);
            out[channels + value..channels + value + nv]
                .copy_from_slice(&step.alpha[row * nv..(row + 1) * nv]);
            out[channels + value + nv..channels + value + 2 * nv]
                .copy_from_slice(&step.beta[row * nv..(row + 1) * nv]);
        }
        packed_buffer = buffer(device, DType::F32, &rows);
        let view =
            |column: usize| packed_buffer.slice_view((column * 4) as u64, width * stride - column);
        qkv = view(0);
        z = view(channels);
        alpha = view(channels + value);
        beta = view(channels + value + nv);
        strides = [stride as u32; 4];
    } else {
        qkv = buffer(device, DType::F32, &step.qkv);
        z = buffer(device, DType::F32, &step.z);
        alpha = buffer(device, DType::F32, &step.alpha);
        beta = buffer(device, DType::F32, &step.beta);
        strides = [channels as u32, value as u32, nv as u32, nv as u32];
    }
    let conv_w = buffer(device, DType::F32, &step.conv_w);
    let dt_bias = buffer(device, DType::F32, &step.dt_bias);
    let ssm_a = buffer(device, DType::F32, &step.ssm_a);
    let norm = buffer(device, DType::F32, &step.norm);
    let conv = [
        buffer(device, DType::F32, &step.conv[0]),
        buffer(device, DType::F32, &step.conv[1]),
    ];
    let rec = [
        buffer(device, DType::F32, &step.rec[0]),
        buffer(device, DType::F32, &step.rec[1]),
    ];
    let gated = match dtype {
        DType::F32 => buffer(device, DType::F32, &vec![SENTINEL; width * value]),
        _ => buffer(device, DType::BF16, &vec![0x7777_u16; width * value]),
    };
    // Tape: per retaining lane q, k, v, raw rows, g, beta, with a gap of
    // sentinels between fields.
    let mut tapes = Vec::with_capacity(lanes.len());
    let mut cursor = 3_usize;
    for lane in lanes {
        let n = lane.rows as usize;
        tapes.push(lane.retain.then(|| {
            let mut take = |elements: usize| {
                let start = cursor;
                cursor += elements + 7;
                start as u32
            };
            GdnWindowTape {
                q: take(n * key),
                k: take(n * key),
                v: take(n * value),
                conv_rows: take(n * channels),
                g: take(n * nv),
                beta: take(n * nv),
            }
        }));
    }
    let tape = buffer(device, DType::F32, &vec![SENTINEL; cursor]);
    let mut table: Vec<GdnWindowLane> = lanes
        .iter()
        .zip(&tapes)
        .map(|(lane, tape)| GdnWindowLane::Window {
            first_row: lane.first_row,
            rows: lane.rows,
            slot: lane.slot,
            state_in: lane.state_in,
            tape: *tape,
        })
        .collect();
    table.extend(
        padding
            .iter()
            .map(|&(first_row, rows)| GdnWindowLane::Padding { first_row, rows }),
    );
    let mut enc = device.command_encoder().expect("encoder");
    dispatch_gdn_window_fused(
        &mut enc,
        &mut gpu.registry,
        device,
        &geometry.shape(),
        &GdnWindowInputs {
            qkv: GdnWindowRows {
                buffer: &qkv,
                stride: strides[0],
            },
            z: GdnWindowRows {
                buffer: &z,
                stride: strides[1],
            },
            alpha: GdnWindowRows {
                buffer: &alpha,
                stride: strides[2],
            },
            beta: GdnWindowRows {
                buffer: &beta,
                stride: strides[3],
            },
        },
        &GdnWindowWeights {
            conv: &conv_w,
            dt_bias: &dt_bias,
            ssm_a: &ssm_a,
            norm: &norm,
        },
        &GdnWindowState {
            conv: [&conv[0], &conv[1]],
            recurrent: [&rec[0], &rec[1]],
        },
        &gated,
        Some(&tape),
        width as u32,
        &table,
    )
    .expect("fused dispatch");
    enc.commit_and_wait().expect("fused commit");
    let read = |buffer: &MlxBuffer| bits(buffer.as_slice::<f32>().expect("read"));
    let (gated_f32, gated_bf16) = match dtype {
        DType::F32 => (read(&gated)[..width * value].to_vec(), Vec::new()),
        _ => (
            Vec::new(),
            gated.as_slice::<u16>().expect("read")[..width * value].to_vec(),
        ),
    };
    Fused {
        gated: gated_f32,
        gated_bf16,
        conv: [read(&conv[0]), read(&conv[1])],
        rec: [read(&rec[0]), read(&rec[1])],
        tape: read(&tape)[..cursor].to_vec(),
        tapes,
    }
}

/// The part of a fused run that belongs to lane `index`.
fn fused_lane(step: &Step, lanes: &[Lane], fused: &Fused, index: usize) -> LaneOutcome {
    let geometry = step.geometry;
    let (channels, key, value) = (geometry.channels(), geometry.key(), geometry.value());
    let nv = geometry.n_v as usize;
    let lane = &lanes[index];
    let n = lane.rows as usize;
    let first = lane.first_row as usize;
    let target = 1 - lane.state_in as usize;
    let slot = lane.slot as usize;
    let field = |start: u32, elements: usize| {
        fused.tape[start as usize..start as usize + elements].to_vec()
    };
    let tape = fused.tapes[index];
    LaneOutcome {
        gated: fused.gated[first * value..(first + n) * value].to_vec(),
        conv: fused.conv[target][slot * geometry.conv_slot()..(slot + 1) * geometry.conv_slot()]
            .to_vec(),
        rec: fused.rec[target][slot * geometry.rec_slot()..(slot + 1) * geometry.rec_slot()]
            .to_vec(),
        q: tape.map(|t| field(t.q, n * key)).unwrap_or_default(),
        k: tape.map(|t| field(t.k, n * key)).unwrap_or_default(),
        v: tape.map(|t| field(t.v, n * value)).unwrap_or_default(),
        conv_rows: tape
            .map(|t| field(t.conv_rows, n * channels))
            .unwrap_or_default(),
        g: tape.map(|t| field(t.g, n * nv)).unwrap_or_default(),
        beta: tape.map(|t| field(t.beta, n * nv)).unwrap_or_default(),
    }
}

fn first_difference(a: &[u32], b: &[u32]) -> String {
    if a.len() != b.len() {
        return format!("lengths {} and {}", a.len(), b.len());
    }
    let count = a.iter().zip(b).filter(|(x, y)| x != y).count();
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(index) => format!(
            "{count} of {} differ, first at {index}: {:e} vs {:e}",
            a.len(),
            f32::from_bits(a[index]),
            f32::from_bits(b[index])
        ),
        None => "equal".into(),
    }
}

fn assert_lane(label: &str, fused: &LaneOutcome, stock: &LaneOutcome, retained: bool) {
    let mut fields = vec![
        ("gated", &fused.gated, &stock.gated),
        ("conv state", &fused.conv, &stock.conv),
        ("recurrent state", &fused.rec, &stock.rec),
    ];
    if retained {
        fields.extend([
            ("tape q", &fused.q, &stock.q),
            ("tape k", &fused.k, &stock.k),
            ("tape v", &fused.v, &stock.v),
            ("tape raw rows", &fused.conv_rows, &stock.conv_rows),
            ("tape g", &fused.g, &stock.g),
            ("tape beta", &fused.beta, &stock.beta),
        ]);
    }
    for (name, got, want) in fields {
        assert!(
            got == want,
            "{label}: {name}: {}",
            first_difference(got, want)
        );
    }
}

fn digest(words: impl IntoIterator<Item = u32>) -> u64 {
    words
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, word| {
            (hash ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn fused_digest(fused: &Fused) -> u64 {
    digest(
        fused
            .gated
            .iter()
            .chain(&fused.conv[0])
            .chain(&fused.conv[1])
            .chain(&fused.rec[0])
            .chain(&fused.rec[1])
            .chain(&fused.tape)
            .copied()
            .chain(fused.gated_bf16.iter().map(|&v| u32::from(v))),
    )
}

/// Lanes of `rows` rows each, packed in order, slot `i` reading buffer
/// `i % 2`, all retaining.
fn packed_lanes(rows: &[u32]) -> Vec<Lane> {
    let mut first_row = 0;
    rows.iter()
        .enumerate()
        .map(|(index, &rows)| {
            let lane = Lane {
                first_row,
                rows,
                slot: index as u32,
                state_in: index as u32 % 2,
                retain: rows > 1,
            };
            first_row += rows;
            lane
        })
        .collect()
}

/// Padding lanes covering packed rows `from..width`.
fn padding_from(from: u32, width: u32) -> Vec<(u32, u32)> {
    (from..width)
        .step_by(WINDOW_ROWS as usize)
        .map(|first| (first, (width - first).min(WINDOW_ROWS)))
        .collect()
}

#[test]
fn fused_window_is_bitwise_the_stock_chain() {
    let Some(mut gpu) = gpu() else { return };
    for (geometry, name) in [(QWEN38, "16/48"), (QWEN36, "16/32")] {
        for (case, rows) in [
            &[8_u32, 8, 8, 8][..],
            &[8, 3, 1, 5],
            &[1],
            &[2, 7],
            &[1, 1, 1, 1],
        ]
        .into_iter()
        .enumerate()
        {
            let lanes = packed_lanes(rows);
            let real: u32 = rows.iter().sum();
            // The dense target pads to 32 rows; the MoE target to the next
            // power of two.
            let width = if geometry.n_v == 48 {
                32
            } else {
                real.next_power_of_two().max(4)
            };
            let mut rng =
                Xor(0x9E37_79B9_7F4A_7C15 ^ (case as u64) << 16 ^ u64::from(geometry.n_v));
            let step = Step::random(geometry, width as usize, &mut rng);
            let padding = padding_from(real, width);
            let fused = run_fused(&mut gpu, &step, &lanes, &padding, case % 2 == 1, DType::F32);
            for (index, lane) in lanes.iter().enumerate() {
                let stock = stock_lane(&mut gpu, &step, lane);
                // The reference itself is live: finite, not constant, and
                // the state moved.
                let live = |words: &[u32]| {
                    words.iter().all(|&w| f32::from_bits(w).is_finite())
                        && words.iter().any(|&w| w != words[0])
                };
                let slot = lane.slot as usize * geometry.rec_slot();
                assert!(
                    live(&stock.gated)
                        && live(&stock.rec)
                        && stock.rec
                            != bits(&step.rec[lane.state_in as usize][slot..slot + geometry.rec_slot()]),
                    "{name} rows {rows:?} lane {index}: the stock chain produced no usable reference"
                );
                assert_lane(
                    &format!("{name} rows {rows:?} lane {index}"),
                    &fused_lane(&step, &lanes, &fused, index),
                    &stock,
                    lane.retain,
                );
            }
            // Containment.
            let value = geometry.value();
            for row in real as usize..width as usize {
                assert!(
                    fused.gated[row * value..(row + 1) * value]
                        .iter()
                        .all(|&bits| bits == 0),
                    "{name} rows {rows:?}: padding row {row} is not zero"
                );
            }
            for side in 0..2 {
                for slot in 0..SLOTS as usize {
                    let written = lanes.iter().any(|lane| {
                        lane.slot as usize == slot && 1 - lane.state_in as usize == side
                    });
                    if written {
                        continue;
                    }
                    let conv = slot * geometry.conv_slot()..(slot + 1) * geometry.conv_slot();
                    let rec = slot * geometry.rec_slot()..(slot + 1) * geometry.rec_slot();
                    assert!(
                        fused.conv[side][conv.clone()] == bits(&step.conv[side][conv])[..]
                            && fused.rec[side][rec.clone()] == bits(&step.rec[side][rec])[..],
                        "{name} rows {rows:?}: state buffer {side} slot {slot} changed"
                    );
                }
            }
            let mut owned = vec![false; fused.tape.len()];
            let nv = geometry.n_v as usize;
            for (lane, tape) in lanes.iter().zip(&fused.tapes) {
                let Some(tape) = tape else { continue };
                let n = lane.rows as usize;
                for (start, elements) in [
                    (tape.q, n * geometry.key()),
                    (tape.k, n * geometry.key()),
                    (tape.v, n * value),
                    (tape.conv_rows, n * geometry.channels()),
                    (tape.g, n * nv),
                    (tape.beta, n * nv),
                ] {
                    owned[start as usize..start as usize + elements].fill(true);
                }
            }
            assert!(
                fused
                    .tape
                    .iter()
                    .zip(&owned)
                    .all(|(&bits, &owned)| owned || bits == SENTINEL.to_bits()),
                "{name} rows {rows:?}: the tape changed outside the lanes' fields"
            );
            // BF16 output: the cast of the F32 output.
            let bf16 = run_fused(
                &mut gpu,
                &step,
                &lanes,
                &padding,
                case % 2 == 0,
                DType::BF16,
            );
            let real_elements = real as usize * value;
            assert!(
                bf16.gated_bf16[..real_elements]
                    == stock_bf16(&mut gpu, &fused.gated[..real_elements])[..],
                "{name} rows {rows:?}: BF16 output is not the cast of the F32 output"
            );
            assert!(bf16.conv == fused.conv && bf16.rec == fused.rec && bf16.tape == fused.tape);
        }
    }
}

#[test]
fn repeated_runs_are_bitwise_equal() {
    let Some(mut gpu) = gpu() else { return };
    for (geometry, name) in [(QWEN38, "16/48"), (QWEN36, "16/32")] {
        let lanes = packed_lanes(&[8, 8, 5, 8]);
        let mut rng = Xor(0xD1B5_4A32_D192_ED03 ^ u64::from(geometry.n_v));
        let step = Step::random(geometry, 32, &mut rng);
        let padding = padding_from(29, 32);
        let first = fused_digest(&run_fused(
            &mut gpu,
            &step,
            &lanes,
            &padding,
            true,
            DType::F32,
        ));
        for _ in 0..4 {
            assert_eq!(
                fused_digest(&run_fused(
                    &mut gpu,
                    &step,
                    &lanes,
                    &padding,
                    true,
                    DType::F32
                )),
                first,
                "{name}: repeated run differs"
            );
        }
        let bf16 = fused_digest(&run_fused(
            &mut gpu,
            &step,
            &lanes,
            &padding,
            true,
            DType::BF16,
        ));
        println!("gdn_window_fused digest {name}: f32 {first:016x} bf16 {bf16:016x}");
    }
}

#[test]
fn a_lane_is_independent_of_its_cohort() {
    let Some(mut gpu) = gpu() else { return };
    for (geometry, name) in [(QWEN38, "16/48"), (QWEN36, "16/32")] {
        let mut rng = Xor(0xA076_1D64_78BD_642F ^ u64::from(geometry.n_v));
        let step = Step::random(geometry, 32, &mut rng);
        let lanes = packed_lanes(&[8, 6, 8, 3]);
        let cohort = run_fused(
            &mut gpu,
            &step,
            &lanes,
            &padding_from(25, 32),
            false,
            DType::F32,
        );
        for (index, lane) in lanes.iter().enumerate() {
            let reference = fused_lane(&step, &lanes, &cohort, index);
            // Alone, at another packed position, not retaining, packed inputs.
            let mut moved = step.clone();
            let (channels, value, nv) =
                (geometry.channels(), geometry.value(), geometry.n_v as usize);
            let target = 32 - lane.rows as usize;
            let n = lane.rows as usize;
            let first = lane.first_row as usize;
            let mut scramble = Xor(0x5851_F42D_4C95_7F2D ^ index as u64);
            moved.qkv = scramble.fill(32 * channels, -9.0, 9.0);
            moved.z = scramble.fill(32 * value, -9.0, 9.0);
            moved.alpha = scramble.fill(32 * nv, -9.0, 9.0);
            moved.beta = scramble.fill(32 * nv, -9.0, 9.0);
            for (field, source, stride) in [
                (&mut moved.qkv, &step.qkv, channels),
                (&mut moved.z, &step.z, value),
                (&mut moved.alpha, &step.alpha, nv),
                (&mut moved.beta, &step.beta, nv),
            ] {
                field[target * stride..(target + n) * stride]
                    .copy_from_slice(&source[first * stride..(first + n) * stride]);
            }
            // Every other slot's state scrambled, in both buffers.
            for side in 0..2 {
                for slot in 0..SLOTS as usize {
                    if slot == lane.slot as usize {
                        continue;
                    }
                    moved.conv[side]
                        [slot * geometry.conv_slot()..(slot + 1) * geometry.conv_slot()]
                        .fill(1e30);
                    moved.rec[side][slot * geometry.rec_slot()..(slot + 1) * geometry.rec_slot()]
                        .fill(-1e30);
                }
            }
            let alone_lane = Lane {
                first_row: target as u32,
                retain: false,
                ..*lane
            };
            let alone = run_fused(&mut gpu, &moved, &[alone_lane], &[], true, DType::F32);
            let got = fused_lane(&moved, &[alone_lane], &alone, 0);
            assert_lane(
                &format!("{name} lane {index} alone"),
                &got,
                &reference,
                false,
            );
            // Rows no lane covers are not written.
            assert!(
                alone.gated[..target * value]
                    .iter()
                    .all(|&bits| bits == SENTINEL.to_bits()),
                "{name} lane {index}: an uncovered row was written"
            );
        }
        // Reversed lane order in the table, same rows.
        let reversed: Vec<Lane> = lanes.iter().rev().copied().collect();
        let other = run_fused(
            &mut gpu,
            &step,
            &reversed,
            &padding_from(25, 32),
            true,
            DType::F32,
        );
        for (index, lane) in lanes.iter().enumerate() {
            assert_lane(
                &format!("{name} lane {index} reversed"),
                &fused_lane(&step, &reversed, &other, lanes.len() - 1 - index),
                &fused_lane(&step, &lanes, &cohort, index),
                lane.retain,
            );
        }
    }
}

#[test]
fn invalid_steps_are_rejected() {
    let Some(mut gpu) = gpu() else { return };
    let geometry = QWEN36;
    let mut rng = Xor(7);
    let step = Step::random(geometry, 8, &mut rng);
    let device = &gpu.device;
    let qkv = buffer(device, DType::F32, &step.qkv);
    let z = buffer(device, DType::F32, &step.z);
    let alpha = buffer(device, DType::F32, &step.alpha);
    let beta = buffer(device, DType::F32, &step.beta);
    let conv_w = buffer(device, DType::F32, &step.conv_w);
    let dt_bias = buffer(device, DType::F32, &step.dt_bias);
    let ssm_a = buffer(device, DType::F32, &step.ssm_a);
    let norm = buffer(device, DType::F32, &step.norm);
    let conv = [
        buffer(device, DType::F32, &step.conv[0]),
        buffer(device, DType::F32, &step.conv[1]),
    ];
    let rec = [
        buffer(device, DType::F32, &step.rec[0]),
        buffer(device, DType::F32, &step.rec[1]),
    ];
    let gated = zeros(device, 8 * geometry.value());
    let small = zeros(device, 16);
    let window = |first_row, rows, slot, state_in, tape| GdnWindowLane::Window {
        first_row,
        rows,
        slot,
        state_in,
        tape,
    };
    let tape_at = GdnWindowTape {
        q: 0,
        k: 0,
        v: 0,
        conv_rows: 0,
        g: 0,
        beta: 0,
    };
    let mut attempt = |shape: GdnWindowShape,
                       qkv_stride: u32,
                       gated: &MlxBuffer,
                       tape: Option<&MlxBuffer>,
                       lanes: &[GdnWindowLane]| {
        let mut enc = device.command_encoder().expect("encoder");
        dispatch_gdn_window_fused(
            &mut enc,
            &mut gpu.registry,
            device,
            &shape,
            &GdnWindowInputs {
                qkv: GdnWindowRows {
                    buffer: &qkv,
                    stride: qkv_stride,
                },
                z: GdnWindowRows {
                    buffer: &z,
                    stride: geometry.value() as u32,
                },
                alpha: GdnWindowRows {
                    buffer: &alpha,
                    stride: geometry.n_v,
                },
                beta: GdnWindowRows {
                    buffer: &beta,
                    stride: geometry.n_v,
                },
            },
            &GdnWindowWeights {
                conv: &conv_w,
                dt_bias: &dt_bias,
                ssm_a: &ssm_a,
                norm: &norm,
            },
            &GdnWindowState {
                conv: [&conv[0], &conv[1]],
                recurrent: [&rec[0], &rec[1]],
            },
            gated,
            tape,
            8,
            lanes,
        )
        .is_err()
    };
    let shape = geometry.shape();
    let channels = geometry.channels() as u32;
    let ok = [window(0, 8, 0, 0, None)];
    assert!(
        !attempt(shape, channels, &gated, None, &ok),
        "a valid step was rejected"
    );
    assert!(attempt(
        GdnWindowShape {
            head_dim: 64,
            ..shape
        },
        channels,
        &gated,
        None,
        &ok
    ));
    assert!(attempt(
        GdnWindowShape {
            n_k_heads: 5,
            ..shape
        },
        channels,
        &gated,
        None,
        &ok
    ));
    assert!(attempt(
        GdnWindowShape {
            conv_taps: 1,
            ..shape
        },
        channels,
        &gated,
        None,
        &ok
    ));
    assert!(
        attempt(shape, channels - 1, &gated, None, &ok),
        "narrow stride"
    );
    assert!(
        attempt(shape, channels + 1, &gated, None, &ok),
        "rows beyond the input"
    );
    assert!(attempt(shape, channels, &small, None, &ok), "short output");
    assert!(attempt(shape, channels, &gated, None, &[]), "no lanes");
    assert!(
        attempt(shape, channels, &gated, None, &[window(0, 9, 0, 0, None)]),
        "nine rows"
    );
    assert!(
        attempt(shape, channels, &gated, None, &[window(4, 5, 0, 0, None)]),
        "past the width"
    );
    assert!(
        attempt(
            shape,
            channels,
            &gated,
            None,
            &[window(0, 2, SLOTS, 0, None)]
        ),
        "slot"
    );
    assert!(
        attempt(shape, channels, &gated, None, &[window(0, 2, 0, 2, None)]),
        "state buffer"
    );
    assert!(
        attempt(
            shape,
            channels,
            &gated,
            None,
            &[window(0, 4, 0, 0, None), window(3, 2, 1, 0, None)]
        ),
        "overlapping rows"
    );
    assert!(
        attempt(
            shape,
            channels,
            &gated,
            None,
            &[window(0, 4, 1, 0, None), window(4, 2, 1, 1, None)]
        ),
        "repeated slot"
    );
    assert!(
        attempt(
            shape,
            channels,
            &gated,
            None,
            &[window(0, 4, 0, 0, Some(tape_at))]
        ),
        "retaining without a tape"
    );
    assert!(
        attempt(
            shape,
            channels,
            &gated,
            Some(&small),
            &[window(0, 4, 0, 0, Some(tape_at))]
        ),
        "short tape"
    );
    assert!(
        attempt(
            shape,
            channels,
            &gated,
            None,
            &[
                window(0, 4, 0, 0, None),
                GdnWindowLane::Padding {
                    first_row: 3,
                    rows: 2
                }
            ]
        ),
        "padding over a lane"
    );
}
