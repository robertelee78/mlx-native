//! Gated DeltaNet core of a joint verify step in one dispatch.
//!
//! A joint step carries, per lane, a window of 1-[`WINDOW_ROWS`] rows at
//! consecutive positions of the lane's cache slot. The stock chain issues,
//! per layer, one `ssm_conv` per window, one `l2_norm` for Q and one for K
//! per row, one `scalar_mul` per row, `compute_g_beta`, one
//! `gated_delta_net_decode` per window and `ssm_norm_gate`, in five
//! serialized stages, and copies each window's raw rows and V rows for the
//! replay commit. [`dispatch_gdn_window_fused`] is all of it in one dispatch:
//! a threadgroup owns one value head of one lane and hands the stages their
//! operands in threadgroup memory.
//!
//! Portions derived from Splash commit 031a25a6 (Apache-2.0): the structure
//! of its fused verify kernel. See `shaders/gdn_window_fused.metal` for the
//! source and the modifications and `LICENSE-APACHE-splash` at the repository
//! root.
//!
//! Numerics: every stage repeats the stock kernel's operations in its order
//! (the 128-thread tree of the norm reductions included), so the gated
//! output, both states and the retained rows are bitwise those of the stock
//! chain (`tests/test_gdn_window_fused.rs`). The BF16 output is the F32
//! output rounded as `cast_f32_to_bf16` rounds it.
//!
//! Invariants: a lane's outputs and state depend only on its own rows, its
//! slot's state and the weights; not on the other lanes, on its packed
//! position, or on the padding.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

/// MSL source.
pub static GDN_WINDOW_FUSED_SHADER_SOURCE: &str = include_str!("../shaders/gdn_window_fused.metal");

/// Most rows of one lane's window (the kernel's `kGdnwRows`).
pub const WINDOW_ROWS: u32 = 8;
/// Key and value head dimension the kernel is compiled for.
pub const HEAD_DIM: u32 = 128;
/// Most lanes of one dispatch (the lane table is bound inline).
pub const MAX_LANES: usize = 32;

const KERNEL_F32: &str = "gdn_window_fused_f32";
const KERNEL_BF16: &str = "gdn_window_fused_bf16";
const THREADS: u64 = 256;
/// `sizeof(GdnWindowShared)`: four `[8][128]` F32 operand tiles and the
/// rows' gates.
const SHARED_BYTES: u64 = 4 * 8 * 128 * 4 + 2 * 8 * 4;
const FLAG_RETAIN: u32 = 1;
const FLAG_PADDING: u32 = 2;

/// Register both entry points against the shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(KERNEL_F32, GDN_WINDOW_FUSED_SHADER_SOURCE);
    registry.register_source(KERNEL_BF16, GDN_WINDOW_FUSED_SHADER_SOURCE);
}

fn invalid(msg: String) -> MlxError {
    MlxError::InvalidArgument(format!("gdn_window_fused: {msg}"))
}

/// Geometry of one DeltaNet layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnWindowShape {
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    /// Key and value head dimension; must be [`HEAD_DIM`].
    pub head_dim: u32,
    /// Convolution taps (`K`); the slot carries `K - 1` rows.
    pub conv_taps: u32,
    /// Epsilon of the Q/K L2 norms and of the output RMS norm.
    pub eps: f32,
    /// Scale applied to the L2-normalized queries.
    pub q_scale: f32,
    /// Cache slots of the state buffers.
    pub slots: u32,
}

impl GdnWindowShape {
    fn key_width(&self) -> usize {
        self.n_k_heads as usize * HEAD_DIM as usize
    }

    fn value_width(&self) -> usize {
        self.n_v_heads as usize * HEAD_DIM as usize
    }

    /// Convolution channels: Q, K and V.
    pub fn channels(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }

    fn conv_slot(&self) -> usize {
        self.channels() * (self.conv_taps as usize - 1)
    }

    fn recurrent_slot(&self) -> usize {
        self.value_width() * HEAD_DIM as usize
    }

    /// Whether the kernel serves this geometry.
    pub fn supported(&self) -> bool {
        self.head_dim == HEAD_DIM
            && self.n_k_heads > 0
            && self.n_v_heads >= self.n_k_heads
            && self.n_v_heads % self.n_k_heads == 0
            && (2..=WINDOW_ROWS + 1).contains(&self.conv_taps)
            && self.slots > 0
    }
}

/// A packed F32 row matrix: `buffer` starts at column 0 of packed row 0 and
/// holds `stride` elements per row, of which the leading ones are the field.
#[derive(Clone, Copy)]
pub struct GdnWindowRows<'a> {
    pub buffer: &'a MlxBuffer,
    pub stride: u32,
}

/// Inputs of one layer over the packed rows.
#[derive(Clone, Copy)]
pub struct GdnWindowInputs<'a> {
    /// Q, K and V projection rows before the convolution.
    pub qkv: GdnWindowRows<'a>,
    /// Output gate rows.
    pub z: GdnWindowRows<'a>,
    /// Alpha and beta logit rows.
    pub alpha: GdnWindowRows<'a>,
    pub beta: GdnWindowRows<'a>,
}

/// Weights of one layer, all F32.
#[derive(Clone, Copy)]
pub struct GdnWindowWeights<'a> {
    /// `[channels][conv_taps]`.
    pub conv: &'a MlxBuffer,
    /// `[n_v_heads]`.
    pub dt_bias: &'a MlxBuffer,
    /// `[n_v_heads]`.
    pub ssm_a: &'a MlxBuffer,
    /// `[HEAD_DIM]`.
    pub norm: &'a MlxBuffer,
}

/// The layer's two physical state buffers of each kind, slot-major:
/// convolution `[slots][channels][conv_taps - 1]`, recurrent
/// `[slots][n_v_heads][HEAD_DIM][HEAD_DIM]`. A lane reads its slot's region
/// of one and writes the same region of the other.
#[derive(Clone, Copy)]
pub struct GdnWindowState<'a> {
    pub conv: [&'a MlxBuffer; 2],
    pub recurrent: [&'a MlxBuffer; 2],
}

/// Where a lane's rows are retained in the tape buffer (element offsets):
/// scaled Q and normalized K `[rows][n_k_heads * HEAD_DIM]`, V
/// `[rows][n_v_heads * HEAD_DIM]`, the raw QKV rows `[rows][channels]`, and
/// `g` and `beta` `[rows][n_v_heads]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnWindowTape {
    pub q: u32,
    pub k: u32,
    pub v: u32,
    pub conv_rows: u32,
    pub g: u32,
    pub beta: u32,
}

/// One lane of a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnWindowLane {
    /// `rows` rows at packed rows `first_row..`, of cache slot `slot`, whose
    /// current state is in physical buffer `state_in` (0 or 1); the state
    /// after the rows goes to the other buffer.
    Window {
        first_row: u32,
        rows: u32,
        slot: u32,
        state_in: u32,
        tape: Option<GdnWindowTape>,
    },
    /// `rows` packed rows from `first_row` that belong to no lane: their
    /// output rows are zeroed.
    Padding { first_row: u32, rows: u32 },
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_k_heads: u32,
    n_v_heads: u32,
    channels: u32,
    k_width: u32,
    qkv_stride: u32,
    z_stride: u32,
    alpha_stride: u32,
    beta_stride: u32,
    out_stride: u32,
    eps: f32,
    q_scale: f32,
    head_dim: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LaneParams {
    first_row: u32,
    rows: u32,
    flags: u32,
    conv_in: u32,
    conv_offset: u32,
    rec_in: u32,
    rec_offset: u32,
    tape_q: u32,
    tape_k: u32,
    tape_v: u32,
    tape_conv: u32,
    tape_g: u32,
    tape_beta: u32,
    reserved: [u32; 3],
}

/// Whether the pipelines can run on this device: 256-thread threadgroups and
/// the bound threadgroup memory within the device's limits.
pub fn gdn_window_fused_available(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<bool> {
    let metal_device = device.metal_device();
    let memory_limit = metal_device.max_threadgroup_memory_length();
    for name in [KERNEL_F32, KERNEL_BF16] {
        let pipeline = registry.get_pipeline(name, metal_device)?;
        if pipeline.max_total_threads_per_threadgroup() < THREADS
            || pipeline
                .static_threadgroup_memory_length()
                .saturating_add(SHARED_BYTES)
                > memory_limit
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn require(buffer: &MlxBuffer, dtype: DType, elements: usize, name: &str) -> Result<()> {
    if buffer.dtype() != dtype || buffer.element_count() < elements {
        return Err(invalid(format!(
            "{name} holds {} {} elements, needs {elements} {dtype}",
            buffer.element_count(),
            buffer.dtype()
        )));
    }
    Ok(())
}

/// Encode the DeltaNet core of every lane of a joint step: convolution with
/// SiLU, the Q and K L2 norms, the query scale, the gates, the delta-rule
/// recurrence and the gated output RMS norm.
///
/// `gated` is `[width][n_v_heads * HEAD_DIM]`, F32 or BF16. Each window lane
/// writes its rows of it, its slot's state after the rows into the buffer
/// opposite to `state_in`, and, with a tape, the rows the replay commit
/// reads. The lanes' rows must be disjoint and their slots distinct; `tape`
/// is required when any lane retains.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_window_fused(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    shape: &GdnWindowShape,
    inputs: &GdnWindowInputs<'_>,
    weights: &GdnWindowWeights<'_>,
    state: &GdnWindowState<'_>,
    gated: &MlxBuffer,
    tape: Option<&MlxBuffer>,
    width: u32,
    lanes: &[GdnWindowLane],
) -> Result<()> {
    if !shape.supported() {
        return Err(invalid(format!("unsupported geometry {shape:?}")));
    }
    if lanes.is_empty() || lanes.len() > MAX_LANES || width == 0 {
        return Err(invalid(format!(
            "{} lanes over {width} packed rows (1..={MAX_LANES} lanes)",
            lanes.len()
        )));
    }
    let kernel = match gated.dtype() {
        DType::F32 => KERNEL_F32,
        DType::BF16 => KERNEL_BF16,
        other => {
            return Err(invalid(format!(
                "gated output must be F32 or BF16, got {other}"
            )))
        }
    };
    let key_width = shape.key_width();
    let value_width = shape.value_width();
    let channels = shape.channels();
    let nv = shape.n_v_heads as usize;
    let rows = width as usize;
    let field = |rows_of: &GdnWindowRows<'_>, columns: usize, name: &str| -> Result<()> {
        if (rows_of.stride as usize) < columns {
            return Err(invalid(format!(
                "{name} stride {} is narrower than its {columns} columns",
                rows_of.stride
            )));
        }
        require(
            rows_of.buffer,
            DType::F32,
            (rows - 1) * rows_of.stride as usize + columns,
            name,
        )
    };
    field(&inputs.qkv, channels, "qkv")?;
    field(&inputs.z, value_width, "z")?;
    field(&inputs.alpha, nv, "alpha")?;
    field(&inputs.beta, nv, "beta")?;
    require(
        weights.conv,
        DType::F32,
        channels * shape.conv_taps as usize,
        "conv weight",
    )?;
    require(weights.dt_bias, DType::F32, nv, "dt_bias")?;
    require(weights.ssm_a, DType::F32, nv, "ssm_a")?;
    require(weights.norm, DType::F32, HEAD_DIM as usize, "norm weight")?;
    let slots = shape.slots as usize;
    for buffer in state.conv {
        require(buffer, DType::F32, slots * shape.conv_slot(), "conv state")?;
    }
    for buffer in state.recurrent {
        require(
            buffer,
            DType::F32,
            slots * shape.recurrent_slot(),
            "recurrent state",
        )?;
    }
    require(gated, gated.dtype(), rows * value_width, "gated output")?;

    let offset = |elements: usize| {
        u32::try_from(elements).map_err(|_| invalid("an element offset exceeds u32".into()))
    };
    let mut covered = vec![false; rows];
    let mut slots_seen = Vec::with_capacity(lanes.len());
    let mut table = [LaneParams::default(); MAX_LANES];
    for (entry, lane) in table.iter_mut().zip(lanes) {
        let (first_row, lane_rows) = match *lane {
            GdnWindowLane::Window {
                first_row, rows, ..
            }
            | GdnWindowLane::Padding { first_row, rows } => (first_row, rows),
        };
        if lane_rows == 0
            || lane_rows > WINDOW_ROWS
            || first_row as usize + lane_rows as usize > rows
        {
            return Err(invalid(format!(
                "lane rows {first_row}..{} outside 1..={WINDOW_ROWS} rows of {rows} packed rows",
                first_row + lane_rows
            )));
        }
        for row in &mut covered[first_row as usize..(first_row + lane_rows) as usize] {
            if std::mem::replace(row, true) {
                return Err(invalid(format!("lanes overlap at packed row {first_row}")));
            }
        }
        entry.first_row = first_row;
        entry.rows = lane_rows;
        match *lane {
            GdnWindowLane::Padding { .. } => entry.flags = FLAG_PADDING,
            GdnWindowLane::Window {
                slot,
                state_in,
                tape: lane_tape,
                ..
            } => {
                if slot >= shape.slots || state_in > 1 || slots_seen.contains(&slot) {
                    return Err(invalid(format!(
                        "lane slot {slot} (of {}) with state buffer {state_in} is invalid or repeated",
                        shape.slots
                    )));
                }
                slots_seen.push(slot);
                entry.conv_in = state_in;
                entry.rec_in = state_in;
                entry.conv_offset = offset(slot as usize * shape.conv_slot())?;
                entry.rec_offset = offset(slot as usize * shape.recurrent_slot())?;
                if let Some(lane_tape) = lane_tape {
                    let tape =
                        tape.ok_or_else(|| invalid("a lane retains without a tape".into()))?;
                    let n = lane_rows as usize;
                    for (start, elements, name) in [
                        (lane_tape.q, n * key_width, "tape q"),
                        (lane_tape.k, n * key_width, "tape k"),
                        (lane_tape.v, n * value_width, "tape v"),
                        (lane_tape.conv_rows, n * channels, "tape conv rows"),
                        (lane_tape.g, n * nv, "tape g"),
                        (lane_tape.beta, n * nv, "tape beta"),
                    ] {
                        require(tape, DType::F32, start as usize + elements, name)?;
                    }
                    entry.flags = FLAG_RETAIN;
                    entry.tape_q = lane_tape.q;
                    entry.tape_k = lane_tape.k;
                    entry.tape_v = lane_tape.v;
                    entry.tape_conv = lane_tape.conv_rows;
                    entry.tape_g = lane_tape.g;
                    entry.tape_beta = lane_tape.beta;
                }
            }
        }
    }

    let metal_device = device.metal_device();
    let pipeline = registry.get_pipeline(kernel, metal_device)?;
    if pipeline.max_total_threads_per_threadgroup() < THREADS
        || pipeline
            .static_threadgroup_memory_length()
            .saturating_add(SHARED_BYTES)
            > metal_device.max_threadgroup_memory_length()
    {
        return Err(invalid(format!(
            "{kernel} exceeds the device's threadgroup limits; check gdn_window_fused_available"
        )));
    }
    let params = Params {
        n_k_heads: shape.n_k_heads,
        n_v_heads: shape.n_v_heads,
        channels: offset(channels)?,
        k_width: shape.conv_taps,
        qkv_stride: inputs.qkv.stride,
        z_stride: inputs.z.stride,
        alpha_stride: inputs.alpha.stride,
        beta_stride: inputs.beta.stride,
        out_stride: offset(value_width)?,
        eps: shape.eps,
        q_scale: shape.q_scale,
        head_dim: HEAD_DIM as f32,
    };
    let table = &table[..lanes.len()];
    // A step without a tape still binds a buffer: no lane writes it.
    let tape = tape.unwrap_or(weights.dt_bias);
    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Buffer(inputs.qkv.buffer)),
            (1, KernelArg::Buffer(inputs.z.buffer)),
            (2, KernelArg::Buffer(inputs.alpha.buffer)),
            (3, KernelArg::Buffer(inputs.beta.buffer)),
            (4, KernelArg::Buffer(weights.conv)),
            (5, KernelArg::Buffer(state.conv[0])),
            (6, KernelArg::Buffer(state.conv[1])),
            (7, KernelArg::Buffer(state.recurrent[0])),
            (8, KernelArg::Buffer(state.recurrent[1])),
            (9, KernelArg::Buffer(weights.dt_bias)),
            (10, KernelArg::Buffer(weights.ssm_a)),
            (11, KernelArg::Buffer(weights.norm)),
            (12, KernelArg::Buffer(gated)),
            (13, KernelArg::Buffer(tape)),
            (14, KernelArg::Bytes(as_bytes(&params))),
            (15, KernelArg::Bytes(bytemuck::cast_slice(table))),
        ],
        &[(0, SHARED_BYTES)],
        MTLSize::new(u64::from(shape.n_v_heads), lanes.len() as u64, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}
