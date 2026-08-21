//! Exact GPU deinterleave for fused Qwen Q/gate projection activations.
//!
//! Qwen stores each attention head's fused projection rows as
//! `[Q(head_dim), gate(head_dim)]`. A projection through that matrix therefore
//! produces token-major activations with logical shape
//! `[m, n_heads, 2 * head_dim]`. This operation copies those raw F32 payloads
//! into separate Q and gate tensors of shape `[m, n_heads, head_dim]`.
//!
//! The Metal kernel reads and writes `uint` payloads deliberately: this is a
//! layout transform, not floating-point math, so every bit (including signed
//! zero and NaN payloads) is preserved exactly.

use metal::foreign_types::ForeignType;
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::as_bytes;

const KERNEL_NAME: &str = "q_gate_deinterleave_f32";

/// MSL source for the fused Q/gate activation deinterleave kernel.
pub static Q_GATE_DEINTERLEAVE_SHADER_SOURCE: &str =
    include_str!("../shaders/q_gate_deinterleave.metal");

/// Register the fused Q/gate activation deinterleave shader.
///
/// [`KernelRegistry::new`] registers this source automatically; this helper is
/// provided for callers constructing or extending registries explicitly.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(KERNEL_NAME, Q_GATE_DEINTERLEAVE_SHADER_SOURCE);
}

/// Logical dimensions for fused Q/gate activation deinterleave.
///
/// This layout is shared directly with the Metal shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QGateDeinterleaveParams {
    /// Number of input rows (decode batch or prefill length).
    pub m: u32,
    /// Number of query heads.
    pub n_heads: u32,
    /// Width of one Q or gate head.
    pub head_dim: u32,
}

#[derive(Clone, Copy)]
struct LogicalRange {
    buffer_id: usize,
    start: u64,
    end: u64,
}

impl LogicalRange {
    fn new(buffer: &MlxBuffer, logical_bytes: usize) -> Result<Self> {
        let logical_bytes = u64::try_from(logical_bytes).map_err(|_| {
            MlxError::InvalidArgument(
                "q_gate_deinterleave_f32: logical byte length exceeds u64".into(),
            )
        })?;
        let end = buffer
            .byte_offset()
            .checked_add(logical_bytes)
            .ok_or_else(|| {
                MlxError::InvalidArgument(
                    "q_gate_deinterleave_f32: logical buffer range overflows u64".into(),
                )
            })?;
        Ok(Self {
            buffer_id: buffer.metal_buffer().as_ptr() as usize,
            start: buffer.byte_offset(),
            end,
        })
    }

    fn overlaps(self, other: Self) -> bool {
        self.buffer_id == other.buffer_id && self.start < other.end && other.start < self.end
    }
}

fn validate_buffer(
    buffer: &MlxBuffer,
    name: &str,
    shape: &[usize],
    required_bytes: usize,
    writable: bool,
) -> Result<()> {
    if buffer.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "q_gate_deinterleave_f32: {name} dtype must be F32, got {}",
            buffer.dtype()
        )));
    }
    if buffer.shape() != shape {
        return Err(MlxError::InvalidArgument(format!(
            "q_gate_deinterleave_f32: {name} shape must be {shape:?}, got {:?}",
            buffer.shape()
        )));
    }
    if buffer.data_byte_len() < required_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "q_gate_deinterleave_f32: {name} logical buffer too small: need {required_bytes} bytes, have {}",
            buffer.data_byte_len()
        )));
    }
    if writable && !buffer.is_cpu_writable() {
        return Err(MlxError::InvalidArgument(format!(
            "q_gate_deinterleave_f32: {name} must be writable"
        )));
    }
    Ok(())
}

/// Deinterleave fused Q/gate activations with exact F32 payload copies.
///
/// Input layout:
///
/// ```text
/// fused[token, head, :] = [Q(head_dim), gate(head_dim)]
/// ```
///
/// Output layouts are `q[token, head, head_dim]` and
/// `gate[token, head, head_dim]`. The input and outputs must not overlap.
/// A single Metal thread copies the Q and gate payload for one
/// `(token, head, column)` coordinate.
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] before encoding any work when:
///
/// - a dimension is zero or the logical element count exceeds `u32` indexing;
/// - any buffer is not F32 or does not have its exact documented shape;
/// - a logical buffer is too short, an output is read-only, or buffers overlap.
pub fn dispatch_q_gate_deinterleave_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    fused: &MlxBuffer,
    q: &MlxBuffer,
    gate: &MlxBuffer,
    params: QGateDeinterleaveParams,
) -> Result<()> {
    if params.m == 0 || params.n_heads == 0 || params.head_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "q_gate_deinterleave_f32: m, n_heads, and head_dim must be > 0".into(),
        ));
    }

    // The shader uses u32 indices. Prove every multiplication below fits that
    // contract before compiling a pipeline or recording a dispatch.
    let fused_head_dim = params.head_dim.checked_mul(2).ok_or_else(|| {
        MlxError::InvalidArgument("q_gate_deinterleave_f32: 2 * head_dim overflows u32".into())
    })?;
    let output_elements_u32 = params
        .m
        .checked_mul(params.n_heads)
        .and_then(|count| count.checked_mul(params.head_dim))
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "q_gate_deinterleave_f32: output element count overflows u32".into(),
            )
        })?;
    let input_elements_u32 = output_elements_u32.checked_mul(2).ok_or_else(|| {
        MlxError::InvalidArgument(
            "q_gate_deinterleave_f32: input element count overflows u32".into(),
        )
    })?;

    let m = params.m as usize;
    let n_heads = params.n_heads as usize;
    let head_dim = params.head_dim as usize;
    let fused_shape = [m, n_heads, fused_head_dim as usize];
    let output_shape = [m, n_heads, head_dim];
    let input_bytes = (input_elements_u32 as usize) * DType::F32.size_of();
    let output_bytes = (output_elements_u32 as usize) * DType::F32.size_of();

    validate_buffer(fused, "fused", &fused_shape, input_bytes, false)?;
    validate_buffer(q, "q", &output_shape, output_bytes, true)?;
    validate_buffer(gate, "gate", &output_shape, output_bytes, true)?;

    let fused_range = LogicalRange::new(fused, input_bytes)?;
    let q_range = LogicalRange::new(q, output_bytes)?;
    let gate_range = LogicalRange::new(gate, output_bytes)?;
    if fused_range.overlaps(q_range)
        || fused_range.overlaps(gate_range)
        || q_range.overlaps(gate_range)
    {
        return Err(MlxError::InvalidArgument(
            "q_gate_deinterleave_f32: fused, q, and gate logical ranges must not overlap".into(),
        ));
    }

    let pipeline = registry.get_pipeline(KERNEL_NAME, device.metal_device())?;
    let threads_x = u64::from(params.head_dim.min(256));
    let threadgroups = MTLSize::new(
        u64::from(params.head_dim).div_ceil(threads_x),
        u64::from(params.n_heads),
        u64::from(params.m),
    );
    let threads_per_threadgroup = MTLSize::new(threads_x, 1, 1);

    encoder.dispatch_tracked_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(fused)),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(gate)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        &[fused],
        &[q, gate],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}
