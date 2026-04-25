//! Fused SiLU-gated multiply: `out[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
//!
//! Used by Qwen3.5 MoE FFN SwiGLU activation step (ADR-013).
//! Replaces the CPU bridge: `download gate_all + download up_all + silu_mul_cpu + upload h_all`
//! with a single GPU kernel dispatch, eliminating 3 CPU-GPU round-trips per FFN layer.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SILU_MUL_SHADER_SOURCE: &str = include_str!("../shaders/silu_mul.metal");

/// Register `silu_mul_f32` shader with the kernel registry.
///
/// Must be called before dispatching any SiLU-multiply operations.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("silu_mul_f32", SILU_MUL_SHADER_SOURCE);
}

/// Dispatch `output[i] = gate[i] * sigmoid(gate[i]) * up[i]` on the GPU.
///
/// All three buffers must be f32 and contain exactly `n` elements.
/// The `params_buf` must hold a single `u32 n` (4 bytes) and **must remain
/// alive until the encoder commits**.
///
/// # Arguments
///
/// * `encoder`   - Command encoder to record the dispatch into.
/// * `registry`  - Kernel registry (must have silu_mul sources registered).
/// * `device`    - Metal device for pipeline compilation.
/// * `gate`      - Gate values buffer (f32, `n` elements).
/// * `up`        - Up-projection values buffer (f32, `n` elements).
/// * `output`    - Output buffer (f32, `n` elements).
/// * `params_buf`- Buffer holding a single `u32 n` (4 bytes). **Caller must
///   keep this alive until after `encoder.commit_and_wait()` completes.**
/// * `n`         - Number of elements.
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] if `n` is zero or any buffer is too small.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_silu_mul(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate: &MlxBuffer,
    up: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "silu_mul: n must be > 0".into(),
        ));
    }
    let expected = n as usize;
    let elem_bytes = expected * DType::F32.size_of();

    for (name, buf) in [("gate", gate), ("up", up), ("output", output)] {
        if buf.byte_len() < elem_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "silu_mul: {name} buffer too small: need {elem_bytes} bytes, have {}",
                buf.byte_len()
            )));
        }
    }
    if params_buf.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "silu_mul: params_buf too small: need 4 bytes, have {}",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("silu_mul_f32", device)?;
    let tg = MTLSize::new(std::cmp::min(n as u64, 256), 1, 1);
    let grid = MTLSize::new(n as u64, 1, 1);

    encoder.encode(
        pipeline,
        &[(0, gate), (1, up), (2, output), (3, params_buf)],
        grid,
        tg,
    );
    Ok(())
}

/// Allocate, dispatch, commit, and return `out[i] = silu(gate[i]) * up[i]`.
///
/// Creates a new command encoder, encodes the silu_mul kernel, commits it,
/// and waits for completion before returning. The `params_buf` lifetime is
/// fully contained within this function — no use-after-free risk.
///
/// # Arguments
///
/// * `registry`  - Kernel registry.
/// * `device`    - Metal device.
/// * `gate`      - Gate projection buffer (f32, `n` elements). Must be committed.
/// * `up`        - Up projection buffer (f32, `n` elements). Must be committed.
/// * `n`         - Number of elements.
///
/// # Returns
///
/// A new f32 buffer `[n]` containing `silu(gate) * up`.
pub fn silu_mul_gpu(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate: &MlxBuffer,
    up: &MlxBuffer,
    n: u32,
) -> Result<MlxBuffer> {
    let n_usize = n as usize;
    let out_bytes = n_usize * DType::F32.size_of();
    let output = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_usize])
        .map_err(|e| MlxError::InvalidArgument(format!("silu_mul_gpu: alloc output: {e}")))?;

    let mut params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| MlxError::InvalidArgument(format!("silu_mul_gpu: alloc params: {e}")))?;

    // Write n into params buffer via CPU mapping (4 bytes, one-time setup per call)
    params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| MlxError::InvalidArgument(format!("silu_mul_gpu: write params: {e}")))?[0] = n;

    // Create encoder, dispatch, commit — params_buf is still alive here
    let mut enc = device
        .command_encoder()
        .map_err(|e| MlxError::InvalidArgument(format!("silu_mul_gpu: command_encoder: {e}")))?;
    dispatch_silu_mul(&mut enc, registry, device.metal_device(), gate, up, &output, &params_buf, n)?;
    enc.commit_and_wait()
        .map_err(|e| MlxError::InvalidArgument(format!("silu_mul_gpu: commit: {e}")))?;
    // params_buf dropped here, AFTER commit_and_wait completes — safe

    Ok(output)
}
