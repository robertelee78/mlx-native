//! Fused GPU kernel for DeltaNet g and beta computation.
//!
//! `g[t, vh]    = softplus(alpha_logit[t, vh] + dt_bias[vh]) * (-ssm_a[vh])`
//! `beta[t, vh] = sigmoid(beta_logit[t, vh])`
//!
//! Replaces the CPU bridge in `compute_g_and_beta_cpu`.
//! For seq=1 this dispatches 1×nv threads — trivial GPU work but eliminates
//! 2 CPU-GPU buffer downloads (alpha_logit, beta_logit) per delta-net layer.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static COMPUTE_G_BETA_SHADER_SOURCE: &str = include_str!("../shaders/compute_g_beta.metal");

/// Register `compute_g_beta_f32` shader with the kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("compute_g_beta_f32", COMPUTE_G_BETA_SHADER_SOURCE);
}

/// Dispatch `compute_g_beta` kernel into `encoder`.
///
/// All f32 buffers. `alpha_logit` and `beta_logit` are `[seq, nv]`.
/// `dt_bias` and `ssm_a` are `[nv]`. `g_out` and `beta_out` are `[seq, nv]`.
/// `params_buf` must hold `[nv: u32, seq: u32]` (8 bytes).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_compute_g_beta(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    alpha_logit: &MlxBuffer,
    beta_logit: &MlxBuffer,
    dt_bias: &MlxBuffer,
    ssm_a: &MlxBuffer,
    g_out: &MlxBuffer,
    beta_out: &MlxBuffer,
    params_buf: &MlxBuffer,
    seq: u32,
    nv: u32,
) -> Result<()> {
    let n = seq * nv;
    if n == 0 {
        return Err(MlxError::InvalidArgument("compute_g_beta: n must be > 0".into()));
    }
    let pipeline = registry.get_pipeline("compute_g_beta_f32", device)?;
    let tg = MTLSize::new(std::cmp::min(n as u64, 256), 1, 1);
    let grid = MTLSize::new(n as u64, 1, 1);
    encoder.encode(
        pipeline,
        &[
            (0, alpha_logit),
            (1, beta_logit),
            (2, dt_bias),
            (3, ssm_a),
            (4, g_out),
            (5, beta_out),
            (6, params_buf),
        ],
        grid,
        tg,
    );
    Ok(())
}

/// Allocate output buffers, dispatch the kernel, commit, and return `(g, beta)`.
///
/// Both returned buffers are `[seq * nv]` f32 in flat token-major layout.
#[allow(clippy::too_many_arguments)]
pub fn compute_g_beta_gpu(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    alpha_logit: &MlxBuffer,
    beta_logit: &MlxBuffer,
    dt_bias: &MlxBuffer,
    ssm_a: &MlxBuffer,
    seq: u32,
    nv: u32,
) -> Result<(MlxBuffer, MlxBuffer)> {
    let n = (seq * nv) as usize;
    let g_out = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: alloc g: {e}")))?;
    let beta_out = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: alloc beta: {e}")))?;

    let mut params_buf = device
        .alloc_buffer(8, DType::U32, vec![2])
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: alloc params: {e}")))?;
    params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: write params: {e}")))?
        [0] = nv;
    params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: write params2: {e}")))?
        [1] = seq;

    let mut enc = device
        .command_encoder()
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: command_encoder: {e}")))?;
    dispatch_compute_g_beta(
        &mut enc,
        registry,
        device.metal_device(),
        alpha_logit,
        beta_logit,
        dt_bias,
        ssm_a,
        &g_out,
        &beta_out,
        &params_buf,
        seq,
        nv,
    )?;
    enc.commit_and_wait()
        .map_err(|e| MlxError::InvalidArgument(format!("compute_g_beta_gpu: commit: {e}")))?;

    Ok((g_out, beta_out))
}
