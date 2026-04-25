//! GPU MoE weighted accumulate + shared expert add + optional residual.
//!
//! Replaces the CPU weighted accumulate loop in `build_moe_ffn_layer_gpu_q`
//! (Step 3e) and folds in the shared expert gated addition (Step 5+sh_gate add)
//! and optionally the post-FFN residual, all in one GPU kernel.
//!
//! This eliminates:
//!   - CPU download of `y_all` (n_tokens * top_k * h floats)
//!   - CPU download of `y_s` (n_tokens * h floats)
//!   - CPU download of `sh_logit` (n_tokens floats)
//!   - CPU weighted accumulate loop
//!   - CPU sigmoid(sh_logit) * y_s addition
//!   - The `residual_add_gpu` commit (for MoeQ path)

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Metal shader source.
pub static MOE_WEIGHTED_REDUCE_SHADER_SOURCE: &str =
    include_str!("../shaders/moe_weighted_reduce.metal");

/// Register the `moe_weighted_reduce_f32` pipeline.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("moe_weighted_reduce_f32", MOE_WEIGHTED_REDUCE_SHADER_SOURCE);
}

/// GPU-side params struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MoeWeightedReduceGpuParams {
    n_tokens:     u32,
    top_k:        u32,
    h:            u32,
    add_residual: u32,
}

const TG_SIZE: u64 = 256;

/// Dispatch fused MoE weighted accumulate + shared expert + optional residual.
///
/// Computes:
/// `output[t, h] = sum_{k} expert_w[t*top_k+k] * y_expert[(t*top_k+k), h]
///                 + sh_gate[t] * y_shared[t, h]
///                 [+ residual[t, h]]`
///
/// # Arguments
///
/// * `expert_w`   — F32 `[n_tokens * top_k]` routing weights (post-renorm).
/// * `y_expert`   — F32 `[n_tokens * top_k, h]` expert down-projection outputs.
/// * `sh_gate`    — F32 `[n_tokens]` shared gate scalars (sigmoid of sh_logit).
/// * `y_shared`   — F32 `[n_tokens, h]` shared expert output.
/// * `residual`   — F32 `[n_tokens, h]` (can be null if `add_residual = false`).
/// * `output`     — F32 `[n_tokens, h]` output buffer.
/// * `add_residual` — If true, add the residual buffer to the output.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_moe_weighted_reduce(
    encoder:      &mut CommandEncoder,
    registry:     &mut KernelRegistry,
    device:       &MlxDevice,
    expert_w:     &MlxBuffer,
    y_expert:     &MlxBuffer,
    sh_gate:      &MlxBuffer,
    y_shared:     &MlxBuffer,
    residual:     &MlxBuffer,  // dummy/unused buffer when add_residual=false
    output:       &mut MlxBuffer,
    n_tokens:     u32,
    top_k:        u32,
    h:            u32,
    add_residual: bool,
) -> Result<()> {
    let f32_sz = DType::F32.size_of();

    if n_tokens == 0 || top_k == 0 || h == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_weighted_reduce: n_tokens, top_k, h must be > 0".into(),
        ));
    }

    let total_slots = (n_tokens as usize) * (top_k  as usize);
    let expect_ew   = total_slots * f32_sz;
    let expect_ye   = total_slots * (h as usize) * f32_sz;
    let expect_sg   = (n_tokens as usize) * f32_sz;
    let expect_ys   = (n_tokens as usize) * (h as usize) * f32_sz;
    let expect_out  = (n_tokens as usize) * (h as usize) * f32_sz;

    macro_rules! check_buf {
        ($buf:expr, $expected:expr, $name:literal) => {
            if $buf.byte_len() < $expected {
                return Err(MlxError::InvalidArgument(format!(
                    "moe_weighted_reduce: {} too small (expected {}, got {})",
                    $name, $expected, $buf.byte_len()
                )));
            }
        };
    }

    check_buf!(expert_w, expect_ew,  "expert_w");
    check_buf!(y_expert, expect_ye,  "y_expert");
    check_buf!(sh_gate,  expect_sg,  "sh_gate");
    check_buf!(y_shared, expect_ys,  "y_shared");
    check_buf!(output,   expect_out, "output");

    let gpu_params = MoeWeightedReduceGpuParams {
        n_tokens,
        top_k,
        h,
        add_residual: add_residual as u32,
    };

    let pipeline = registry.get_pipeline("moe_weighted_reduce_f32", device.metal_device())?;

    // Grid: (ceil(h/256), n_tokens, 1).
    let threadgroups = MTLSize::new(
        (h as u64 + TG_SIZE - 1) / TG_SIZE,
        n_tokens as u64,
        1,
    );
    let threadgroup_size = MTLSize::new(TG_SIZE, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(expert_w)),
            (2, KernelArg::Buffer(y_expert)),
            (3, KernelArg::Buffer(sh_gate)),
            (4, KernelArg::Buffer(y_shared)),
            (5, KernelArg::Buffer(residual)),
            (6, KernelArg::Buffer(output)),
        ],
        &[],  // no threadgroup memory
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}
