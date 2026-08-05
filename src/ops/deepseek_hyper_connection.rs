//! In-process Metal implementation of DeepSeek-V4 Hyper-Connections.
//!
//! The layouts follow the official inference source directly. In particular,
//! `comb[token, source, destination]` is not transposed at the API boundary.
//! Dynamic non-finite inputs fail closed in the shader by producing finite
//! zero outputs; dtype and shape errors are rejected before encoding.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Production DeepSeek-V4 Hyper-Connection stream count.
pub const DEEPSEEK_HC_MULT: usize = 4;
/// Production Sinkhorn iteration count.
pub const DEEPSEEK_HC_SINKHORN_ITERS: usize = 20;
/// Production Sinkhorn epsilon.
pub const DEEPSEEK_HC_EPS: f32 = 1.0e-6;

const MIX_WIDTH: usize = (2 + DEEPSEEK_HC_MULT) * DEEPSEEK_HC_MULT;
const SPLIT_KERNEL: &str = "deepseek_hc_split_sinkhorn_f32";
const HEAD_KERNEL: &str = "deepseek_hc_head_weights_f32";
const PRE_KERNEL: &str = "deepseek_hc_pre_f32";
const POST_KERNEL: &str = "deepseek_hc_post_f32";
const SIMD_WIDTH: u64 = 32;
const MAX_SIMDGROUPS: u64 = 4;

/// Embedded source for all Hyper-Connection kernels.
pub static DEEPSEEK_HC_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_hyper_connection.metal");

/// Register the Hyper-Connection entry points.
pub fn register(registry: &mut KernelRegistry) {
    for name in [SPLIT_KERNEL, HEAD_KERNEL, PRE_KERNEL, POST_KERNEL] {
        registry.register_source(name, DEEPSEEK_HC_SHADER_SOURCE);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcSplitParams {
    n_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcVectorParams {
    n_tokens: u32,
    n_embd: u32,
}

fn validate_tokens(n_tokens: u32) -> Result<usize> {
    if n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_hc: n_tokens must be greater than zero".into(),
        ));
    }
    Ok(n_tokens as usize)
}

fn validate_vector_dims(n_tokens: u32, n_embd: u32) -> Result<(usize, usize)> {
    let tokens = validate_tokens(n_tokens)?;
    if n_embd == 0 {
        Err(MlxError::InvalidArgument(
            "deepseek_hc: n_embd must be greater than zero".into(),
        ))
    } else {
        Ok((tokens, n_embd as usize))
    }
}

fn checked_shape(dims: &[usize]) -> Result<()> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!("deepseek_hc: shape product overflows: {dims:?}"))
        })
    })?;
    Ok(())
}

fn validate_buffer(buffer: &MlxBuffer, name: &str, shape: &[usize]) -> Result<()> {
    checked_shape(shape)?;
    if buffer.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_hc: {name} must be F32, got {:?}",
            buffer.dtype()
        )));
    }
    if buffer.shape() != shape {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_hc: {name} shape must be {shape:?}, got {:?}",
            buffer.shape()
        )));
    }
    Ok(())
}

/// Split projection output into pre/post weights and a Sinkhorn-normalized
/// combination matrix.
///
/// Layouts are `mixes [tokens, 24]`, `scale [3]`, `base [24]`,
/// `pre/post [tokens, 4]`, and `comb [tokens, source, destination]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hc_split_sinkhorn(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    mixes: &MlxBuffer,
    scale: &MlxBuffer,
    base: &MlxBuffer,
    pre: &MlxBuffer,
    post: &MlxBuffer,
    comb: &MlxBuffer,
    n_tokens: u32,
) -> Result<()> {
    let tokens = validate_tokens(n_tokens)?;
    validate_buffer(mixes, "mixes", &[tokens, MIX_WIDTH])?;
    validate_buffer(scale, "scale", &[3])?;
    validate_buffer(base, "base", &[MIX_WIDTH])?;
    validate_buffer(pre, "pre", &[tokens, DEEPSEEK_HC_MULT])?;
    validate_buffer(post, "post", &[tokens, DEEPSEEK_HC_MULT])?;
    validate_buffer(comb, "comb", &[tokens, DEEPSEEK_HC_MULT, DEEPSEEK_HC_MULT])?;

    let pipeline = registry.get_pipeline(SPLIT_KERNEL, device.metal_device())?;
    let simdgroups = MAX_SIMDGROUPS.min(n_tokens as u64);
    let grid = MTLSize::new((n_tokens as u64).div_ceil(simdgroups), 1, 1);
    let threads = MTLSize::new(SIMD_WIDTH, simdgroups, 1);
    let params = HcSplitParams { n_tokens };
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(mixes)),
            (2, KernelArg::Buffer(scale)),
            (3, KernelArg::Buffer(base)),
            (4, KernelArg::Buffer(pre)),
            (5, KernelArg::Buffer(post)),
            (6, KernelArg::Buffer(comb)),
        ],
        grid,
        threads,
    );
    Ok(())
}

/// Produce the final Hyper-Connection collapse weights.
///
/// Layouts are `mixes [tokens, 4]`, `scale [1]`, `base [4]`, and
/// `weights [tokens, 4]`. The exact transform is
/// `sigmoid(mixes * scale + base) + DEEPSEEK_HC_EPS`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hc_head_weights(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    mixes: &MlxBuffer,
    scale: &MlxBuffer,
    base: &MlxBuffer,
    weights: &MlxBuffer,
    n_tokens: u32,
) -> Result<()> {
    let tokens = validate_tokens(n_tokens)?;
    validate_buffer(mixes, "head mixes", &[tokens, DEEPSEEK_HC_MULT])?;
    validate_buffer(scale, "head scale", &[1])?;
    validate_buffer(base, "head base", &[DEEPSEEK_HC_MULT])?;
    validate_buffer(weights, "head weights", &[tokens, DEEPSEEK_HC_MULT])?;

    let pipeline = registry.get_pipeline(HEAD_KERNEL, device.metal_device())?;
    let grid = MTLSize::new(n_tokens as u64, 1, 1);
    let threads = MTLSize::new(SIMD_WIDTH, 1, 1);
    let params = HcSplitParams { n_tokens };
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(mixes)),
            (2, KernelArg::Buffer(scale)),
            (3, KernelArg::Buffer(base)),
            (4, KernelArg::Buffer(weights)),
        ],
        grid,
        threads,
    );
    Ok(())
}

/// Reduce four Hyper-Connection streams into one transformer input.
///
/// `x [tokens, source, embedding] * weights [tokens, source]`
/// reduces to `output [tokens, embedding]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hc_pre(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    x: &MlxBuffer,
    weights: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: u32,
    n_embd: u32,
) -> Result<()> {
    let (tokens, embd) = validate_vector_dims(n_tokens, n_embd)?;
    validate_buffer(x, "x", &[tokens, DEEPSEEK_HC_MULT, embd])?;
    validate_buffer(weights, "weights", &[tokens, DEEPSEEK_HC_MULT])?;
    validate_buffer(output, "output", &[tokens, embd])?;
    dispatch_vector(
        encoder, registry, device, PRE_KERNEL, x, weights, None, None, output, n_tokens, n_embd,
    )
}

/// Expand a transformer output back into four Hyper-Connection streams.
///
/// For every destination: `output = post * x + sum_source(comb * residual)`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hc_post(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    x: &MlxBuffer,
    residual: &MlxBuffer,
    post: &MlxBuffer,
    comb: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: u32,
    n_embd: u32,
) -> Result<()> {
    let (tokens, embd) = validate_vector_dims(n_tokens, n_embd)?;
    validate_buffer(x, "x", &[tokens, embd])?;
    validate_buffer(residual, "residual", &[tokens, DEEPSEEK_HC_MULT, embd])?;
    validate_buffer(post, "post", &[tokens, DEEPSEEK_HC_MULT])?;
    validate_buffer(comb, "comb", &[tokens, DEEPSEEK_HC_MULT, DEEPSEEK_HC_MULT])?;
    validate_buffer(output, "output", &[tokens, DEEPSEEK_HC_MULT, embd])?;
    dispatch_vector(
        encoder,
        registry,
        device,
        POST_KERNEL,
        x,
        residual,
        Some(post),
        Some(comb),
        output,
        n_tokens,
        n_embd,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_vector(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    kernel: &str,
    first: &MlxBuffer,
    second: &MlxBuffer,
    third: Option<&MlxBuffer>,
    fourth: Option<&MlxBuffer>,
    output: &MlxBuffer,
    n_tokens: u32,
    n_embd: u32,
) -> Result<()> {
    let pipeline = registry.get_pipeline(kernel, device.metal_device())?;
    let tiles = (n_embd as u64).div_ceil(SIMD_WIDTH);
    let simdgroups = MAX_SIMDGROUPS.min(tiles);
    let grid = MTLSize::new(tiles.div_ceil(simdgroups), n_tokens as u64, 1);
    let threads = MTLSize::new(SIMD_WIDTH, simdgroups, 1);
    let params = HcVectorParams { n_tokens, n_embd };
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(first)),
        (2, KernelArg::Buffer(second)),
    ];
    match (third, fourth) {
        (Some(third), Some(fourth)) => {
            args.push((3, KernelArg::Buffer(third)));
            args.push((4, KernelArg::Buffer(fourth)));
            args.push((5, KernelArg::Buffer(output)));
        }
        (None, None) => args.push((3, KernelArg::Buffer(output))),
        _ => unreachable!("post coefficients are supplied together"),
    }
    encoder.encode_threadgroups_with_args(pipeline, &args, grid, threads);
    Ok(())
}
