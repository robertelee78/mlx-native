//! Hadamard + symmetric affine INT4 KV cache encoder.
//!
//! Same rotation as TQ (FWHT via SIMD shuffle), but replaces Lloyd-Max codebook
//! quantization with symmetric uniform (affine) quantization.
//!
//! Encode: FWHT → absmax → uniform quantize to [0,15] → pack nibbles
//! Stored norm = absmax * sqrt(head_dim) / 7.5
//!
//! Decode (in flash_attn_vec_affine4 kernel):
//!   value = norm * rsqrt(head_dim) * (float(nibble) - 7.5)

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_threadgroups_with_args, KernelArg};

/// MSL source for the affine4 quantize kernel.
pub static SHADER_SOURCE: &str =
    include_str!("../shaders/hadamard_quantize_kv_affine4.metal");

/// Register the affine4 quantize shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("hadamard_quantize_kv_affine4_d256", SHADER_SOURCE);
    registry.register_source("hadamard_quantize_kv_affine4_d512", SHADER_SOURCE);
}

/// Parameters matching `HadamardQuantizeParams` in the Metal shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadamardQuantizeParams {
    head_dim: u32,
    num_kv_heads: u32,
    write_pos: u32,
    cache_capacity: u32,
    is_sliding: u32,
}

/// Dispatch the Hadamard + affine4 quantize kernel.
///
/// Same interface as `hadamard_quantize_kv::dispatch_hadamard_quantize_kv`.
/// Buffer layouts are identical — only the quantization math differs.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hadamard_quantize_kv_affine4(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_affine4_d256",
        512 => "hadamard_quantize_kv_affine4_d512",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "hadamard_quantize_kv_affine4: unsupported head_dim {head_dim} (only 256, 512)"
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let params = HadamardQuantizeParams {
        head_dim,
        num_kv_heads,
        write_pos,
        cache_capacity,
        is_sliding: u32::from(is_sliding),
    };

    // 1 simdgroup (32 threads) per head, no shared memory.
    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(packed)),
            (2, KernelArg::Buffer(norms)),
            (3, KernelArg::Bytes(bytemuck::bytes_of(&params))),
        ],
        MTLSize::new(num_kv_heads as u64, 1, 1),
        MTLSize::new(32, 1, 1),
    );

    Ok(())
}
