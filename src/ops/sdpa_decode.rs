//! GPU SDPA decode kernel — F32 Q/K/V, SIMD-vectorized, single-token decode.
//!
//! Replaces the naive `sdpa` kernel for seq_len=1 decode with a SIMD-parallel
//! implementation that avoids the F16 Q precision loss of `flash_attn_vec`.
//!
//! Each threadgroup (32 threads, one SIMD group) handles one query head.
//! Threads cooperate on QK dot products via `simd_sum`, eliminating the
//! scalar inner loop from the naive kernel.
//!
//! Constraints:
//! - seq_len must be 1
//! - head_dim must be a multiple of 32 (128, 256, 512)
//! - Q/K/V must be F32

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Metal shader source.
pub static SDPA_DECODE_SHADER_SOURCE: &str =
    include_str!("../shaders/sdpa_decode.metal");

/// Register `sdpa_decode` pipeline.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("sdpa_decode", SDPA_DECODE_SHADER_SOURCE);
}

/// GPU-side params struct (must match `SdpaDecodeParams` in MSL exactly).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdpaDecodeParamsGpu {
    n_heads:     u32,
    n_kv_heads:  u32,
    head_dim:    u32,
    kv_seq_len:  u32,
    kv_capacity: u32,
    scale:       f32,
}

/// Dispatch the vectorized decode SDPA kernel.
///
/// # Constraints
/// - `head_dim` must be a multiple of 32 (128, 256, 512 are supported).
/// - Q layout: `[n_heads, head_dim]` F32 (seq_len=1, seq dim elided).
/// - K/V layout: `[n_kv_heads, kv_capacity, head_dim]` F32.
/// - Output layout: `[n_heads, head_dim]` F32.
/// - kv_seq_len: number of valid positions in the KV cache (must be > 0).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sdpa_decode(
    encoder:     &mut CommandEncoder,
    registry:    &mut KernelRegistry,
    device:      &MlxDevice,
    q:           &MlxBuffer,
    k:           &MlxBuffer,
    v:           &MlxBuffer,
    output:      &MlxBuffer,
    n_heads:     u32,
    n_kv_heads:  u32,
    head_dim:    u32,
    kv_seq_len:  u32,
    kv_capacity: u32,
    scale:       f32,
) -> Result<()> {
    // Validate head_dim.
    if head_dim % 32 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "sdpa_decode: head_dim ({}) must be a multiple of 32", head_dim
        )));
    }
    if kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "sdpa_decode: kv_seq_len must be > 0".into(),
        ));
    }

    let q_elems  = (n_heads    * head_dim) as usize;
    let kv_elems = (n_kv_heads * kv_capacity * head_dim) as usize;
    let o_elems  = q_elems;

    macro_rules! chk {
        ($buf:expr, $exp:expr, $name:literal) => {
            if $buf.element_count() < $exp {
                return Err(MlxError::InvalidArgument(format!(
                    "sdpa_decode: {} too small ({} < {})", $name,
                    $buf.element_count(), $exp
                )));
            }
        };
    }
    chk!(q,      q_elems,  "Q");
    chk!(k,      kv_elems, "K");
    chk!(v,      kv_elems, "V");
    chk!(output, o_elems,  "output");

    let gpu_params = SdpaDecodeParamsGpu {
        n_heads,
        n_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
    };

    let pipeline = registry.get_pipeline("sdpa_decode", device.metal_device())?;

    // Grid: one TG per query head, 32 threads per TG (1 SIMD group).
    let threadgroups   = MTLSize::new(n_heads as u64, 1, 1);
    let threadgroup_sz = MTLSize::new(32, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(q)),
            (1, KernelArg::Buffer(k)),
            (2, KernelArg::Buffer(v)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[],
        threadgroups,
        threadgroup_sz,
    );

    Ok(())
}
