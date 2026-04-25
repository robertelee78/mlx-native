//! GPU SDPA decode kernel — F32 Q/K/V, multi-simdgroup tiled, single-token decode.
//!
//! The kernel divides the KV sequence across N_SG simdgroups that each scan an
//! independent KV chunk and produce a local (max, sum, unnorm_acc) triple.
//! Simdgroup 0 then merges all N_SG partial results using the log-sum-exp
//! combination rule and writes the final F32 output.
//!
//! # Constraints
//! - seq_len must be 1 (decode path only)
//! - head_dim must be a multiple of 32 (128, 256, 512 are supported)
//! - Q/K/V must be F32
//! - n_sg must be 1, 2, or 4

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
    n_sg:        u32,
}

/// Select the number of simdgroups based on kv_seq_len.
///
/// The threadgroup overhead of writing/reading shared memory and executing
/// the barrier dominates when kv_seq_len is small, so we start with n_sg=1
/// and ramp up as the KV cache grows:
///
/// - kv_seq_len < 32   → 1  (overhead dominates; single sg is faster)
/// - kv_seq_len < 128  → 2  (2× speedup, barrier cost amortized)
/// - otherwise         → 4  (4× speedup at long context)
fn select_n_sg(kv_seq_len: u32) -> u32 {
    if kv_seq_len < 32 {
        1
    } else if kv_seq_len < 128 {
        2
    } else {
        4
    }
}

/// Dispatch the tiled decode SDPA kernel.
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

    let n_sg = select_n_sg(kv_seq_len);

    let gpu_params = SdpaDecodeParamsGpu {
        n_heads,
        n_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        n_sg,
    };

    // Shared memory layout:
    //   sg_max : [n_sg]          floats
    //   sg_sum : [n_sg]          floats
    //   sg_acc : [n_sg*head_dim] floats
    // Total: 4 * n_sg * (head_dim + 2) bytes
    let shmem_bytes: u64 = 4 * n_sg as u64 * (head_dim as u64 + 2);

    let pipeline = registry.get_pipeline("sdpa_decode", device.metal_device())?;

    // Grid: one TG per query head; TG has n_sg * 32 threads.
    let threadgroups   = MTLSize::new(n_heads as u64, 1, 1);
    let threadgroup_sz = MTLSize::new(n_sg as u64 * 32, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(q)),
            (1, KernelArg::Buffer(k)),
            (2, KernelArg::Buffer(v)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shmem_bytes)],
        threadgroups,
        threadgroup_sz,
    );

    Ok(())
}
