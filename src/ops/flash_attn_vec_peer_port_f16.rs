//! Flash attention vector kernel — verbatim peer port (f16-K + f16-V, DK=DV=256).
//!
//! Ports llama.cpp's `kernel_flash_attn_ext_vec` body verbatim for the
//! NWG=1, NSG=1, NE=1, DK=DV=256, f16-K/f16-V instantiation targeting
//! gemma4 sliding decode.
//!
//! Hypothesis: Apple Metal compiler PSO quality is sensitive to peer's exact source
//! pattern; verbatim port should produce peer-equivalent performance.
//!
//! Buffer slots: 0=params, 1=Q(float*), 2=K_f16(half*), 3=V_f16(half*), 4=dst(float*).
//! No function constants — flags are physically baked into the shader source.
//! No reduce kernel — NWG=1 writes dst directly.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the peer-port SDPA kernel.
pub static FLASH_ATTN_VEC_PEER_PORT_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_peer_port_f16.metal");

/// Register peer-port SDPA shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "flash_attn_vec_peer_port_f16_dk256_dv256",
        FLASH_ATTN_VEC_PEER_PORT_SHADER_SOURCE,
    );
}

/// Host-side parameters for the peer-port kernel.
/// Fields subset from `FlashAttnVecTqHbParams` — only what the NWG=1/NSG=1 kernel needs.
#[derive(Debug, Clone)]
pub struct FlashAttnVecPeerPortParams {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub scale: f32,
    pub mask_type: u32,
    pub sliding_window: u32,
    pub ring_start: u32,
}

/// GPU-side parameter struct. Must match `FlashAttnVecPeerPortParams` in MSL exactly.
/// 9 fields × 4 bytes = 36 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecPeerPortParamsGpu {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    ring_start: u32,
}

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

/// Dispatch the peer-port f16-K/f16-V flash attention vector kernel.
///
/// Preconditions:
/// - `params.head_dim` must be 256 (only DK=DV=256 instantiated this iteration).
/// - `k_f16.dtype()` must be `DType::F16`.
/// - `v_f16.dtype()` must be `DType::F16`.
/// - Q is F32 `[n_heads, head_dim]` (raw, NOT FWHT-rotated — matches RULE-4).
/// - NWG=1 is hardcoded; caller must NOT pass a tmp buffer or expect a reduce step.
///
/// Returns `Err(InvalidArgument)` if any precondition fails.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_peer_port_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_f16: &MlxBuffer,
    v_f16: &MlxBuffer,
    output: &MlxBuffer,
    params: &FlashAttnVecPeerPortParams,
) -> Result<()> {
    if params.head_dim != 256 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16: head_dim must be 256, got {}",
            params.head_dim
        )));
    }
    if k_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16: k_f16 must be DType::F16, got {:?}",
            k_f16.dtype()
        )));
    }
    if v_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16: v_f16 must be DType::F16, got {:?}",
            v_f16.dtype()
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_peer_port_f16: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_peer_port_f16: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }

    let gpu_params = FlashAttnVecPeerPortParamsGpu {
        num_heads: params.num_heads,
        num_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        mask_type: params.mask_type,
        sliding_window: params.sliding_window,
        ring_start: params.ring_start,
    };

    // FATTN_SMEM(nsg=1) from ggml-metal-ops.cpp:2938 specialized for DK=DV=256, C=32, nsg=1.
    // Formula: GGML_PAD((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128)) * nsg
    //           * sizeof(float)/2, 16)
    // = GGML_PAD((256 + 128 + 512) * 1 * 2, 16) = GGML_PAD(1792, 16) = 1792 bytes.
    //
    // Equivalently from our layout: pk + nsg*(sh + 2*pv) in halfs × 2 bytes.
    // pk=256, sh=128, pv=256, nsg=1 → (256 + 128 + 512) = 896 halfs → 1792 bytes.
    let dk = params.head_dim as usize;
    let dv = params.head_dim as usize;
    let c = 32_usize;
    let pk = pad2(dk, 128);
    let pv = pad2(dv, 128);
    let sh = 4 * c;
    let shmem_halfs = pk + 1 * (sh + 2 * pv);
    let shmem_bytes = shmem_halfs * 2;

    let pipeline = registry.get_pipeline(
        "flash_attn_vec_peer_port_f16_dk256_dv256",
        device.metal_device(),
    )?;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // Threadgroup grid for decode-single-query:
    // threadgroups = (ne01, ne02, ne03*NWG) = (1, num_heads, 1) [NWG=1, NE=1 baked].
    // threadgroup_size = (32, NSG, 1) = (32, 1, 1) [NSG=1 baked].
    // Matches peer ggml-metal-ops.cpp:3019 at nqptg=nhptg=nwg=nsg=1, ne01=ne02=ne03=1.
    let threadgroups = MTLSize::new(1, params.num_heads as u64, 1);
    let threadgroup_size = MTLSize::new(32, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_f16)),
            (3, KernelArg::Buffer(v_f16)),
            (4, KernelArg::Buffer(output)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // NWG=1: no reduce kernel needed — kernel writes dst directly.

    Ok(())
}

// ============================================================================
// NWG=32 variant (peer's actual runtime config).
// Pairs with the reduce kernel.
// ============================================================================

/// Reduce-kernel host params. Matches `FlashAttnVecPeerPortReduceParams` in MSL.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnVecPeerPortReduceParams {
    pub nrows: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecPeerPortReduceParamsGpu {
    nrows: i32,
}

/// Size in bytes of the tmp buffer required by the NWG=32 vec dispatcher.
/// Layout (peer ggml-metal-ops.cpp `ggml_metal_op_flash_attn_ext_extra_tmp`):
///   partials: nrows × DV × NWG × sizeof(f32)
///   S/M    : nrows × 2 × NWG × sizeof(f32)
/// Total = nrows × NWG × (DV + 2) × 4 bytes.
pub fn flash_attn_vec_peer_port_f16_nwg32_tmp_bytes(num_heads: u32, head_dim: u32) -> usize {
    const NWG: u64 = 32;
    let nrows = num_heads as u64;
    let dv = head_dim as u64;
    ((nrows * NWG * (dv + 2)) * 4) as usize
}

/// Dispatch the NWG=32 peer-port vec kernel + reduce kernel.
///
/// Mirrors peer's runtime dispatch (ggml-metal-ops.cpp:3010-3055 at nwg=32):
///   1. Vec kernel writes partials to `tmp`, threadgroups (1, num_heads, NWG=32).
///   2. Encoder barrier (peer's `ggml_metal_op_concurrency_reset`).
///   3. Reduce kernel reads `tmp`, writes final `output`, threadgroups (num_heads, 1, 1).
///
/// Caller-allocated `tmp` must be at least `flash_attn_vec_peer_port_f16_nwg32_tmp_bytes()`.
/// `output` is the final SDPA dst (F32 `[num_heads, head_dim]`).
///
/// Preconditions match `flash_attn_vec_peer_port_f16` (head_dim==256, F16-K, F16-V).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_peer_port_f16_nwg32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_f16: &MlxBuffer,
    v_f16: &MlxBuffer,
    tmp: &MlxBuffer,
    output: &MlxBuffer,
    params: &FlashAttnVecPeerPortParams,
) -> Result<()> {
    // Preconditions identical to the NWG=1 dispatcher.
    if params.head_dim != 256 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16_nwg32: head_dim must be 256, got {}",
            params.head_dim
        )));
    }
    if k_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16_nwg32: k_f16 must be DType::F16, got {:?}",
            k_f16.dtype()
        )));
    }
    if v_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16_nwg32: v_f16 must be DType::F16, got {:?}",
            v_f16.dtype()
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_peer_port_f16_nwg32: num_heads/num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16_nwg32: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_peer_port_f16_nwg32: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16_nwg32: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }

    let required_tmp = flash_attn_vec_peer_port_f16_nwg32_tmp_bytes(
        params.num_heads,
        params.head_dim,
    );
    if tmp.byte_len() < required_tmp {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_peer_port_f16_nwg32: tmp buffer too small ({} < {} bytes)",
            tmp.byte_len(),
            required_tmp,
        )));
    }

    let gpu_params = FlashAttnVecPeerPortParamsGpu {
        num_heads: params.num_heads,
        num_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        mask_type: params.mask_type,
        sliding_window: params.sliding_window,
        ring_start: params.ring_start,
    };

    // FATTN_SMEM identical to NWG=1 path (nsg=1).
    let dk = params.head_dim as usize;
    let dv = params.head_dim as usize;
    let c = 32_usize;
    let pk = pad2(dk, 128);
    let pv = pad2(dv, 128);
    let sh = 4 * c;
    let shmem_halfs = pk + 1 * (sh + 2 * pv);
    let shmem_bytes = shmem_halfs * 2;

    // -- Step 1: dispatch the NWG=32 vec kernel, writing partials to `tmp`. --
    let vec_pipeline = registry.get_pipeline(
        "flash_attn_vec_peer_port_f16_nwg32_dk256_dv256",
        device.metal_device(),
    )?;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // Peer ggml-metal-ops.cpp:3019/3032 at nwg=32: dispatch (ne01, ne02, ne03*nwg) = (1, num_heads, 32).
    let vec_threadgroups = MTLSize::new(1, params.num_heads as u64, 32);
    let vec_threadgroup_size = MTLSize::new(32, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        vec_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_f16)),
            (3, KernelArg::Buffer(v_f16)),
            (4, KernelArg::Buffer(tmp)),
        ],
        &[(0, shmem_bytes as u64)],
        vec_threadgroups,
        vec_threadgroup_size,
    );

    // -- Step 2: encoder barrier (peer's `ggml_metal_op_concurrency_reset`). --
    encoder.memory_barrier();

    // -- Step 3: dispatch the reduce kernel, writing final dst. --
    let reduce_pipeline = registry.get_pipeline(
        "flash_attn_vec_peer_port_f16_reduce_dv256_nwg32",
        device.metal_device(),
    )?;

    let reduce_params = FlashAttnVecPeerPortReduceParamsGpu {
        nrows: params.num_heads as i32,
    };

    // Peer ggml-metal-ops.cpp:3052: dispatch (nrows, 1, 1) × (32*nwg, 1, 1) = (num_heads, 1, 1) × (1024, 1, 1).
    let reduce_threadgroups = MTLSize::new(params.num_heads as u64, 1, 1);
    let reduce_threadgroup_size = MTLSize::new(32 * 32, 1, 1);

    encoder.encode_threadgroups_with_args(
        reduce_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&reduce_params))),
            (1, KernelArg::Buffer(tmp)),
            (2, KernelArg::Buffer(output)),
        ],
        reduce_threadgroups,
        reduce_threadgroup_size,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        // 9 fields × 4 bytes = 36 bytes; must match MSL struct layout.
        assert_eq!(std::mem::size_of::<FlashAttnVecPeerPortParamsGpu>(), 36);
    }

    #[test]
    fn test_shmem_formula() {
        // FATTN_SMEM(1) at DK=DV=256, C=32 must be 1792 bytes.
        let dk = 256_usize;
        let dv = 256_usize;
        let c = 32_usize;
        let pk = pad2(dk, 128);
        let pv = pad2(dv, 128);
        let sh = 4 * c;
        let shmem_halfs = pk + 1 * (sh + 2 * pv);
        let shmem_bytes = shmem_halfs * 2;
        assert_eq!(shmem_bytes, 1792);
    }

    #[test]
    fn pipeline_registers_and_compiles() {
        let device = match crate::device::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return, // No Metal device in CI; skip silently.
        };
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        // Verify the Metal compiler accepts the MSL source.
        registry
            .get_pipeline("flash_attn_vec_peer_port_f16_dk256_dv256", device.metal_device())
            .expect("Metal compiler rejected flash_attn_vec_peer_port_f16_dk256_dv256 — check MSL source");
    }

    #[test]
    fn reduce_pipeline_registers_and_compiles() {
        // ADR-029: peer reduce kernel (paired with the NWG=32 vec kernel).
        let device = match crate::device::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        registry
            .get_pipeline(
                "flash_attn_vec_peer_port_f16_reduce_dv256_nwg32",
                device.metal_device(),
            )
            .expect(
                "Metal compiler rejected flash_attn_vec_peer_port_f16_reduce_dv256_nwg32 \
                 — check MSL source",
            );
    }

    #[test]
    fn nwg32_pipeline_registers_and_compiles() {
        // ADR-029: NWG=32 vec kernel (peer's actual runtime config).
        let device = match crate::device::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        registry
            .get_pipeline(
                "flash_attn_vec_peer_port_f16_nwg32_dk256_dv256",
                device.metal_device(),
            )
            .expect(
                "Metal compiler rejected flash_attn_vec_peer_port_f16_nwg32_dk256_dv256 \
                 — check MSL source",
            );
    }
}
