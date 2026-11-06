//! Flash attention vector kernel dispatch for hybrid F16-K + TQ-HB-V KV cache.
//!
//! ADR-028 Phase 10d (iter-349): closes the structural 1.81× per-dispatch
//! K-side gap measured iter-326..342 by reading K as F16 dense (peer-equivalent
//! layout, no codebook lookup) while V stays byte-packed TQ-HB.
//!
//! Memory cost: 158 MB at gemma4 32K context (vs 128 MB pure TQ-HB) — 3.19×
//! savings vs raw F32, preserving 81% of the TQ-HB advantage. See
//! `ADR-028 §iter-346` for the full memory math.
//!
//! ABI: re-uses `FlashAttnVecTqHbParams` (V-side fields are identical;
//! K-side `K_norms` simply absent from the buffer list, K codebook unused).
//! V codebook width controlled by the same `CBITS_FC` function constant
//! (5/6/8) at index 50 — identical to flash_attn_vec_tq_hb.
//!
//! Buffer slots (compacted vs TQ-HB which has 7 buffers):
//!   0 = params, 1 = Q, 2 = K_f16, 3 = V_packed, 4 = V_norms, 5 = dst.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

// Re-export the shared params struct + tmp helpers from the TQ-HB module —
// the hybrid kernel uses byte-identical layout (only K-side buffer differs).
pub use super::flash_attn_vec_tq_hb::{
    compute_nsg, tmp_buffer_bytes, FlashAttnVecTqHbParams,
};

/// MSL source for the hybrid SDPA kernel.
pub static FLASH_ATTN_VEC_HYBRID_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_hybrid.metal");

/// Register hybrid SDPA shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_hybrid_dk256", FLASH_ATTN_VEC_HYBRID_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_hybrid_dk512", FLASH_ATTN_VEC_HYBRID_SHADER_SOURCE);
}

/// GPU-side parameter struct (byte-identical layout to `FlashAttnVecTqHbParamsGpu`
/// in the TQ-HB module — re-defined locally to avoid cross-module pub-vis).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecHybridParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    softcap: f32,
    nwg: u32,
    ring_start: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
    fuse_fwht_pre: u32,
    nsg: u32,
    /// ADR-040 M4: batched-flash query count (1 for the non-batched kernel).
    n_queries: u32,
}

/// GPU-side reduce params (re-uses flash_attn_vec_reduce_dk{256,512} kernel).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

fn validate_params(params: &FlashAttnVecTqHbParams) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_hybrid: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_hybrid: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    // V codebook bits — same allowed set as TQ-HB (5/6/8). Hybrid path's K is F16
    // and ignores this; only V-side honors it.
    if !matches!(params.codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: V codebook_bits must be 5, 6, or 8, got {}",
            params.codebook_bits
        )));
    }
    if params.nsg == 0 || (params.nsg & (params.nsg - 1)) != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: nsg must be a power of 2 (1, 2, 4, ...), got {}",
            params.nsg
        )));
    }
    if params.nsg > 4 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: nsg must be ≤ 4 (kernel reduce cap), got {}",
            params.nsg
        )));
    }
    Ok(())
}

/// Mirror of `flash_attn_vec_tq_hb::compute_nwg`. Hybrid uses the same kL
/// adaptive policy because the workgroup-axis split is independent of K dtype.
///
/// ADR-029 iter-175 Step 1v (2026-05-15) NWG sweep at HEAD (gemma4-APEX-Q5_K_M, M5 Max, tg100):
///   default (NWG=16 at kv≤512):  95.5 / 92.9 (range 2.6 — thermal-affected)
///   NWG=32 forced (`HF2Q_HYBRID_NWG=32`): 95.6 / 95.6 (tight, range 0.0)
///   NWG=8  forced (`HF2Q_HYBRID_NWG=8`):  95.4
/// At tg100 the FA contribution is only ~0.34% of decode wall (kv mean=50,
/// ~1.2 µs/call × 30 calls/layer × 1 layer = small).  NWG=32 forced is
/// marginally tighter (less variance) but mean delta within bench noise.
/// Threshold kept at 512 because no measurable wall-level gain crossing it.
fn compute_nwg(kv_seq_len: u32) -> u32 {
    // ADR-029 iter-175 Step 1as: cache parsed override.
    // Sliding-attn (this kernel) fires ~26 times/token at gemma4 decode;
    // uncached env::var is ~70 ns/call (H-N bench).  Cached read is ~2 ns.
    use std::sync::atomic::{AtomicI32, Ordering};
    static CACHED_HYBRID_NWG: AtomicI32 = AtomicI32::new(-1);
    // -1 = uninitialized; 0 = no override (use kv_seq_len heuristic);
    //  1..=32 = forced override.
    let mut v = CACHED_HYBRID_NWG.load(Ordering::Relaxed);
    if v < 0 {
        let parsed = std::env::var("HF2Q_HYBRID_NWG")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|n| (1..=32).contains(n))
            .unwrap_or(0);
        CACHED_HYBRID_NWG.store(parsed as i32, Ordering::Relaxed);
        v = parsed as i32;
    }
    if v > 0 {
        return v as u32;
    }
    if kv_seq_len > 512 { 32 } else { 16 }
}

/// Dispatch the hybrid F16-K + TQ-HB-V flash attention vector kernel.
///
/// Caller responsibilities (same as `flash_attn_vec_tq_hb`):
///   * Q is F32 `[n_heads, head_dim]` (or `[n_heads, 1, head_dim]` for decode).
///   * Q must be FWHT-pre-rotated UNLESS `params.fuse_fwht_pre == 1` (kernel
///     applies the rotation internally).
///   * K_f16 is `[num_kv_heads, kv_capacity, head_dim]` half (F16, 2 bytes/elem).
///   * V_packed is `[num_kv_heads, kv_capacity, head_dim]` u8 (byte-packed).
///   * V_norms layout matches `flash_attn_vec_tq_hb` (D=256: 1/pos, D=512: 2/pos).
///   * Caller applies inverse-FWHT to `output` if Q was pre-rotated (mirrors
///     existing TQ-HB caller pattern at hf2q `forward_mlx.rs:3744+`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_hybrid(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_f16: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &FlashAttnVecTqHbParams,
) -> Result<()> {
    validate_params(params)?;

    // K_f16 dtype check — hybrid path requires F16; F32 K would silently
    // misalign the kernel's `device const half *K_f16` reads (2 bytes/elem
    // vs 4) → 2× overrun on every K row → memory corruption. Catch loud.
    if k_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: k_f16 must be DType::F16, got {:?}",
            k_f16.dtype()
        )));
    }

    let head_dim = params.head_dim;
    let nwg = compute_nwg(params.kv_seq_len);

    let gpu_params = FlashAttnVecHybridParamsGpu {
        n_heads: params.num_heads,
        n_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        mask_type: params.mask_type,
        sliding_window: params.sliding_window,
        softcap: params.softcap,
        nwg,
        ring_start: params.ring_start,
        scale_factor_d512: params.scale_factor_d512,
        codebook_bits: params.codebook_bits,
        fuse_fwht_pre: params.fuse_fwht_pre,
        nsg: params.nsg,
        n_queries: 1,
    };

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_hybrid_dk256",
        512 => "flash_attn_vec_hybrid_dk512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: unsupported head_dim {head_dim}"
        ))),
    };
    // V codebook function constant (same as TQ-HB at index 50). Hybrid path's
    // K-side ignores this; only the V dequant inner loop honors it.
    let cbits_const = (params.codebook_bits as i32, 50usize);
    // ADR-029 iter-20 H27: V-dtype function constant (slot 51).  When V buffer
    // is F16-typed (caller has allocated full F16 KV cache via HF2Q_FULL_F16_KV),
    // the kernel takes the F16-V direct-read branch.  When U8-typed (legacy
    // TQ-HB byte-packed V), the kernel takes the dequant_hb_float4 branch.
    let v_is_f16: i32 = match v_packed.dtype() {
        crate::DType::F16 => 1,
        _ => 0,
    };
    let pipeline = registry
        .get_pipeline_with_constants(
            kernel_name,
            device.metal_device(),
            &[],
            &[(cbits_const.1, cbits_const.0), (51usize, v_is_f16)],
        )?;

    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    // Same NSG-aware shmem layout as TQ-HB.
    let nsg = params.nsg as usize;
    let shmem_halfs = pk + nsg * (sh + 2 * pv);
    let shmem_bytes = shmem_halfs * 2;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    let threadgroups = MTLSize::new(1, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, params.nsg as u64, 1);

    let dst_buf = if nwg == 1 { output } else { tmp };

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_f16)),     // F16 dense (vs k_packed + k_norms in TQ-HB)
            (3, KernelArg::Buffer(v_packed)),
            (4, KernelArg::Buffer(v_norms)),
            (5, KernelArg::Buffer(dst_buf)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // Reduce kernel (NWG > 1) — same kernel as TQ-HB.
    if nwg > 1 {
        encoder.memory_barrier();

        let reduce_params = FlashAttnVecReduceParamsGpu { nrows: params.num_heads };

        let reduce_kernel = match head_dim {
            256 => "flash_attn_vec_reduce_dk256",
            512 => "flash_attn_vec_reduce_dk512",
            _ => unreachable!(),
        };
        let reduce_pipeline = registry.get_pipeline(reduce_kernel, device.metal_device())?;

        let reduce_tg = MTLSize::new(params.num_heads as u64, 1, 1);
        let reduce_tg_size = MTLSize::new(32 * nwg as u64, 1, 1);

        encoder.encode_threadgroups_with_args(
            reduce_pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&reduce_params))),
                (1, KernelArg::Buffer(tmp)),
                (2, KernelArg::Buffer(output)),
                (3, KernelArg::Bytes(as_bytes(&nwg))),
            ],
            reduce_tg,
            reduce_tg_size,
        );
    }

    Ok(())
}

/// ADR-040 M4 — BATCHED multi-sequence decode flash. One dispatch over `n_q`
/// queries (grid.x = n_q), each attending to its OWN physical-slot KV region in
/// the shared multi-seq buffers. Per-query input addressing (Q/K/V/V_norms base
/// offsets + kv_seq_len + ring_start passed as `[n_q]` u32 device arrays);
/// inner attention math identical to `flash_attn_vec_hybrid` ⇒ per-row
/// bit-identical. The reduce kernel is reused unchanged with `nrows = n_q*heads`.
///
/// Requires all queries to share the SAME (nwg, nsg) bucket — the caller gates
/// on it (decode lengths in the same kv_seq_len band) and falls back to N
/// per-slot dispatches otherwise. `q`/`dst` are `[n_q, heads*head_dim]`;
/// `k_f16`/`v_packed`/`v_norms` are the FULL multi-seq buffers; `tmp` is
/// `[n_q, heads*head_dim*nwg + heads*2*nwg]`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_hybrid_batched(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    n_q: u32,
    q: &MlxBuffer,
    k_f16: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    slot_id_arr: &MlxBuffer,
    seq_pos_arr: &MlxBuffer,
    params: &FlashAttnVecTqHbParams,
) -> Result<()> {
    validate_params(params)?;
    if k_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched: k_f16 must be DType::F16, got {:?}",
            k_f16.dtype()
        )));
    }
    let head_dim = params.head_dim;
    // params.kv_seq_len must be the MAX over queries (used only for nwg bucket).
    let nwg = compute_nwg(params.kv_seq_len);

    let gpu_params = FlashAttnVecHybridParamsGpu {
        n_heads: params.num_heads,
        n_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        mask_type: params.mask_type,
        sliding_window: params.sliding_window,
        softcap: params.softcap,
        nwg,
        ring_start: params.ring_start,
        scale_factor_d512: params.scale_factor_d512,
        codebook_bits: params.codebook_bits,
        fuse_fwht_pre: params.fuse_fwht_pre,
        nsg: params.nsg,
        n_queries: n_q,
    };

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_hybrid_batched_dk256",
        512 => "flash_attn_vec_hybrid_batched_dk512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched: unsupported head_dim {head_dim}"
        ))),
    };
    let cbits_const = (params.codebook_bits as i32, 50usize);
    let v_is_f16: i32 = match v_packed.dtype() {
        crate::DType::F16 => 1,
        _ => 0,
    };
    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(cbits_const.1, cbits_const.0), (51usize, v_is_f16)],
    )?;

    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    let nsg = params.nsg as usize;
    let shmem_halfs = pk + nsg * (sh + 2 * pv);
    let shmem_bytes = shmem_halfs * 2;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // grid.x = n_q (was 1). Each (query, head, wg) threadgroup.
    let threadgroups = MTLSize::new(n_q as u64, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, params.nsg as u64, 1);

    let dst_buf = if nwg == 1 { output } else { tmp };

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_f16)),
            (3, KernelArg::Buffer(v_packed)),
            (4, KernelArg::Buffer(v_norms)),
            (5, KernelArg::Buffer(dst_buf)),
            (6, KernelArg::Buffer(slot_id_arr)),
            (7, KernelArg::Buffer(seq_pos_arr)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    if nwg > 1 {
        encoder.memory_barrier();
        // nrows = n_q * heads so the reduce's S/M base clears ALL queries' partials.
        let reduce_params = FlashAttnVecReduceParamsGpu { nrows: n_q * params.num_heads };
        let reduce_kernel = match head_dim {
            256 => "flash_attn_vec_reduce_dk256",
            512 => "flash_attn_vec_reduce_dk512",
            _ => unreachable!(),
        };
        let reduce_pipeline = registry.get_pipeline(reduce_kernel, device.metal_device())?;
        // grid.x = n_q * heads (one row per (query, head)).
        let reduce_tg = MTLSize::new((n_q * params.num_heads) as u64, 1, 1);
        let reduce_tg_size = MTLSize::new(32 * nwg as u64, 1, 1);
        encoder.encode_threadgroups_with_args(
            reduce_pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&reduce_params))),
                (1, KernelArg::Buffer(tmp)),
                (2, KernelArg::Buffer(output)),
                (3, KernelArg::Bytes(as_bytes(&nwg))),
            ],
            reduce_tg,
            reduce_tg_size,
        );
    }

    Ok(())
}

/// ADR-063 lever A: the (nwg, nsg) bucket one slot runs at, i.e. the split
/// count and SIMD-group count its solo `flash_attn_vec_hybrid_batched`
/// dispatch would use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridSlotBucket {
    /// Split-K workgroups (1..=32; the stock reduce handles at most 32).
    pub nwg: u32,
    /// SIMD groups per workgroup (1, 2 or 4).
    pub nsg: u32,
}

impl HybridSlotBucket {
    /// The stock bucket for a slot whose effective KV length is
    /// `kv_seq_len` (`compute_nwg` and `compute_nsg`, unchanged heuristics).
    pub fn for_kv_seq_len(kv_seq_len: u32) -> Self {
        Self {
            nwg: compute_nwg(kv_seq_len),
            nsg: compute_nsg(kv_seq_len),
        }
    }
}

/// Query-head group sizes the grouped kernel is instantiated for.
pub const FLASH_ATTN_VEC_HYBRID_GQA_GROUPS: [u32; 4] = [1, 2, 4, 8];

/// Threadgroup-memory cap a grouped dispatch must fit (Apple GPUs: 32 KiB).
pub const FLASH_ATTN_VEC_HYBRID_GQA_MAX_SHMEM_BYTES: usize = 32 * 1024;

/// Threadgroup memory (bytes) of one grouped threadgroup: per head the stock
/// `pk + nsg·(sh + 2·pv)` halves, times `group`.
pub fn flash_attn_vec_hybrid_batched_gqa_shmem_bytes(
    head_dim: u32,
    group: u32,
    nsg_max: u32,
) -> usize {
    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    group as usize * (pk + nsg_max as usize * (sh + 2 * pv)) * 2
}

/// Whether `group` query heads per threadgroup is legal: an instantiated
/// size, dividing `num_heads / num_kv_heads`, and fitting 32 KiB of
/// threadgroup memory at `nsg_max`.
///
/// Head dim 512 is legal only at `group == 1`. The stock d512 V scale
/// `vnorm / sf * w` has no source form whose floating-point association is
/// the same on every Metal toolchain once heads are grouped: the pinned
/// `(w * (1/sf)) * vnorm` form drifts from stock on the CI toolchain even at
/// G = 1, and both the per-head literal form and an opaque per-head divisor
/// match stock on CI at every G but drift 1 ULP on the M5 toolchain at
/// G >= 2. Only the ungrouped kernel is bitwise-equal to stock everywhere.
/// Head dim 256 keeps every group in [`FLASH_ATTN_VEC_HYBRID_GQA_GROUPS`].
pub fn flash_attn_vec_hybrid_batched_gqa_group_is_legal(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    group: u32,
    nsg_max: u32,
) -> bool {
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return false;
    }
    if head_dim == 512 && group != 1 {
        return false;
    }
    let heads_per_kv = num_heads / num_kv_heads;
    FLASH_ATTN_VEC_HYBRID_GQA_GROUPS.contains(&group)
        && heads_per_kv % group == 0
        && flash_attn_vec_hybrid_batched_gqa_shmem_bytes(head_dim, group, nsg_max)
            <= FLASH_ATTN_VEC_HYBRID_GQA_MAX_SHMEM_BYTES
}

/// The largest legal group size (see
/// [`flash_attn_vec_hybrid_batched_gqa_group_is_legal`]); 1 when no larger
/// size is legal.
pub fn flash_attn_vec_hybrid_batched_gqa_pick_group(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    nsg_max: u32,
) -> u32 {
    FLASH_ATTN_VEC_HYBRID_GQA_GROUPS
        .iter()
        .rev()
        .copied()
        .find(|&g| {
            flash_attn_vec_hybrid_batched_gqa_group_is_legal(
                num_heads,
                num_kv_heads,
                head_dim,
                g,
                nsg_max,
            )
        })
        .unwrap_or(1)
}

/// GPU-side per-slot bucket (`FlashAttnHybridSlotBucket` in the shader).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HybridSlotBucketGpu {
    nwg: u32,
    nsg: u32,
    part_off: u32,
    sm_off: u32,
}

/// One stock reduce dispatch: a maximal run of consecutive slots sharing an
/// NWG > 1, laid out exactly as a stock dispatch over those rows.
struct HybridReduceRun {
    slot0: usize,
    slots: usize,
    nwg: u32,
    tmp_off_floats: usize,
    tmp_floats: usize,
}

/// Slots per grouped dispatch: the bucket array travels as inline bytes
/// (Metal caps `setBytes` at 4 KiB).
const GQA_MAX_SLOTS: usize = 4096 / std::mem::size_of::<HybridSlotBucketGpu>();

fn plan_gqa_layout(
    num_heads: u32,
    head_dim: u32,
    buckets: &[HybridSlotBucket],
) -> Result<(Vec<HybridSlotBucketGpu>, Vec<HybridReduceRun>, usize)> {
    let rows = num_heads as usize;
    let dv = head_dim as usize;
    let mut gpu = Vec::with_capacity(buckets.len());
    let mut runs = Vec::new();
    let mut off = 0usize;
    let to_u32 = |v: usize| {
        u32::try_from(v).map_err(|_| {
            MlxError::InvalidArgument(format!(
                "flash_attn_vec_hybrid_batched_gqa: tmp offset {v} exceeds u32"
            ))
        })
    };
    let mut s = 0usize;
    while s < buckets.len() {
        let nwg = buckets[s].nwg;
        let mut e = s + 1;
        while e < buckets.len() && buckets[e].nwg == nwg {
            e += 1;
        }
        if nwg == 1 {
            // Written straight to the output, as the stock NWG == 1 dispatch.
            for b in &buckets[s..e] {
                gpu.push(HybridSlotBucketGpu { nwg: 1, nsg: b.nsg, part_off: 0, sm_off: 0 });
            }
        } else {
            // float4 partials: keep every run base 16-byte aligned.
            off = pad2(off, 4);
            let w = nwg as usize;
            let slots = e - s;
            let part_floats = slots * rows * dv * w;
            let sm_floats = slots * rows * 2 * w;
            for (k, b) in buckets[s..e].iter().enumerate() {
                gpu.push(HybridSlotBucketGpu {
                    nwg,
                    nsg: b.nsg,
                    part_off: to_u32(off + k * rows * dv * w)?,
                    sm_off: to_u32(off + part_floats + k * rows * 2 * w)?,
                });
            }
            runs.push(HybridReduceRun {
                slot0: s,
                slots,
                nwg,
                tmp_off_floats: off,
                tmp_floats: part_floats + sm_floats,
            });
            off += part_floats + sm_floats;
            to_u32(off)?;
        }
        s = e;
    }
    Ok((gpu, runs, off))
}

/// Bytes of `tmp` [`flash_attn_vec_hybrid_batched_gqa`] writes for these
/// per-slot buckets. With one shared bucket this equals the stock batched
/// layout (`n_q · heads · nwg · (head_dim + 2)` floats); it never exceeds
/// `n_q · tmp_buffer_bytes(heads, head_dim)` for NWG values that are
/// multiples of 2.
pub fn flash_attn_vec_hybrid_batched_gqa_tmp_bytes(
    num_heads: u32,
    head_dim: u32,
    buckets: &[HybridSlotBucket],
) -> Result<usize> {
    let (_, _, floats) = plan_gqa_layout(num_heads, head_dim, buckets)?;
    Ok(floats * std::mem::size_of::<f32>())
}

/// ADR-063 lever A (A1 + A2) — batched multi-sequence decode flash with
/// GQA-grouped query heads and per-slot (nwg, nsg) buckets. Class A against
/// [`flash_attn_vec_hybrid_batched`] run per slot at that slot's bucket:
/// every output row (and every partial/S/M value) is bitwise that solo
/// dispatch's.
///
/// * `group` query heads sharing one KV head run in one threadgroup; each
///   K row and packed V row + norm is loaded and decoded once for them.
///   Must satisfy [`flash_attn_vec_hybrid_batched_gqa_group_is_legal`] at the
///   largest bucket NSG ([`flash_attn_vec_hybrid_batched_gqa_pick_group`]
///   returns the largest legal one). `group == 1` is the ungrouped A2 path.
/// * `buckets[i]` is slot row i's bucket (normally
///   [`HybridSlotBucket::for_kv_seq_len`] of its own effective KV length);
///   `n_q = buckets.len()`. Slots may straddle buckets: no per-slot fallback
///   is needed.
/// * `q`, `k_f16`, `v_packed`, `v_norms`, `output`, `slot_id_arr`,
///   `seq_pos_arr` and `params` are exactly as for
///   [`flash_attn_vec_hybrid_batched`]. The kernel derives every slot's KV
///   length and ring start from `seq_pos_arr` and `params.mask_type`; it
///   does not read `params.kv_seq_len`, `params.ring_start` or `params.nsg`
///   (they are still validated as for the stock call).
/// * `tmp` must hold [`flash_attn_vec_hybrid_batched_gqa_tmp_bytes`]. Each
///   maximal run of consecutive slots sharing an NWG > 1 occupies the stock
///   batched layout for its rows, and is reduced by one stock
///   `flash_attn_vec_reduce_dk{256,512}` dispatch at `32·nwg` width. With
///   one shared bucket the main dispatch writes the stock batched `tmp`
///   layout byte for byte and issues the stock single reduce.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_hybrid_batched_gqa(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_f16: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    slot_id_arr: &MlxBuffer,
    seq_pos_arr: &MlxBuffer,
    params: &FlashAttnVecTqHbParams,
    group: u32,
    buckets: &[HybridSlotBucket],
) -> Result<()> {
    validate_params(params)?;
    if k_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: k_f16 must be DType::F16, got {:?}",
            k_f16.dtype()
        )));
    }
    if buckets.is_empty() || buckets.len() > GQA_MAX_SLOTS {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: need 1..={GQA_MAX_SLOTS} slot buckets, got {}",
            buckets.len()
        )));
    }
    for (i, b) in buckets.iter().enumerate() {
        if !(1..=32).contains(&b.nwg) || !matches!(b.nsg, 1 | 2 | 4) {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_vec_hybrid_batched_gqa: slot {i} bucket {b:?} outside nwg 1..=32, nsg {{1,2,4}}"
            )));
        }
    }
    let n_q = buckets.len() as u32;
    let nwg_max = buckets.iter().map(|b| b.nwg).max().unwrap_or(1);
    let nsg_max = buckets.iter().map(|b| b.nsg).max().unwrap_or(1);
    let head_dim = params.head_dim;
    if head_dim == 512 && group != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: head_dim 512 requires group 1, got {group} \
             (grouped d512 V scale is not bitwise-stable across Metal toolchains)"
        )));
    }
    if !flash_attn_vec_hybrid_batched_gqa_group_is_legal(
        params.num_heads,
        params.num_kv_heads,
        head_dim,
        group,
        nsg_max,
    ) {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: group {group} illegal for heads {}/{} head_dim {head_dim} nsg_max {nsg_max}",
            params.num_heads, params.num_kv_heads
        )));
    }
    if output.dtype() != crate::DType::F32 || tmp.dtype() != crate::DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: output/tmp must be F32, got {:?}/{:?}",
            output.dtype(),
            tmp.dtype()
        )));
    }
    let out_bytes = n_q as usize * params.num_heads as usize * head_dim as usize * 4;
    if output.data_byte_len() < out_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: output holds {} bytes, need {out_bytes}",
            output.data_byte_len()
        )));
    }
    let (gpu_buckets, runs, tmp_floats) = plan_gqa_layout(params.num_heads, head_dim, buckets)?;
    if tmp.data_byte_len() < tmp_floats * 4 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid_batched_gqa: tmp holds {} bytes, need {}",
            tmp.data_byte_len(),
            tmp_floats * 4
        )));
    }

    let gpu_params = FlashAttnVecHybridParamsGpu {
        n_heads: params.num_heads,
        n_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        mask_type: params.mask_type,
        sliding_window: params.sliding_window,
        softcap: params.softcap,
        nwg: nwg_max,
        ring_start: params.ring_start,
        scale_factor_d512: params.scale_factor_d512,
        codebook_bits: params.codebook_bits,
        fuse_fwht_pre: params.fuse_fwht_pre,
        nsg: nsg_max,
        n_queries: n_q,
    };

    let kernel_name = match (group, head_dim) {
        (1, 256) => "flash_attn_vec_hybrid_batched_gqa1_dk256",
        (2, 256) => "flash_attn_vec_hybrid_batched_gqa2_dk256",
        (4, 256) => "flash_attn_vec_hybrid_batched_gqa4_dk256",
        (8, 256) => "flash_attn_vec_hybrid_batched_gqa8_dk256",
        (1, 512) => "flash_attn_vec_hybrid_batched_gqa1_dk512",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_vec_hybrid_batched_gqa: unsupported group {group} / head_dim {head_dim}"
            )))
        }
    };
    let cbits_const = (params.codebook_bits as i32, 50usize);
    let v_is_f16: i32 = match v_packed.dtype() {
        crate::DType::F16 => 1,
        _ => 0,
    };
    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(cbits_const.1, cbits_const.0), (51usize, v_is_f16)],
    )?;

    let shmem_bytes = flash_attn_vec_hybrid_batched_gqa_shmem_bytes(head_dim, group, nsg_max);

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    let threadgroups = MTLSize::new(
        n_q as u64,
        (params.num_heads / group) as u64,
        nwg_max as u64,
    );
    let threadgroup_size = MTLSize::new(32, nsg_max as u64, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_f16)),
            (3, KernelArg::Buffer(v_packed)),
            (4, KernelArg::Buffer(v_norms)),
            (5, KernelArg::Buffer(tmp)),
            (6, KernelArg::Buffer(slot_id_arr)),
            (7, KernelArg::Buffer(seq_pos_arr)),
            (8, KernelArg::Bytes(bytemuck::cast_slice(&gpu_buckets))),
            (9, KernelArg::Buffer(output)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    if runs.is_empty() {
        return Ok(());
    }
    encoder.memory_barrier();
    let reduce_kernel = match head_dim {
        256 => "flash_attn_vec_reduce_dk256",
        512 => "flash_attn_vec_reduce_dk512",
        _ => unreachable!(),
    };
    let row_floats = params.num_heads as usize * head_dim as usize;
    for run in &runs {
        let reduce_pipeline = registry.get_pipeline(reduce_kernel, device.metal_device())?;
        let nrows = run.slots as u32 * params.num_heads;
        let reduce_params = FlashAttnVecReduceParamsGpu { nrows };
        let tmp_view = tmp.slice_view((run.tmp_off_floats * 4) as u64, run.tmp_floats);
        let out_view = output.slice_view(
            (run.slot0 * row_floats * 4) as u64,
            run.slots * row_floats,
        );
        let nwg = run.nwg;
        encoder.encode_threadgroups_with_args(
            reduce_pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&reduce_params))),
                (1, KernelArg::Buffer(&tmp_view)),
                (2, KernelArg::Buffer(&out_view)),
                (3, KernelArg::Bytes(as_bytes(&nwg))),
            ],
            MTLSize::new(nrows as u64, 1, 1),
            MTLSize::new(32 * nwg as u64, 1, 1),
        );
    }

    Ok(())
}

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        // 15 u32-sized fields × 4 bytes = 60 bytes — must match TQ-HB layout
        // exactly so the kernel's `constant FlashAttnVecTqHbParams &params`
        // binding sees the same bytes.
        assert_eq!(std::mem::size_of::<FlashAttnVecHybridParamsGpu>(), 64);
    }

    #[test]
    fn test_validate_bad_bits() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 4,  // invalid for hybrid V-side (TQ-HB only takes 5/6/8)
            fuse_fwht_pre: 0,
            nsg: 1,
        };
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn test_validate_ok_8bit() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
            fuse_fwht_pre: 0,
            nsg: 1,
        };
        assert!(validate_params(&p).is_ok());
    }

    #[test]
    fn test_validate_bad_head_dim() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 128,  // unsupported
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
            fuse_fwht_pre: 0,
            nsg: 1,
        };
        assert!(validate_params(&p).is_err());
    }
}
