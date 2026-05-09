//! Flash attention vector kernel dispatch for higher-bit TurboQuant KV cache.
//!
//! Variant of `flash_attn_vec_tq` that reads K/V from byte-packed (1 byte/element)
//! higher-bit codebook indices. Supports 5-bit (32 centroids), 6-bit (64 centroids),
//! and 8-bit (256 centroids) Lloyd-Max codebooks for N(0,1).
//!
//! Bit-width is controlled at runtime via `FlashAttnVecTqHbParams::codebook_bits`.
//!
//! ADR-007 iter-24: measure Gate A/B/C at 5/6/8-bit to find smallest shippable bit-width.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the HB TQ flash attention vector kernel.
pub static FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_tq_hb.metal");

/// Register HB TQ flash attention vector shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_tq_hb_dk256", FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_tq_hb_dk512", FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE);
}

/// Parameters for the HB TQ flash attention vector kernel.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnVecTqHbParams {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub scale: f32,
    pub mask_type: u32,
    pub sliding_window: u32,
    pub softcap: f32,
    /// Ring buffer start slot (same semantics as FlashAttnVecTqParams::ring_start).
    pub ring_start: u32,
    /// Scale divisor for D=512 per-block norms (matches hadamard_quantize_kv_hb convention).
    pub scale_factor_d512: f32,
    /// Codebook bit-width: 5, 6, or 8.
    pub codebook_bits: u32,
    /// ADR-028 iter-106: when 1, the kernel applies FWHT-pre internally on Q
    /// (sign-premult + simd-shuffle butterfly + 1/sqrt(d) normalize). When 0
    /// (default — production-byte-identical), the caller must pre-rotate Q
    /// via `dispatch_fwht_sign_premult_f32` before this call.
    /// Setting this to 1 eliminates one dispatch + one forced memory_barrier
    /// per layer (~9% decode lever per ADR-028 iter-104).
    pub fuse_fwht_pre: u32,
    /// ADR-028 iter-127 Path D: number of simdgroups per workgroup (NSG axis).
    ///
    /// llama.cpp's K-loop uses `for (ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG)`
    /// to split K-blocks across NSG simdgroups within each workgroup, on top
    /// of NWG workgroups. At qwen-realistic kL=4096, NSG=4 cuts per-WG K-iters
    /// from 4 to 1 (predicted ~4× FA speedup, ~28% decode at qwen production).
    ///
    /// Constraints:
    /// - Must be ≥ 1.
    /// - Must be a power of 2 (1, 2, 4, ...) — required for clean cross-
    ///   simdgroup reduce + threadgroup memory layout.
    /// - threadgroup_size = (32, nsg, 1) → 32*nsg threads/workgroup.
    ///   Apple Metal max threads/threadgroup is 1024 → nsg ≤ 32. Practically
    ///   capped at 4 (matches llama.cpp).
    pub nsg: u32,
}

/// GPU-side parameter struct. Must match `FlashAttnVecTqHbParams` in the MSL exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecTqHbParamsGpu {
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
    /// ADR-028 iter-127 Path D: NSG axis. See `FlashAttnVecTqHbParams::nsg`.
    nsg: u32,
}

/// GPU-side reduce params. Reuses the same reduce kernel as flash_attn_vec_tq.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

fn validate_params(params: &FlashAttnVecTqHbParams) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_tq_hb: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_tq_hb: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    if !matches!(params.codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: codebook_bits must be 5, 6, or 8, got {}",
            params.codebook_bits
        )));
    }
    // ADR-028 iter-127 Path D: NSG must be a power of 2 in [1, 32].
    // 32 is Apple Metal's max threads/threadgroup divided by simdgroup-width
    // (1024 / 32). Practically capped at 4 (matches llama.cpp policy).
    if params.nsg == 0 || (params.nsg & (params.nsg - 1)) != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: nsg must be a power of 2 (1, 2, 4, ...), got {}",
            params.nsg
        )));
    }
    // ADR-028 iter-127c: kernel reduce uses a fixed-size NSG_MAX=4 stack array
    // for per-simdgroup rescale factors. Tighten validation to match the
    // kernel-side cap. llama.cpp also caps at nsg=4 (`ggml-metal-ops.cpp:2954`).
    if params.nsg > 4 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: nsg must be ≤ 4 (kernel reduce cap), got {}",
            params.nsg
        )));
    }
    Ok(())
}

/// ADR-028 iter-127 Path D: select NSG (simdgroups per workgroup) from kv_seq_len.
///
/// Mirrors llama.cpp's policy at `ggml-metal-ops.cpp:2953`:
/// `while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) { nsg *= 2; }`
///
/// With our nwg=32 (computed by `compute_nwg`) and ncpsg=32 (C in the metal
/// shader), the NSG schedule becomes:
/// - kL ≤ 2048 — nsg=1 (32 simdgroups suffice for 64 K-blocks at most)
/// - 2049 ≤ kL ≤ 4096 — nsg=2
/// - kL > 4096 — nsg=4 (cap, matches llama.cpp)
///
/// Override via `HF2Q_TQ_NSG` env var (1, 2, or 4 only). Default policy
/// keeps short-context behavior byte-identical (nsg=1) per
/// `feedback_metal_compiler_auto_optimizes_static_levers`.
pub fn compute_nsg(kv_seq_len: u32) -> u32 {
    if let Ok(v) = std::env::var("HF2Q_TQ_NSG") {
        if let Ok(n) = v.parse::<u32>() {
            if n == 1 || n == 2 || n == 4 {
                return n;
            }
        }
    }
    // ADR-028 iter-127d Path D: adaptive NSG policy from measured bench data.
    //
    // bench_fa_vec_tq_hb_gemma_decode (mlx-native commit 5aafd7a, M5 Max, NWG=32):
    //
    // | kL    | NSG=1 µs/call | NSG=4 µs/call | speedup |
    // |-------|---------------|---------------|---------|
    // | 1024  | 44.21         | 53.59         | 0.83× (overhead dominates) |
    // | 4096  | 208.71        | 113.32        | **1.84× faster** |
    // | 8192  | 423.79        | 231.10        | **1.83× faster** |
    //
    // Threshold: kL > 1024 (i.e. K_blocks > NWG=32) crosses the K-iter loop
    // into >1 iter/simdgroup at NSG=1, and NSG=4 splits that work 4-way.
    // Below threshold, the cross-simdgroup reduce overhead dominates (kL≤1024
    // is already saturated at NSG=1).
    //
    // Why NSG=4 not NSG=2: bench shows NSG=4 strictly beats NSG=2 at all
    // kL > 1024 (we measured 4096: NSG=2=184µs vs NSG=4=113µs).
    // Why not NSG > 4: validate_params caps at 4 (kernel NSG_MAX=4 ms_arr).
    if kv_seq_len > 1024 { 4 } else { 1 }
}

fn compute_nwg(kv_seq_len: u32) -> u32 {
    if let Ok(v) = std::env::var("HF2Q_TQ_NWG") {
        if let Ok(n) = v.parse::<u32>() {
            if n >= 1 && n <= 32 {
                return n;
            }
        }
    }
    // ADR-028 iter-119 kL-adaptive nwg.
    //
    // iter-103 measured FA-vec-tq-hb GPU pure-time at gemma decode shape
    // (16 query heads, head_dim=256): 40 µs/call FLAT for kL ∈ [64, 512],
    // then linear (78 µs at kL=1024, 115 µs at kL=1536). Past 512 the
    // 16 simdgroups split-K saturate before kL is consumed, so each WG
    // does multiple outer-loop iterations.
    //
    // iter-100 measured nwg ∈ {16, 32} at short kL (≤170): identical
    // throughput (GPU-saturated, doubling parallelism doesn't help).
    // iter-118 measured at long kL (~860 with 800-token decode): nwg=32
    // gave +4.7% production decode (62.4 vs 59.6 t/s). Threshold around
    // 512: short kL prefers 16 (less reduce overhead), long kL prefers
    // 32 (more parallelism for the longer K-dim).
    //
    // Per `feedback_metal_compiler_auto_optimizes_static_levers`, this
    // is the FIRST positive lever after 11 falsifications; verified
    // empirically with HF2Q_TQ_NWG=32 vs default at long-context production
    // bench, NOT static-analysis-derived. No coherence change (FA-vec-tq-hb
    // produces byte-identical output regardless of nwg).
    if kv_seq_len > 512 { 32 } else { 16 }
}

/// Dispatch HB TQ flash attention vector kernel (5/6/8-bit byte-packed K/V).
///
/// Same calling convention as `flash_attn_vec_tq` except K/V are byte-packed
/// (1 byte/element) from the higher-bit encode path.
///
/// FWHT of Q must be applied by the caller before this call; inverse FWHT of
/// output must be applied by the caller after. Same as the 4-bit TQ path.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_tq_hb(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_packed: &MlxBuffer,
    k_norms: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &FlashAttnVecTqHbParams,
) -> Result<()> {
    validate_params(params)?;

    let head_dim = params.head_dim;
    let nwg = compute_nwg(params.kv_seq_len);

    let gpu_params = FlashAttnVecTqHbParamsGpu {
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
    };

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_tq_hb_dk256",
        512 => "flash_attn_vec_tq_hb_dk512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: unsupported head_dim {head_dim}"
        ))),
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    // ADR-028 iter-127b: shmem layout is NSG-aware:
    //   [0, PK)                                                  — Q (shared)
    //   [PK, PK + NSG*SH)                                        — per-simdgroup ss
    //   [PK + NSG*SH, PK + NSG*SH + NSG*2*PV)                    — per-simdgroup so4
    // At NSG=1 → `pk + sh + 2*pv` (pre-iter-127 layout, byte-identical).
    let nsg = params.nsg as usize;
    let shmem_halfs = pk + nsg * (sh + 2 * pv);
    let shmem_bytes = shmem_halfs * 2;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // ADR-028 iter-127a Path D: threadgroup is `(simdgroup_width=32, NSG, 1)`.
    // At NSG=1 (default), this is `(32, 1, 1)` — byte-identical to pre-iter-127
    // dispatch shape. The NSG axis is read by the kernel via `params.nsg`.
    let threadgroups = MTLSize::new(1, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, params.nsg as u64, 1);

    let dst_buf = if nwg == 1 { output } else { tmp };

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_packed)),
            (3, KernelArg::Buffer(k_norms)),
            (4, KernelArg::Buffer(v_packed)),
            (5, KernelArg::Buffer(v_norms)),
            (6, KernelArg::Buffer(dst_buf)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // Reduce kernel (NWG > 1)
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

/// Size in bytes of the temporary buffer needed for HB TQ SDPA.
pub fn tmp_buffer_bytes(num_heads: u32, head_dim: u32) -> usize {
    let nrows = num_heads as usize;
    let max_nwg = 32usize;
    let dv = head_dim as usize;
    (nrows * max_nwg * (dv + 2)) * std::mem::size_of::<f32>()
}

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        // ADR-028 iter-127a: 15 fields × 4 bytes = 60 bytes (added nsg).
        // iter-106 was 14 (added fuse_fwht_pre); iter-127a adds nsg.
        assert_eq!(std::mem::size_of::<FlashAttnVecTqHbParamsGpu>(), 60);
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
            codebook_bits: 4,  // invalid
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

    // ADR-028 iter-127a — NSG-axis validation tests.

    #[test]
    fn test_validate_nsg_zero_rejected() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8, num_kv_heads: 4, head_dim: 256,
            kv_seq_len: 64, kv_capacity: 1024, scale: 1.0, mask_type: 0,
            sliding_window: 0, softcap: 0.0, ring_start: 0,
            scale_factor_d512: 1.0, codebook_bits: 8, fuse_fwht_pre: 0,
            nsg: 0,
        };
        assert!(validate_params(&p).is_err(), "nsg=0 must reject");
    }

    #[test]
    fn test_validate_nsg_non_pow2_rejected() {
        // ADR-028 iter-127c: cap tightened to ≤ 4 (kernel NSG_MAX). Test
        // covers non-pow2 in [1, 4], plus the > 4 cap rejection.
        for nsg in [3u32, 5, 6, 7, 9, 16, 31, 33] {
            let p = FlashAttnVecTqHbParams {
                num_heads: 8, num_kv_heads: 4, head_dim: 256,
                kv_seq_len: 64, kv_capacity: 1024, scale: 1.0, mask_type: 0,
                sliding_window: 0, softcap: 0.0, ring_start: 0,
                scale_factor_d512: 1.0, codebook_bits: 8, fuse_fwht_pre: 0,
                nsg,
            };
            assert!(validate_params(&p).is_err(), "nsg={nsg} must reject (not pow-2 or > 4)");
        }
    }

    #[test]
    fn test_validate_nsg_pow2_accepted() {
        // ADR-028 iter-127c: only {1, 2, 4} accepted (matches kernel cap).
        for nsg in [1u32, 2, 4] {
            let p = FlashAttnVecTqHbParams {
                num_heads: 8, num_kv_heads: 4, head_dim: 256,
                kv_seq_len: 64, kv_capacity: 1024, scale: 1.0, mask_type: 0,
                sliding_window: 0, softcap: 0.0, ring_start: 0,
                scale_factor_d512: 1.0, codebook_bits: 8, fuse_fwht_pre: 0,
                nsg,
            };
            assert!(validate_params(&p).is_ok(), "nsg={nsg} (pow-2 ≤ 4) must accept");
        }
    }

    /// ADR-028 iter-127a: env-var-mutating tests serialize through a mutex to
    /// avoid races on parallel unit-test execution. `test_compute_nsg_default`
    /// and `test_compute_nsg_env_override` both touch `HF2Q_TQ_NSG`.
    static NSG_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_compute_nsg_adaptive_threshold() {
        let _guard = NSG_ENV_LOCK.lock().unwrap();
        // iter-127d adaptive policy: NSG=1 below threshold, NSG=4 above.
        // Threshold derived from bench data — see compute_nsg docstring.
        std::env::remove_var("HF2Q_TQ_NSG");
        // Below threshold: NSG=1 (cross-simdgroup reduce overhead dominates).
        for kl in [1u32, 64, 256, 1024] {
            assert_eq!(compute_nsg(kl), 1, "compute_nsg({kl}) must be 1 (kL ≤ 1024)");
        }
        // Above threshold: NSG=4 (engaged for ≥1.83× speedup at kL=4096+).
        for kl in [1025u32, 1536, 2048, 4096, 8192, 16384] {
            assert_eq!(compute_nsg(kl), 4, "compute_nsg({kl}) must be 4 (kL > 1024)");
        }
    }

    #[test]
    fn test_compute_nsg_env_override() {
        let _guard = NSG_ENV_LOCK.lock().unwrap();
        std::env::set_var("HF2Q_TQ_NSG", "4");
        assert_eq!(compute_nsg(64), 4);
        std::env::set_var("HF2Q_TQ_NSG", "2");
        assert_eq!(compute_nsg(64), 2);
        std::env::set_var("HF2Q_TQ_NSG", "1");
        assert_eq!(compute_nsg(64), 1);
        // Invalid values fall through to default.
        std::env::set_var("HF2Q_TQ_NSG", "3");
        assert_eq!(compute_nsg(64), 1);
        std::env::remove_var("HF2Q_TQ_NSG");
    }
}
