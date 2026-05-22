//! Flash attention vector kernel dispatch — SIMD-vectorized decode-path SDPA.
//!
//! Ported from llama.cpp's `flash_attn_ext_vec` kernel. This replaces the naive
//! SDPA kernel with a workgroup-parallel implementation that splits the KV cache
//! across `nwg` workgroups, each computing partial softmax results, then a
//! reduce kernel combines them.
//!
//! This kernel is optimized for the decode path (seq_len=1) with F32 Q/K/V.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

/// MSL source for the flash attention vector kernel (embedded at compile time).
pub static FLASH_ATTN_VEC_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec.metal");

/// Register flash attention vector shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_dk256", FLASH_ATTN_VEC_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_dk512", FLASH_ATTN_VEC_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_reduce_dk256", FLASH_ATTN_VEC_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_reduce_dk512", FLASH_ATTN_VEC_SHADER_SOURCE);
    // F16 KV variants (Phase 4a)
    registry.register_source("flash_attn_vec_f16kv_dk256", FLASH_ATTN_VEC_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_f16kv_dk512", FLASH_ATTN_VEC_SHADER_SOURCE);
}

/// Parameters for the flash attention vector kernel.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnVecParams {
    /// Number of query attention heads.
    pub num_heads: u32,
    /// Number of key/value attention heads (GQA: may be < num_heads).
    pub num_kv_heads: u32,
    /// Dimension of each attention head (256 or 512).
    pub head_dim: u32,
    /// Current KV sequence length (number of valid positions).
    pub kv_seq_len: u32,
    /// KV cache capacity (stride between KV heads in positions).
    pub kv_capacity: u32,
    /// Attention score scaling factor (e.g. 1/sqrt(head_dim) or 1.0).
    pub scale: f32,
    /// Mask type: 0=none, 1=causal, 2=sliding_window.
    pub mask_type: u32,
    /// Sliding window size (only used when mask_type == 2).
    pub sliding_window: u32,
    /// Logit softcapping (0 = disabled).
    pub softcap: f32,
    /// ADR-034 task #89 (2026-05-21) — number of queries to process
    /// per dispatch. Default 1 (decode). For spec-decode verify with
    /// `cur_len > 0` + `seq_len in [2, 8]`, call with `q_seq_len =
    /// seq_len`. The shader dispatches `q_seq_len` threadgroups in
    /// `grid.x`; each handles one (query, head) pair with causal mask
    /// `abs_pos = kv_seq_len - q_seq_len + iq1`. Peer-code reference:
    /// llama.cpp's `kernel_flash_attn_ext_vec` (NQPSG=1, ne01
    /// threadgroups in qL dim).
    pub q_seq_len: u32,
}

impl FlashAttnVecParams {
    /// Default `q_seq_len` for decode (single query). Use for all
    /// existing call sites; spec-decode verify sets `q_seq_len > 1`.
    pub const DEFAULT_Q_SEQ_LEN: u32 = 1;
}

/// GPU-side parameter struct. Must match the MSL `FlashAttnVecParams` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecParamsGpu {
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
    /// ADR-034 task #89 (2026-05-21) — see FlashAttnVecParams::q_seq_len.
    q_l: u32,
}

/// GPU-side reduce params. Must match MSL `FlashAttnVecReduceParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

/// Number of workgroups to split the KV cache across.
/// llama.cpp uses 32 as default. Must be <= 32 (one per SIMD lane in reduce).
const NWG: u32 = 32;

/// Validate flash attention parameters.
fn validate_params(params: &FlashAttnVecParams) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec: num_heads ({}) must be divisible by num_kv_heads ({})",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec: kv_capacity ({}) must be >= kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    // ADR-034 task #89: q_seq_len >= 1 invariant. The shader uses
    // `abs_pos = kv_seq_len - q_seq_len + iq1`, so q_seq_len > kv_seq_len
    // would underflow the unsigned subtraction.
    if params.q_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec: q_seq_len must be > 0".into(),
        ));
    }
    if params.q_seq_len > params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec: q_seq_len ({}) must be <= kv_seq_len ({})",
            params.q_seq_len, params.kv_seq_len
        )));
    }
    Ok(())
}

/// Dispatch flash attention vector kernel on the GPU.
///
/// This dispatches two Metal compute passes:
/// 1. The main kernel with NWG workgroups per head computing partial results
/// 2. The reduce kernel combining results from all workgroups
///
/// # Arguments
///
/// * `encoder`  — Command encoder to record dispatches into.
/// * `registry` — Kernel registry for pipeline lookup/compilation.
/// * `device`   — Metal device for buffer allocation.
/// * `q`        — Query buffer `[num_heads, 1, head_dim]`, F32.
/// * `k`        — Key buffer `[num_kv_heads, kv_capacity, head_dim]`, F32.
/// * `v`        — Value buffer `[num_kv_heads, kv_capacity, head_dim]`, F32.
/// * `output`   — Output buffer `[num_heads, 1, head_dim]`, F32, pre-allocated.
/// * `tmp`      — Temporary buffer for workgroup partial results, pre-allocated.
///                Size: `num_heads * nwg * (head_dim + 2) * sizeof(f32)` bytes.
/// * `params`   — Flash attention parameters.
pub fn flash_attn_vec(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &FlashAttnVecParams,
) -> Result<()> {
    validate_params(params)?;

    let head_dim = params.head_dim;
    let nwg = NWG;

    // Build GPU params.
    let gpu_params = FlashAttnVecParamsGpu {
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
        q_l: params.q_seq_len,
    };
    // Select kernel by head dimension and KV dtype.
    // F16 KV: K/V buffers are half-precision, halving bandwidth (Phase 4a).
    let kv_is_f16 = k.dtype() == DType::F16;
    let kernel_name = match (head_dim, kv_is_f16) {
        (256, false) => "flash_attn_vec_dk256",
        (512, false) => "flash_attn_vec_dk512",
        (256, true)  => "flash_attn_vec_f16kv_dk256",
        (512, true)  => "flash_attn_vec_f16kv_dk512",
        _ => unreachable!(), // validated above
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    // Shared memory size.
    // Layout: PK halfs (Q) + SH halfs (scratch) + 2*PV halfs (output as float4)
    // PK = PAD2(head_dim, 128), PV = PAD2(head_dim, 128)
    // SH = 4 * C = 128 halfs
    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32; // 4 * C = 128 halfs
    let shmem_halfs = pk + sh + 2 * pv;
    let shmem_bytes = shmem_halfs * 2; // 2 bytes per half

    // Tag for the reorder pass (Phase 4e.3): SDPA is NOT reorderable.
    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // Dispatch main kernel.
    // Grid: (q_seq_len, num_heads, nwg)
    // ADR-034 task #89 (2026-05-21) — q_seq_len = 1 (decode) keeps the
    // pre-change behavior exactly (single threadgroup per head, single
    // query at position kv_seq_len-1). q_seq_len > 1 (spec-decode
    // verify) dispatches one threadgroup per (query, head) pair; each
    // computes attention for its own query at its own causal position.
    let threadgroups = MTLSize::new(
        params.q_seq_len as u64,
        params.num_heads as u64,
        nwg as u64,
    );
    let threadgroup_size = MTLSize::new(32, 1, 1); // 1 simdgroup of 32 threads

    // Pass params as inline bytes — no Metal buffer allocation.
    // This eliminates 1 [MTLDevice newBufferWithLength:] per SDPA call (30/token).
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k)),
            (3, KernelArg::Buffer(v)),
            (4, KernelArg::Buffer(tmp)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // --- Reduce kernel ---
    // Only needed when NWG > 1.
    // Barrier: reduce reads `tmp` written by the main dispatch above.
    // With MTLDispatchTypeConcurrent, both dispatches would run simultaneously
    // without this barrier, causing the reduce to read stale/partial `tmp` data.
    if nwg > 1 {
        encoder.memory_barrier();
        // ADR-034 task #89 (2026-05-21) — nrows = num_heads * q_seq_len.
        // At q_seq_len=1 this is byte-identical to pre-task-#89 (was just
        // num_heads). At q_seq_len>1, the reduce dispatches one
        // threadgroup per (query, head) pair, matching the main kernel's
        // partial-output layout `dst[rid * DV4 * NWG + ...]` where
        // rid in [0, num_heads * q_seq_len).
        let total_rows = params.num_heads * params.q_seq_len;
        let reduce_params = FlashAttnVecReduceParamsGpu {
            nrows: total_rows,
        };

        let reduce_kernel = match head_dim {
            256 => "flash_attn_vec_reduce_dk256",
            512 => "flash_attn_vec_reduce_dk512",
            _ => unreachable!(),
        };
        let reduce_pipeline =
            registry.get_pipeline(reduce_kernel, device.metal_device())?;

        // Grid: (num_heads * q_seq_len, 1, 1), Threadgroup: (32*NWG, 1, 1)
        let reduce_tg = MTLSize::new(total_rows as u64, 1, 1);
        let reduce_tg_size = MTLSize::new(32 * nwg as u64, 1, 1);

        // Annotate reads/writes for the capture graph so the reorder pass
        // (Phase 4e.3) can track data dependencies on the reduce kernel.
        {
            let read_ranges = vec![
                {
                    let s = tmp.contents_ptr() as usize;
                    (s, s + tmp.byte_len())
                },
            ];
            let write_ranges = vec![
                {
                    let s = output.contents_ptr() as usize;
                    (s, s + output.byte_len())
                },
            ];
            encoder.set_pending_buffer_ranges(read_ranges, write_ranges);
        }

        // Pass params as inline bytes — no Metal buffer allocation.
        // This eliminates 2 [MTLDevice newBufferWithLength:] per SDPA call (60/token).
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

/// Compute the size in bytes of the temporary buffer needed for flash_attn_vec.
///
/// The temp buffer stores partial results from NWG workgroups:
/// - `nrows * head_dim * NWG` floats for the partial output vectors
/// - `nrows * 2 * NWG` floats for the S and M values
///
/// ADR-034 task #89 (2026-05-21) — `nrows = num_heads * q_seq_len`. At
/// the default `q_seq_len = 1` this returns the same value as
/// pre-task-#89 (`num_heads * NWG * (head_dim + 2) * 4`). For
/// `q_seq_len > 1` the buffer scales linearly with qL.
pub fn tmp_buffer_bytes(num_heads: u32, head_dim: u32) -> usize {
    tmp_buffer_bytes_with_qL(num_heads, head_dim, 1)
}

/// ADR-034 task #89 (2026-05-21) — qL-aware temp buffer size. Callers
/// that dispatch with `q_seq_len > 1` (spec-decode verify) must use
/// this to size the tmp buffer correctly. `tmp_buffer_bytes(...)` is
/// kept as the qL=1 wrapper for backward compatibility.
#[allow(non_snake_case)]
pub fn tmp_buffer_bytes_with_qL(num_heads: u32, head_dim: u32, q_seq_len: u32) -> usize {
    let nrows = (num_heads as usize) * (q_seq_len as usize);
    let nwg = NWG as usize;
    let dv = head_dim as usize;
    // DV * NWG floats per row for output, plus 2 * NWG floats per row for S/M.
    (nrows * nwg * (dv + 2)) * std::mem::size_of::<f32>()
}

/// Pad x up to next multiple of n (n must be power of 2).
fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_params_ok() {
        let p = FlashAttnVecParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 1,
            sliding_window: 0,
            softcap: 0.0,
            q_seq_len: FlashAttnVecParams::DEFAULT_Q_SEQ_LEN,
        };
        assert!(validate_params(&p).is_ok());
    }

    #[test]
    fn test_validate_params_bad_head_dim() {
        let p = FlashAttnVecParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            q_seq_len: FlashAttnVecParams::DEFAULT_Q_SEQ_LEN,
        };
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn adr_034_task_89_q_seq_len_zero_rejected_2026_05_21() {
        let p = FlashAttnVecParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 1,
            sliding_window: 0,
            softcap: 0.0,
            q_seq_len: 0,
        };
        let err = validate_params(&p).unwrap_err().to_string();
        assert!(err.contains("q_seq_len must be > 0"), "got: {err}");
    }

    #[test]
    fn adr_034_task_89_q_seq_len_exceeds_kv_rejected_2026_05_21() {
        let p = FlashAttnVecParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            kv_seq_len: 4,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 1,
            sliding_window: 0,
            softcap: 0.0,
            q_seq_len: 8, // > kv_seq_len, would underflow abs_pos
        };
        let err = validate_params(&p).unwrap_err().to_string();
        assert!(
            err.contains("must be <= kv_seq_len"),
            "got: {err}"
        );
    }

    #[test]
    fn test_gpu_params_layout() {
        assert_eq!(
            std::mem::size_of::<FlashAttnVecParamsGpu>(),
            // ADR-034 task #89 (2026-05-21): added q_l field; was 40
            // bytes (10 fields), now 44 bytes (11 fields).
            44, // 11 x u32/f32 = 44 bytes
        );
    }

    #[test]
    fn test_tmp_buffer_size() {
        // 16 heads, dk256, nwg=32
        let bytes = tmp_buffer_bytes(16, 256);
        // 16 * 32 * (256 + 2) * 4 = 16 * 32 * 258 * 4 = 528384
        assert_eq!(bytes, 16 * 32 * 258 * 4);
    }
}
