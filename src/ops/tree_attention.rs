//! Tree-attention vector kernel dispatch — ADR-037 Phase E1.1 (2026-05-22).
//!
//! Variant of `flash_attn_vec` that consumes an *explicit* mask buffer
//! shaped `[q_l, mask_stride]` (row-major float) instead of computing
//! an implicit causal mask from absolute position. Tree topologies for
//! EAGLE-3 + dynamic tree speculative decoding (chains, fixed trees,
//! dynamic asymmetric trees) all reduce to this representation.
//!
//! Byte-identity contract: when the explicit mask matches the implicit
//! causal mask that `flash_attn_vec` would compute for the same params,
//! this kernel's output is byte-identical to `flash_attn_vec`'s output.
//! Validated by `tests/test_tree_attention_e1_1_parity.rs`.
//!
//! Reuses `flash_attn_vec_reduce` for the second pass — output layout
//! is identical.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

/// MSL source for the tree-attention kernel (embedded at compile time).
pub static TREE_ATTENTION_SHADER_SOURCE: &str =
    include_str!("../shaders/tree_attention.metal");

/// Register tree-attention shader source with the given kernel registry.
///
/// **Dependency**: this helper registers only the tree-attention main
/// kernels. The reduce pass (`flash_attn_vec_reduce_*`) is reused
/// verbatim from `flash_attn_vec`, so callers that use a minimal
/// registry must ALSO call `flash_attn_vec::register(registry)`. The
/// default `KernelRegistry::new()` pre-registers both — the explicit
/// `register()` helpers exist for tooling that builds incremental
/// registries.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("tree_attention_dk128", TREE_ATTENTION_SHADER_SOURCE);
    registry.register_source("tree_attention_dk256", TREE_ATTENTION_SHADER_SOURCE);
    registry.register_source("tree_attention_dk512", TREE_ATTENTION_SHADER_SOURCE);
    registry.register_source("tree_attention_f16kv_dk128", TREE_ATTENTION_SHADER_SOURCE);
    registry.register_source("tree_attention_f16kv_dk256", TREE_ATTENTION_SHADER_SOURCE);
    registry.register_source("tree_attention_f16kv_dk512", TREE_ATTENTION_SHADER_SOURCE);
}

/// Sentinel value for masked positions in the tree-mask buffer. Matches
/// the flash_attn_vec implicit-causal sentinel (-MAXHALF) so that
/// chunks fully composed of masked cells take the same early-exit
/// branch in the shader.
pub const TREE_MASK_MASKED: f32 = -65504.0;

/// Sentinel value for attended positions.
pub const TREE_MASK_ATTENDED: f32 = 0.0;

/// Parameters for the tree-attention kernel.
#[derive(Debug, Clone, Copy)]
pub struct TreeAttentionParams {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub scale: f32,
    /// Number of tree nodes (queries) to verify in this dispatch.
    pub q_seq_len: u32,
    /// Stride between rows of the mask buffer. Must be >= `kv_seq_len`.
    /// Callers usually pass `mask_stride == kv_seq_len` for tight
    /// packing; higher values are legal but waste memory.
    pub mask_stride: u32,
}

/// GPU-side parameter struct. Must match the MSL `TreeAttentionParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeAttentionParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    nwg: u32,
    q_l: u32,
    mask_stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

/// Number of workgroups (matches `flash_attn_vec::NWG`). The reduce
/// kernel assumes one (host-)thread per NWG and NWG <= 32.
const NWG: u32 = 32;

/// Codex /cfa Phase E1 gate (2026-05-22): defensive buffer dtype +
/// byte-length validation. Caught:
///   - K/V dtype mismatch (would read V with wrong stride)
///   - Q/mask/output/tmp dtype slips (must all be F32)
///   - Undersized buffers (GPU OOB read/write)
/// Without these checks the kernel reads/writes purely from params,
/// so undersized buffers escape to GPU.
#[allow(clippy::too_many_arguments)]
fn validate_buffers(
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    tree_mask: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &TreeAttentionParams,
) -> Result<()> {
    // --- dtype contract ---
    if q.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: q dtype must be F32, got {:?}",
            q.dtype()
        )));
    }
    if k.dtype() != v.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: k.dtype ({:?}) must match v.dtype ({:?})",
            k.dtype(),
            v.dtype()
        )));
    }
    if !matches!(k.dtype(), DType::F32 | DType::F16) {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: k/v dtype must be F32 or F16, got {:?}",
            k.dtype()
        )));
    }
    if tree_mask.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: tree_mask dtype must be F32, got {:?}",
            tree_mask.dtype()
        )));
    }
    if output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: output dtype must be F32, got {:?}",
            output.dtype()
        )));
    }
    if tmp.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: tmp dtype must be F32, got {:?}",
            tmp.dtype()
        )));
    }

    // --- byte-length contract ---
    let head_dim = params.head_dim as usize;
    let num_heads = params.num_heads as usize;
    let num_kv_heads = params.num_kv_heads as usize;
    let q_seq_len = params.q_seq_len as usize;
    let kv_capacity = params.kv_capacity as usize;
    let mask_stride = params.mask_stride as usize;
    let kv_dtype_bytes: usize = match k.dtype() {
        DType::F32 => 4,
        DType::F16 => 2,
        _ => unreachable!("validated above"),
    };

    let req_q = num_heads * q_seq_len * head_dim * 4;
    let req_kv = num_kv_heads * kv_capacity * head_dim * kv_dtype_bytes;
    let req_mask = q_seq_len * mask_stride * 4;
    let req_output = num_heads * q_seq_len * head_dim * 4;
    let req_tmp =
        tmp_buffer_bytes(params.num_heads, params.head_dim, params.q_seq_len);

    if q.byte_len() < req_q {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: q buffer too small: have {} bytes, need >= {}",
            q.byte_len(),
            req_q
        )));
    }
    if k.byte_len() < req_kv {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: k buffer too small: have {} bytes, need >= {}",
            k.byte_len(),
            req_kv
        )));
    }
    if v.byte_len() < req_kv {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: v buffer too small: have {} bytes, need >= {}",
            v.byte_len(),
            req_kv
        )));
    }
    if tree_mask.byte_len() < req_mask {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: tree_mask buffer too small: have {} bytes, need >= {}",
            tree_mask.byte_len(),
            req_mask
        )));
    }
    if output.byte_len() < req_output {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: output buffer too small: have {} bytes, need >= {}",
            output.byte_len(),
            req_output
        )));
    }
    if tmp.byte_len() < req_tmp {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: tmp buffer too small: have {} bytes, need >= {}",
            tmp.byte_len(),
            req_tmp
        )));
    }

    Ok(())
}

fn validate_params(params: &TreeAttentionParams) -> Result<()> {
    // ADR-037 Phase E4b.6 (2026-05-22): added 128 for Qwen 3.6 27B.
    // Template `static_assert(DK % 32 == 0)` satisfied for all three.
    if params.head_dim != 128 && params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: head_dim must be 128, 256, or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "tree_attention: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: num_heads ({}) must be divisible by num_kv_heads ({})",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "tree_attention: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: kv_capacity ({}) must be >= kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    if params.q_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "tree_attention: q_seq_len must be > 0".into(),
        ));
    }
    if params.mask_stride < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "tree_attention: mask_stride ({}) must be >= kv_seq_len ({})",
            params.mask_stride, params.kv_seq_len
        )));
    }
    Ok(())
}

/// Dispatch tree-attention kernel on the GPU.
///
/// Two Metal compute passes:
/// 1. Main kernel with NWG workgroups per (query, head) computing partial results
/// 2. `flash_attn_vec_reduce` (reused — same output layout) combining results
///
/// # Buffers
/// * `q` — `[num_heads, q_seq_len, head_dim]`, F32.
/// * `k` — `[num_kv_heads, kv_capacity, head_dim]`, F32 or F16.
///   K and V must share the same dtype (validated below).
/// * `v` — `[num_kv_heads, kv_capacity, head_dim]`, same dtype as K.
/// * `tree_mask` — `[q_seq_len, mask_stride]`, F32. Cell (i, j) ∈
///   `{TREE_MASK_ATTENDED (0.0), TREE_MASK_MASKED (-65504.0)}` indicating
///   whether query i can attend to KV position j. Cells with j >=
///   kv_seq_len within a row are ignored (bounds-checked by shader).
/// * `output` — **`[q_seq_len, num_heads, head_dim]`**, F32, pre-allocated.
///   NOTE: output is **query-outer, head-inner** — different from Q
///   layout — because the kernel writes via `rid = iq2 + iq1 *
///   n_heads` (see `flash_attn_vec.metal:322`, inherited identically
///   by tree_attention.metal). Callers reading row `iq1` for head `h`
///   should index `output[iq1 * num_heads * head_dim + h * head_dim + d]`.
/// * `tmp` — partial-result scratch sized by `tmp_buffer_bytes(...)`.
pub fn tree_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    tree_mask: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &TreeAttentionParams,
) -> Result<()> {
    validate_params(params)?;
    validate_buffers(q, k, v, tree_mask, output, tmp, params)?;

    let head_dim = params.head_dim;
    let nwg = NWG;

    let gpu_params = TreeAttentionParamsGpu {
        n_heads: params.num_heads,
        n_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        nwg,
        q_l: params.q_seq_len,
        mask_stride: params.mask_stride,
    };

    // Codex /cfa Phase E1 gate (2026-05-22): K and V dtype must match;
    // shader templates K and V on the same `KV_T` type. K/V mismatch
    // would cause one side to be read with the wrong stride/precision.
    // Q/tree_mask/output/tmp are always F32 (validated below).
    let kv_is_f16 = k.dtype() == DType::F16;
    let kernel_name = match (head_dim, kv_is_f16) {
        (128, false) => "tree_attention_dk128",
        (256, false) => "tree_attention_dk256",
        (512, false) => "tree_attention_dk512",
        (128, true)  => "tree_attention_f16kv_dk128",
        (256, true)  => "tree_attention_f16kv_dk256",
        (512, true)  => "tree_attention_f16kv_dk512",
        _ => unreachable!(),
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    // Shared memory layout mirrors flash_attn_vec exactly.
    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    let shmem_halfs = pk + sh + 2 * pv;
    let shmem_bytes = shmem_halfs * 2;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    let threadgroups = MTLSize::new(
        params.q_seq_len as u64,
        params.num_heads as u64,
        nwg as u64,
    );
    let threadgroup_size = MTLSize::new(32, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k)),
            (3, KernelArg::Buffer(v)),
            (4, KernelArg::Buffer(tmp)),
            (5, KernelArg::Buffer(tree_mask)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // --- Reduce pass (reuses flash_attn_vec_reduce — same output layout) ---
    if nwg > 1 {
        encoder.memory_barrier();
        let total_rows = params.num_heads * params.q_seq_len;
        let reduce_params = FlashAttnVecReduceParamsGpu { nrows: total_rows };

        let reduce_kernel = match head_dim {
            128 => "flash_attn_vec_reduce_dk128",
            256 => "flash_attn_vec_reduce_dk256",
            512 => "flash_attn_vec_reduce_dk512",
            _ => unreachable!(),
        };
        let reduce_pipeline =
            registry.get_pipeline(reduce_kernel, device.metal_device())?;

        let reduce_tg = MTLSize::new(total_rows as u64, 1, 1);
        let reduce_tg_size = MTLSize::new(32 * nwg as u64, 1, 1);

        {
            let read_ranges = vec![{
                let s = tmp.contents_ptr() as usize;
                (s, s + tmp.byte_len())
            }];
            let write_ranges = vec![{
                let s = output.contents_ptr() as usize;
                (s, s + output.byte_len())
            }];
            encoder.set_pending_buffer_ranges(read_ranges, write_ranges);
        }

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

/// Compute the temp buffer size needed for tree-attention. Identical
/// to flash_attn_vec's tmp buffer (same output layout) — kept as a
/// separate function for clarity at call sites.
#[allow(non_snake_case)]
pub fn tmp_buffer_bytes(num_heads: u32, head_dim: u32, q_seq_len: u32) -> usize {
    let nrows = (num_heads as usize) * (q_seq_len as usize);
    let nwg = NWG as usize;
    let dv = head_dim as usize;
    (nrows * nwg * (dv + 2)) * std::mem::size_of::<f32>()
}

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_params_ok() {
        let p = TreeAttentionParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            q_seq_len: 1,
            mask_stride: 100,
        };
        assert!(validate_params(&p).is_ok());
    }

    #[test]
    fn test_validate_params_mask_stride_too_small() {
        let p = TreeAttentionParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            q_seq_len: 1,
            mask_stride: 50,
        };
        let err = validate_params(&p).unwrap_err().to_string();
        assert!(
            err.contains("mask_stride"),
            "expected mask_stride error, got: {err}"
        );
    }

    #[test]
    fn test_validate_params_bad_head_dim() {
        // 64 isn't supported (only 128, 256, 512 are templated).
        let p = TreeAttentionParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            q_seq_len: 1,
            mask_stride: 100,
        };
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn adr_037_e4b6_validate_params_accepts_dk128_2026_05_22() {
        // Qwen 3.6 27B uses head_dim=128. Added by Phase E4b.6.
        let p = TreeAttentionParams {
            num_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            kv_seq_len: 64,
            kv_capacity: 256,
            scale: 1.0,
            q_seq_len: 1,
            mask_stride: 64,
        };
        validate_params(&p).expect("dk128 should validate");
    }

    #[test]
    fn test_validate_params_q_seq_len_zero() {
        let p = TreeAttentionParams {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            kv_seq_len: 100,
            kv_capacity: 1024,
            scale: 1.0,
            q_seq_len: 0,
            mask_stride: 100,
        };
        let err = validate_params(&p).unwrap_err().to_string();
        assert!(err.contains("q_seq_len"), "got: {err}");
    }

    #[test]
    fn test_gpu_params_layout() {
        // 9 fields x 4 bytes = 36 bytes
        assert_eq!(std::mem::size_of::<TreeAttentionParamsGpu>(), 36);
    }

    #[test]
    fn test_tmp_buffer_bytes() {
        // 16 heads, dk256, q_seq_len=1, nwg=32
        // 16 * 1 * 32 * (256 + 2) * 4 = 528_384
        let bytes = tmp_buffer_bytes(16, 256, 1);
        assert_eq!(bytes, 16 * 32 * 258 * 4);
    }
}
