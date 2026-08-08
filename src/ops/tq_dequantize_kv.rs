//! TQ KV dequantize kernel dispatch — iter-20 Leg F ablation.
//!
//! Reads nibble-packed TurboQuant KV cache at one position and writes a
//! dense F32 buffer of shape `[num_kv_heads, head_dim]`.
//!
//! This isolates the SDPA kernel math from TQ representation noise:
//!   1. K/V are still encoded via `dispatch_hadamard_quantize_kv` (TQ path).
//!   2. This kernel decodes them back to F32 in the FWHT-rotated domain.
//!   3. Caller dispatches `flash_attn_vec` (dense) with those F32 buffers.
//!
//! Decision interpretation:
//!   Leg_F ≥ 3094 → TQ SDPA kernel has a math bug; dense SDPA on TQ K/V recovers parity.
//!   Leg_F ≈ 127  → Representation floor: 4-bit encode/decode round-trip is the source
//!                  of the byte-prefix gap, not a kernel bug.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_threadgroups_with_args, KernelArg};

/// MSL source for the TQ KV dequantize kernel (embedded at compile time).
pub static TQ_DEQUANTIZE_KV_SHADER_SOURCE: &str = include_str!("../shaders/tq_dequantize_kv.metal");

/// Register the `tq_dequantize_kv` shader with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("tq_dequantize_kv", TQ_DEQUANTIZE_KV_SHADER_SOURCE);
    registry.register_source(
        "tq_dequantize_hb_kv_seq_f16",
        TQ_DEQUANTIZE_KV_SHADER_SOURCE,
    );
}

/// GPU-side parameter struct. Must match `TqDequantizeKvParams` in the MSL exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqDequantizeKvParamsGpu {
    head_dim: u32,
    num_kv_heads: u32,
    read_pos: u32,
    cache_capacity: u32,
    norms_per_pos: u32,
    scale_factor_d512: f32,
}

/// Dispatch the TQ KV dequantize kernel.
///
/// Reads the TQ-encoded K or V at `read_pos` from `packed` and `norms` and
/// writes the dequantized F32 values to `dst` of shape `[num_kv_heads, head_dim]`.
///
/// The output is in the FWHT-rotated domain — the same domain the
/// `hadamard_quantize_kv` kernel encodes into. The caller must apply an
/// inverse FWHT if the original (non-rotated) K/V is needed; for the Leg F
/// ablation this is NOT required because Q was also pre-rotated by FWHT before
/// `flash_attn_vec`, so the rotated-domain dot products are correct.
///
/// # Arguments
///
/// * `encoder`           — Command encoder.
/// * `registry`          — Kernel registry.
/// * `device`            — Metal device.
/// * `packed`            — `[num_kv_heads, cache_capacity, head_dim/2]` u8.
/// * `norms`             — `[num_kv_heads, cache_capacity, norms_per_pos]` f32.
/// * `dst`               — `[num_kv_heads, head_dim]` f32 output buffer (pre-allocated).
/// * `num_kv_heads`      — Number of KV heads.
/// * `head_dim`          — Head dimension (256 or 512).
/// * `cache_capacity`    — KV cache capacity.
/// * `read_pos`          — Logical cache position to read (already wrapped for ring buffers).
/// * `scale_factor_d512` — D=512 per-block norm scale divisor (1.0 = bare, iter-16 control).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq_dequantize_kv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    dst: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    read_pos: u32,
    scale_factor_d512: f32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "tq_dequantize_kv: head_dim must be a power of two, got {}",
            head_dim
        )));
    }

    // Validate dst buffer has room for [num_kv_heads, head_dim] f32.
    let required_dst = (num_kv_heads as u64) * (head_dim as u64);
    if (dst.element_count() as u64) < required_dst {
        return Err(MlxError::InvalidArgument(format!(
            "tq_dequantize_kv: dst has {} elements, need {}",
            dst.element_count(),
            required_dst
        )));
    }

    let norms_per_pos = (head_dim / 256).max(1);

    let params = TqDequantizeKvParamsGpu {
        head_dim,
        num_kv_heads,
        read_pos,
        cache_capacity,
        norms_per_pos,
        scale_factor_d512,
    };
    let params_bytes = bytemuck::bytes_of(&params);

    let pipeline = registry.get_pipeline("tq_dequantize_kv", device)?;

    // One threadgroup per KV head; head_dim threads per threadgroup.
    let threadgroups = MTLSize {
        width: num_kv_heads as u64,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: head_dim.min(1024) as u64,
        height: 1,
        depth: 1,
    };

    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(packed)),
            (1, KernelArg::Buffer(norms)),
            (2, KernelArg::Buffer(dst)),
            (3, KernelArg::Bytes(params_bytes)),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}

// ============================================================================
// Track B (iter-21): higher-bit dequantize dispatch.
// ============================================================================

/// GPU-side params for the higher-bit dequantize kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqDequantizeHbKvParamsGpu {
    head_dim: u32,
    num_kv_heads: u32,
    read_pos: u32,
    cache_capacity: u32,
    norms_per_pos: u32,
    scale_factor_d512: f32,
    codebook_bits: u32, // 5 or 6
}

/// Dispatch the higher-bit TQ KV dequantize kernel.
///
/// Reads byte-packed 5-bit, 6-bit, or 8-bit indices from `packed` at `read_pos` and
/// writes F32 dequantized values to `dst` in the FWHT-rotated domain.
/// Same scale convention as `dispatch_tq_dequantize_kv`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq_dequantize_hb_kv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    packed: &MlxBuffer, // [nkv, capacity, head_dim] u8 (byte-packed)
    norms: &MlxBuffer,
    dst: &MlxBuffer, // [nkv, head_dim] f32
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    read_pos: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv: codebook_bits must be 5, 6, or 8, got {}",
            codebook_bits
        )));
    }

    let norms_per_pos = (head_dim / 256).max(1);

    let params = TqDequantizeHbKvParamsGpu {
        head_dim,
        num_kv_heads,
        read_pos,
        cache_capacity,
        norms_per_pos,
        scale_factor_d512,
        codebook_bits,
    };
    let params_bytes = bytemuck::bytes_of(&params);

    let pipeline = registry.get_pipeline("tq_dequantize_hb_kv", device)?;

    let threadgroups = MTLSize {
        width: num_kv_heads as u64,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: head_dim.min(1024) as u64,
        height: 1,
        depth: 1,
    };

    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(packed)),
            (1, KernelArg::Buffer(norms)),
            (2, KernelArg::Buffer(dst)),
            (3, KernelArg::Bytes(params_bytes)),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}

// ============================================================================
// ADR-027 Phase B iter-30 (hf2q sub-sub-iter 23c-β.1): sequence-batch dequant.
// ============================================================================

/// GPU-side params for the seq-variant. MUST match the MSL struct exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqDequantizeHbKvSeqParamsGpu {
    head_dim: u32,
    num_kv_heads: u32,
    start_pos: u32,
    n_tokens: u32,
    cache_capacity: u32,
    norms_per_pos: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
}

/// Sequence-batch TQ KV dequantize: reads positions
/// `[start_pos..start_pos+n_tokens)` from a higher-bit (5/6/8-bit) byte-packed
/// KV cache and writes dense F32 values to `dst` at layout
/// `[num_kv_heads, n_tokens, head_dim]` (head-major) — matches hf2q's
/// full-attn KV cache layout.
///
/// **Parity contract** with [`dispatch_tq_dequantize_hb_kv`]: for any
/// `read_pos`, calling this function with `start_pos=read_pos`, `n_tokens=1`
/// produces byte-identical output to the per-position dispatcher. Verified
/// by `tq_dequantize_hb_kv_seq_n1_byte_identical_to_per_position` test.
///
/// **Why this exists**: hf2q's TQ-active prefill SDPA path needs the full
/// `[0..cur_len)` K/V dequanted into a temp F32 buffer for the dense prefill
/// kernel to read. The per-position dispatcher would require `cur_len`
/// separate dispatches; this kernel does it in one dispatch with
/// `(num_kv_heads × n_tokens)` threadgroups.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq_dequantize_hb_kv_seq(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    dst: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    start_pos: u32,
    n_tokens: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq: head_dim must be power of two, got {}",
            head_dim
        )));
    }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq: codebook_bits must be 5, 6, or 8, got {}",
            codebook_bits
        )));
    }
    let required_dst = (num_kv_heads as u64) * (n_tokens as u64) * (head_dim as u64);
    if (dst.element_count() as u64) < required_dst {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq: dst has {} elements, need {} \
             (num_kv_heads={} × n_tokens={} × head_dim={})",
            dst.element_count(),
            required_dst,
            num_kv_heads,
            n_tokens,
            head_dim
        )));
    }
    if (start_pos as u64) + (n_tokens as u64) > (cache_capacity as u64) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq: start_pos({}) + n_tokens({}) > cache_capacity({})",
            start_pos, n_tokens, cache_capacity
        )));
    }

    let norms_per_pos = (head_dim / 256).max(1);

    let params = TqDequantizeHbKvSeqParamsGpu {
        head_dim,
        num_kv_heads,
        start_pos,
        n_tokens,
        cache_capacity,
        norms_per_pos,
        scale_factor_d512,
        codebook_bits,
    };
    let params_bytes = bytemuck::bytes_of(&params);

    let pipeline = registry.get_pipeline("tq_dequantize_hb_kv_seq", device)?;

    let threadgroups = MTLSize {
        width: num_kv_heads as u64,
        height: n_tokens as u64,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: head_dim.min(1024) as u64,
        height: 1,
        depth: 1,
    };

    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(packed)),
            (1, KernelArg::Buffer(norms)),
            (2, KernelArg::Buffer(dst)),
            (3, KernelArg::Bytes(params_bytes)),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}

/// F16-output sibling of [`dispatch_tq_dequantize_hb_kv_seq`].
///
/// This is the bounded staging path for tiled prefill-resume attention: the
/// persistent cache stays byte-packed while one active layer/range is
/// expanded directly into `[num_kv_heads, n_tokens, head_dim]` F16 scratch.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq_dequantize_hb_kv_seq_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    dst: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    start_pos: u32,
    n_tokens: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq_f16: head_dim must be power of two, got {head_dim}"
        )));
    }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq_f16: codebook_bits must be 5, 6, or 8, got {codebook_bits}"
        )));
    }
    if dst.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq_f16: dst must be F16, got {:?}",
            dst.dtype()
        )));
    }
    let required_dst = num_kv_heads as u64 * n_tokens as u64 * head_dim as u64;
    if (dst.element_count() as u64) < required_dst {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq_f16: dst has {} elements, need {required_dst}",
            dst.element_count()
        )));
    }
    if start_pos as u64 + n_tokens as u64 > cache_capacity as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_tq_dequantize_hb_kv_seq_f16: start_pos({start_pos}) + n_tokens({n_tokens}) > cache_capacity({cache_capacity})"
        )));
    }

    let params = TqDequantizeHbKvSeqParamsGpu {
        head_dim,
        num_kv_heads,
        start_pos,
        n_tokens,
        cache_capacity,
        norms_per_pos: (head_dim / 256).max(1),
        scale_factor_d512,
        codebook_bits,
    };
    let pipeline = registry.get_pipeline("tq_dequantize_hb_kv_seq_f16", device)?;
    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(packed)),
            (1, KernelArg::Buffer(norms)),
            (2, KernelArg::Buffer(dst)),
            (3, KernelArg::Bytes(bytemuck::bytes_of(&params))),
        ],
        MTLSize {
            width: num_kv_heads as u64,
            height: n_tokens as u64,
            depth: 1,
        },
        MTLSize {
            width: head_dim.min(1024) as u64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::MlxDevice;

    /// Iter-30 deliverable: parity contract between the seq-variant and the
    /// per-position dispatcher. For any read_pos, calling the seq variant
    /// with n_tokens=1 must produce byte-identical output.
    #[test]
    fn tq_dequantize_hb_kv_seq_n1_byte_identical_to_per_position() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut registry = KernelRegistry::new();

        let num_kv_heads: u32 = 2;
        let cap: u32 = 8;
        let head_dim: u32 = 256;
        let nbytes_packed = (num_kv_heads * cap * head_dim) as usize;
        let mut packed_cpu = vec![0u8; nbytes_packed];
        for (i, b) in packed_cpu.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let nelems_norms = (num_kv_heads * cap * 1) as usize;
        let mut norms_cpu = vec![0f32; nelems_norms];
        for (i, n) in norms_cpu.iter_mut().enumerate() {
            *n = 0.5 + (i as f32) * 0.0625;
        }

        let mut packed = device
            .alloc_buffer(nbytes_packed, DType::U8, vec![nbytes_packed])
            .expect("alloc packed");
        packed
            .as_mut_slice::<u8>()
            .expect("packed mut")
            .copy_from_slice(&packed_cpu);
        let mut norms = device
            .alloc_buffer(nelems_norms * 4, DType::F32, vec![nelems_norms])
            .expect("alloc norms");
        norms
            .as_mut_slice::<f32>()
            .expect("norms mut")
            .copy_from_slice(&norms_cpu);

        let dst_per_pos_size = (num_kv_heads * head_dim) as usize;
        let mut dst_per_pos = device
            .alloc_buffer(dst_per_pos_size * 4, DType::F32, vec![dst_per_pos_size])
            .expect("alloc dst_per_pos");
        let mut dst_seq = device
            .alloc_buffer(dst_per_pos_size * 4, DType::F32, vec![dst_per_pos_size])
            .expect("alloc dst_seq");

        for read_pos in 0..cap {
            for v in dst_per_pos.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = f32::NAN;
            }
            for v in dst_seq.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = f32::NAN;
            }

            for cb_bits in [5u32, 6, 8] {
                let mut enc = device.command_encoder().expect("enc");
                dispatch_tq_dequantize_hb_kv(
                    &mut enc,
                    &mut registry,
                    device.metal_device(),
                    &packed,
                    &norms,
                    &dst_per_pos,
                    num_kv_heads,
                    head_dim,
                    cap,
                    read_pos,
                    1.0,
                    cb_bits,
                )
                .expect("per-pos dispatch");
                dispatch_tq_dequantize_hb_kv_seq(
                    &mut enc,
                    &mut registry,
                    device.metal_device(),
                    &packed,
                    &norms,
                    &dst_seq,
                    num_kv_heads,
                    head_dim,
                    cap,
                    /*start_pos=*/ read_pos,
                    /*n_tokens=*/ 1,
                    1.0,
                    cb_bits,
                )
                .expect("seq dispatch");
                enc.commit_and_wait().expect("commit");

                let a = dst_per_pos.as_slice::<f32>().expect("a slice");
                let b = dst_seq.as_slice::<f32>().expect("b slice");
                assert_eq!(
                    a, b,
                    "read_pos={read_pos} cb_bits={cb_bits}: per-pos vs seq(n=1) mismatch"
                );
            }
        }
    }

    /// Iter-30 sanity: seq dispatcher with n_tokens > 1 produces the same
    /// per-position results, just batched. Confirms the threadgroup
    /// fan-out (one tg per (kv_head, position)) addresses each position
    /// independently with no cross-position aliasing.
    #[test]
    fn tq_dequantize_hb_kv_seq_n_gt_1_matches_concat_per_position() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut registry = KernelRegistry::new();

        let num_kv_heads: u32 = 2;
        let cap: u32 = 8;
        let head_dim: u32 = 256;
        let n_tokens: u32 = 4;
        let start_pos: u32 = 2;
        let cb_bits: u32 = 8;

        let nbytes_packed = (num_kv_heads * cap * head_dim) as usize;
        let mut packed_cpu = vec![0u8; nbytes_packed];
        for (i, b) in packed_cpu.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(53).wrapping_add(17);
        }
        let nelems_norms = (num_kv_heads * cap * 1) as usize;
        let mut norms_cpu = vec![0f32; nelems_norms];
        for (i, n) in norms_cpu.iter_mut().enumerate() {
            *n = 0.25 + (i as f32) * 0.125;
        }

        let mut packed = device
            .alloc_buffer(nbytes_packed, DType::U8, vec![nbytes_packed])
            .unwrap();
        packed
            .as_mut_slice::<u8>()
            .unwrap()
            .copy_from_slice(&packed_cpu);
        let mut norms = device
            .alloc_buffer(nelems_norms * 4, DType::F32, vec![nelems_norms])
            .unwrap();
        norms
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&norms_cpu);

        let seq_size = (num_kv_heads * n_tokens * head_dim) as usize;
        let mut dst_seq = device
            .alloc_buffer(seq_size * 4, DType::F32, vec![seq_size])
            .unwrap();
        for v in dst_seq.as_mut_slice::<f32>().unwrap().iter_mut() {
            *v = f32::NAN;
        }
        let mut enc = device.command_encoder().unwrap();
        dispatch_tq_dequantize_hb_kv_seq(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &packed,
            &norms,
            &dst_seq,
            num_kv_heads,
            head_dim,
            cap,
            start_pos,
            n_tokens,
            1.0,
            cb_bits,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
        let seq_slice = dst_seq.as_slice::<f32>().unwrap().to_vec();

        let pp_size = (num_kv_heads * head_dim) as usize;
        let mut dst_pp = device
            .alloc_buffer(pp_size * 4, DType::F32, vec![pp_size])
            .unwrap();

        for tok in 0..n_tokens {
            for v in dst_pp.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = f32::NAN;
            }
            let mut enc = device.command_encoder().unwrap();
            dispatch_tq_dequantize_hb_kv(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &packed,
                &norms,
                &dst_pp,
                num_kv_heads,
                head_dim,
                cap,
                start_pos + tok,
                1.0,
                cb_bits,
            )
            .unwrap();
            enc.commit_and_wait().unwrap();
            let pp_slice = dst_pp.as_slice::<f32>().unwrap();

            for head in 0..num_kv_heads {
                let pp_off = (head as usize) * head_dim as usize;
                let seq_off = (head as usize) * (n_tokens as usize) * (head_dim as usize)
                    + (tok as usize) * (head_dim as usize);
                let pp_h = &pp_slice[pp_off..pp_off + head_dim as usize];
                let seq_h = &seq_slice[seq_off..seq_off + head_dim as usize];
                assert_eq!(pp_h, seq_h, "tok={tok} head={head}: seq != per-pos");
            }
        }
    }

    /// The tiled Gemma resume path consumes F16 staging buffers directly.
    /// Prove that the F16 kernel differs from the established F32 kernel only
    /// by the documented final half-precision rounding, including D=512's
    /// two-norm convention and every supported higher-bit codebook.
    #[test]
    fn tq_dequantize_hb_kv_seq_f16_matches_f32_after_half_rounding() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut registry = KernelRegistry::new();

        let num_kv_heads = 2u32;
        let cap = 9u32;
        let start_pos = 2u32;
        let n_tokens = 5u32;

        for head_dim in [256u32, 512] {
            let norms_per_pos = (head_dim / 256).max(1);
            let packed_len = (num_kv_heads * cap * head_dim) as usize;
            let mut packed = device
                .alloc_buffer(packed_len, DType::U8, vec![packed_len])
                .expect("alloc packed");
            for (i, value) in packed.as_mut_slice::<u8>().unwrap().iter_mut().enumerate() {
                *value = (i as u8).wrapping_mul(73).wrapping_add(19);
            }

            let norms_len = (num_kv_heads * cap * norms_per_pos) as usize;
            let mut norms = device
                .alloc_buffer(norms_len * 4, DType::F32, vec![norms_len])
                .expect("alloc norms");
            for (i, value) in norms.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
                *value = 0.375 + (i as f32) * 0.03125;
            }

            let output_len = (num_kv_heads * n_tokens * head_dim) as usize;
            for codebook_bits in [5u32, 6, 8] {
                let dst_f32 = device
                    .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
                    .expect("alloc F32 output");
                let dst_f16 = device
                    .alloc_buffer(output_len * 2, DType::F16, vec![output_len])
                    .expect("alloc F16 output");

                let mut encoder = device.command_encoder().expect("encoder");
                dispatch_tq_dequantize_hb_kv_seq(
                    &mut encoder,
                    &mut registry,
                    device.metal_device(),
                    &packed,
                    &norms,
                    &dst_f32,
                    num_kv_heads,
                    head_dim,
                    cap,
                    start_pos,
                    n_tokens,
                    1.25,
                    codebook_bits,
                )
                .expect("F32 dispatch");
                dispatch_tq_dequantize_hb_kv_seq_f16(
                    &mut encoder,
                    &mut registry,
                    device.metal_device(),
                    &packed,
                    &norms,
                    &dst_f16,
                    num_kv_heads,
                    head_dim,
                    cap,
                    start_pos,
                    n_tokens,
                    1.25,
                    codebook_bits,
                )
                .expect("F16 dispatch");
                encoder.commit_and_wait().expect("commit");

                let reference = dst_f32.as_slice::<f32>().expect("F32 slice");
                let actual = dst_f16.as_slice::<half::f16>().expect("F16 slice");
                for (index, (&want_f32, &got_f16)) in
                    reference.iter().zip(actual.iter()).enumerate()
                {
                    let want_f16 = half::f16::from_f32(want_f32);
                    assert_eq!(
                        got_f16.to_bits(),
                        want_f16.to_bits(),
                        "head_dim={head_dim} codebook_bits={codebook_bits} index={index}"
                    );
                }
            }
        }
    }
}
