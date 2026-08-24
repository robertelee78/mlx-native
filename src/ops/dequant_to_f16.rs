// ADR-029 iter-28 H29 — whole-tensor dequant from block-quantized formats
// to F16, used at model load to materialize an F16 shadow of dense weights
// so the runtime dispatch can use `kernel_mul_mm_f16_f32_*` (peer's
// gemma4 pattern).
//
// See `src/shaders/dequant_to_f16.metal` for the kernel design and the
// per-type instantiation list.  Public Rust API: one entry point
// `dispatch_dequant_to_f16(...)` that picks the right kernel via the
// caller-supplied `GgmlType`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::encode_helpers::KernelArg;
use crate::ops::quantized_matmul_ggml::GgmlType;
use crate::DType;

/// Number of K-quants sub-groups per block_q5_K / block_q4_K / block_q6_K.
/// Matches `QK_NL = 16` in `dequant_to_f16.metal`.
const QK_NL_K: u32 = 16;

/// Number of legacy-block sub-groups per block_q4_0 / block_q8_0 / etc.
/// Matches `nl = 2` in `dequant_to_f16.metal`.
const QK_NL_LEGACY: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DequantToF16Config {
    block_values: u32,
    qk_nl: u32,
    kernel_name: &'static str,
}

fn dequant_to_f16_config(ggml_type: GgmlType) -> Result<DequantToF16Config> {
    let (block_values, qk_nl, kernel_name) = match ggml_type {
        GgmlType::Q4_0 => (32u32, QK_NL_LEGACY, "hf2q_dequant_q4_0_to_f16"),
        GgmlType::Q8_0 => (32, QK_NL_LEGACY, "hf2q_dequant_q8_0_to_f16"),
        GgmlType::Q5_1 => (32, QK_NL_LEGACY, "hf2q_dequant_q5_1_to_f16"),
        GgmlType::IQ4_NL => (32, QK_NL_LEGACY, "hf2q_dequant_iq4_nl_to_f16"),
        GgmlType::Q4_K => (256, QK_NL_K, "hf2q_dequant_q4_K_to_f16"),
        GgmlType::Q5_K => (256, QK_NL_K, "hf2q_dequant_q5_K_to_f16"),
        GgmlType::Q6_K => (256, QK_NL_K, "hf2q_dequant_q6_K_to_f16"),
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_dequant_to_f16: unsupported ggml_type {:?} \
                 (only Q4_0 / Q8_0 / Q5_1 / IQ4_NL / Q4_K / Q5_K / Q6_K)",
                other
            )));
        }
    };
    Ok(DequantToF16Config {
        block_values,
        qk_nl,
        kernel_name,
    })
}

/// Pure contract hook used by source-only tests to prove that an unsupported
/// codec cannot acquire an F16-shadow route through a wildcard dispatcher.
#[doc(hidden)]
pub fn test_only_dequant_to_f16_kernel_name(ggml_type: GgmlType) -> Result<&'static str> {
    Ok(dequant_to_f16_config(ggml_type)?.kernel_name)
}

/// Dispatch the whole-tensor dequant-to-F16 kernel.
///
/// `weight` is the source quantized buffer (caller-allocated, holds the
/// GGUF-format bytes for `n_rows × n_cols` elements of `ggml_type`).
/// `f16_shadow` is the destination buffer, must be at least
/// `n_rows * n_cols * 2` bytes (F16 = 2 bytes/elem).
///
/// `n_rows` / `n_cols` are the logical tensor shape; the kernel writes
/// `n_rows * n_cols` F16 values into `f16_shadow` in row-major order
/// (matching the row-major dequant layout the matmul kernels expect).
///
/// Returns InvalidArgument if `ggml_type` is unsupported (F32 / F16 /
/// I16 — no dequant needed) or if buffer sizes don't match.
pub fn dispatch_dequant_to_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight: &MlxBuffer,
    f16_shadow: &MlxBuffer,
    n_rows: u32,
    n_cols: u32,
    ggml_type: GgmlType,
) -> Result<()> {
    let config = dequant_to_f16_config(ggml_type)?;
    let block_values = config.block_values;

    // Validate shapes and buffer sizes.
    if n_rows == 0 || n_cols == 0 {
        return Err(MlxError::InvalidArgument(
            "dispatch_dequant_to_f16: n_rows and n_cols must be > 0".into(),
        ));
    }
    if n_cols % block_values != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_dequant_to_f16: n_cols ({}) must be divisible by block_values ({}) for {:?}",
            n_cols, block_values, ggml_type
        )));
    }
    if weight.dtype() != DType::U8 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_dequant_to_f16: weight must contain native U8 GGUF blocks, got {:?}",
            weight.dtype()
        )));
    }
    if f16_shadow.dtype() != DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_dequant_to_f16: f16_shadow must be DType::F16, got {:?}",
            f16_shadow.dtype()
        )));
    }

    let n_elements = (n_rows as u64) * (n_cols as u64);
    let needed_weight_bytes = n_elements
        .checked_div(u64::from(block_values))
        .and_then(|blocks| blocks.checked_mul(u64::from(ggml_type.block_bytes())))
        .ok_or_else(|| {
            MlxError::InvalidArgument("dispatch_dequant_to_f16: weight size overflow".into())
        })?;
    if (weight.data_byte_len() as u64) < needed_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_dequant_to_f16: weight too small ({} bytes; need {})",
            weight.data_byte_len(),
            needed_weight_bytes
        )));
    }
    let needed_bytes = n_elements * 2;
    if (f16_shadow.byte_len() as u64) < needed_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_dequant_to_f16: f16_shadow too small ({} bytes; need {})",
            f16_shadow.byte_len(),
            needed_bytes
        )));
    }

    // Total threads = n_elements / 16.  Each thread dequants one 16-elem
    // group (= one (block_idx, il) pair).
    let n_groups: u32 = (n_elements / 16) as u32;
    if (n_elements as u64) != (n_groups as u64) * 16 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_dequant_to_f16: total elements ({}) must be a multiple of 16 \
             (got n_rows={}, n_cols={})",
            n_elements, n_rows, n_cols
        )));
    }

    let pipeline = registry.get_pipeline(config.kernel_name, device)?;

    // Threadgroup size: pick 256 for good occupancy.  Total grid = n_groups
    // threadgroups of 256 threads, rounded up.
    const TG_SIZE: u64 = 256;
    let n_tg = ((n_groups as u64) + TG_SIZE - 1) / TG_SIZE;

    let threadgroups = MTLSize::new(n_tg, 1, 1);
    let threads_per_tg = MTLSize::new(TG_SIZE, 1, 1);

    let n_groups_bytes = n_groups.to_ne_bytes();

    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(&n_groups_bytes)),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(f16_shadow)),
        ],
        threadgroups,
        threads_per_tg,
    );

    // ABI sanity: silence dead-arg warning when block_values is only used
    // in the validation arm above.
    let _ = block_values;
    let _ = config.qk_nl;
    Ok(())
}

/// One-shot helper: allocate an F16 shadow buffer + dispatch + commit-and-wait.
///
/// Intended for use at model load — caller has the source quantized buffer
/// already on GPU and wants a paired F16 shadow.  Returns the new F16
/// buffer.  Performs a `commit_and_wait` so the buffer is ready for
/// downstream use by the time this function returns.
pub fn materialize_f16_shadow(
    device: &crate::MlxDevice,
    registry: &mut KernelRegistry,
    weight: &MlxBuffer,
    n_rows: u32,
    n_cols: u32,
    ggml_type: GgmlType,
) -> Result<MlxBuffer> {
    // Reject unsupported codecs before allocation. In particular, native
    // Q5_0 must never gain an F16 representation shadow as a side effect of
    // calling this convenience API.
    let config = dequant_to_f16_config(ggml_type)?;
    if n_rows == 0 || n_cols == 0 {
        return Err(MlxError::InvalidArgument(
            "materialize_f16_shadow: n_rows and n_cols must be > 0".into(),
        ));
    }
    if n_cols % config.block_values != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "materialize_f16_shadow: n_cols ({n_cols}) must be divisible by block values ({})",
            config.block_values
        )));
    }
    if weight.dtype() != DType::U8 {
        return Err(MlxError::InvalidArgument(format!(
            "materialize_f16_shadow: weight must contain native U8 GGUF blocks, got {:?}",
            weight.dtype()
        )));
    }
    let n_elements = (n_rows as usize)
        .checked_mul(n_cols as usize)
        .ok_or_else(|| MlxError::InvalidArgument("materialize_f16_shadow size overflow".into()))?;
    let shadow_bytes = n_elements
        .checked_mul(DType::F16.size_of())
        .ok_or_else(|| {
            MlxError::InvalidArgument("materialize_f16_shadow byte count overflow".into())
        })?;
    let f16_shadow = device
        .alloc_buffer(
            shadow_bytes,
            DType::F16,
            vec![n_rows as usize, n_cols as usize],
        )
        .map_err(|e| MlxError::InvalidArgument(format!("materialize_f16_shadow alloc: {e}")))?;

    let mut encoder = device
        .command_encoder()
        .map_err(|e| MlxError::InvalidArgument(format!("materialize_f16_shadow encoder: {e}")))?;

    dispatch_dequant_to_f16(
        &mut encoder,
        registry,
        device.metal_device(),
        weight,
        &f16_shadow,
        n_rows,
        n_cols,
        ggml_type,
    )?;

    encoder
        .commit_and_wait()
        .map_err(|e| MlxError::InvalidArgument(format!("materialize_f16_shadow commit: {e}")))?;

    Ok(f16_shadow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MlxDevice;

    /// Round-trip: build a tiny Q8_0 tensor on CPU, dequant it via the
    /// kernel, compare against a CPU-side dequant.  Confirms the kernel
    /// produces correct F16 output for at least one type.
    #[test]
    fn dequant_q8_0_to_f16_roundtrip() {
        // 1 block_q8_0 = 32 elements.  We use 2 blocks = 64 elements (1 row).
        const N_BLOCKS: usize = 2;
        const N_ELEMENTS: usize = N_BLOCKS * 32;

        let device = MlxDevice::new().expect("new device");

        // Build a block_q8_0 with d=0.5, qs = [0, 1, 2, ..., 31].
        // dequant_q8_0_t at il produces 16 elements: qs[16*il + i] * d
        // (i ∈ [0, 16))
        let block_bytes = 2 + 32; // half (2) + 32 int8
        let mut src: Vec<u8> = vec![0u8; N_BLOCKS * block_bytes];
        for b in 0..N_BLOCKS {
            // half(0.5) = 0x3800
            src[b * block_bytes + 0] = 0x00;
            src[b * block_bytes + 1] = 0x38;
            for i in 0..32 {
                src[b * block_bytes + 2 + i] = ((b * 32 + i) % 128) as u8;
            }
        }

        let mut weight = device
            .alloc_buffer(src.len(), DType::U8, vec![src.len()])
            .expect("alloc src");
        weight
            .as_mut_slice::<u8>()
            .expect("slice src")
            .copy_from_slice(&src);

        let f16_shadow = device
            .alloc_buffer(N_ELEMENTS * 2, DType::F16, vec![N_ELEMENTS])
            .expect("alloc f16");

        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        // Note: this test uses raw row dimensions; the kernel only cares about
        // total elements = n_rows * n_cols.  We test as a 1×64 tensor.
        let res = dispatch_dequant_to_f16(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &weight,
            &f16_shadow,
            1,
            N_ELEMENTS as u32,
            GgmlType::Q8_0,
        );
        // dispatch_dequant_to_f16 returns Ok if the kernel was queued —
        // actual GPU execution happens at commit.
        res.expect("dispatch ok");

        encoder.commit_and_wait().expect("commit");

        // Read back and confirm element 0 = d * qs[0] = 0.5 * 0 = 0.
        let out: &[u16] = f16_shadow.as_slice().expect("read f16");
        // F16 0.0 = 0x0000.
        assert_eq!(
            out[0], 0x0000,
            "out[0] should be F16 0.0, got 0x{:04X}",
            out[0]
        );
        // F16(0.5 * 1) = F16(0.5) = 0x3800.
        assert_eq!(
            out[1], 0x3800,
            "out[1] should be F16 0.5, got 0x{:04X}",
            out[1]
        );
        // F16(0.5 * 2) = F16(1.0) = 0x3C00.
        assert_eq!(
            out[2], 0x3C00,
            "out[2] should be F16 1.0, got 0x{:04X}",
            out[2]
        );
    }
}
