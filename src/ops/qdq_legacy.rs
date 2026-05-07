//! GGUF-legacy quantize-dequantize round-trip primitives (Q4_0, Q8_0).
//!
//! Used by hf2q's ADR-020 Track 1 dynamic_quant port — produces the
//! `W_low` and `W_high` weight tensors for the gradient-Taylor
//! sensitivity formula `Σ grad · (W_low − W_high)`.
//!
//! For each input fp32 tensor, applies the Q4_0 (or Q8_0) GGUF
//! quantize → dequantize round-trip BLOCK-WISE and writes the
//! rounded fp32 values back.  Output is byte-identical to
//! `quantize_row_q{4,8}_0 → dequantize_row_q{4,8}_0` from
//! `hf2q::quantize::q_legacy`.
//!
//! Element count must be divisible by 32 (the GGUF block size for
//! both Q4_0 and Q8_0).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static QDQ_LEGACY_SHADER_SOURCE: &str = include_str!("../shaders/qdq_legacy.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("qdq_q4_0_f32", QDQ_LEGACY_SHADER_SOURCE);
    registry.register_source("qdq_q8_0_f32", QDQ_LEGACY_SHADER_SOURCE);
}

/// GGUF block size for Q4_0 / Q8_0 (both use 32-element blocks).
pub const QDQ_BLOCK_SIZE: u32 = 32;

/// Encode the Q4_0 quantize-dequantize round-trip kernel.
///
/// `input.element_count()` must be divisible by [`QDQ_BLOCK_SIZE`] (32).
/// `output` must have the same element count and dtype (f32).
pub fn dispatch_qdq_q4_0_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    validate(input, output, "qdq_q4_0_f32")?;
    let n = input.element_count() as u64;
    let num_blocks = n / u64::from(QDQ_BLOCK_SIZE);

    let pipeline = registry.get_pipeline("qdq_q4_0_f32", device)?;

    // Threadgroup shared memory: 32 amax floats + 32 max floats = 64 * 4 = 256 bytes.
    let shared_mem_bytes: u64 = 64 * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, output)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(num_blocks, 1, 1),
        MTLSize::new(u64::from(QDQ_BLOCK_SIZE), 1, 1),
    );

    Ok(())
}

/// Encode the Q8_0 quantize-dequantize round-trip kernel.
pub fn dispatch_qdq_q8_0_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    validate(input, output, "qdq_q8_0_f32")?;
    let n = input.element_count() as u64;
    let num_blocks = n / u64::from(QDQ_BLOCK_SIZE);

    let pipeline = registry.get_pipeline("qdq_q8_0_f32", device)?;

    // Threadgroup shared memory: 32 amax floats = 128 bytes.
    let shared_mem_bytes: u64 = 32 * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, output)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(num_blocks, 1, 1),
        MTLSize::new(u64::from(QDQ_BLOCK_SIZE), 1, 1),
    );

    Ok(())
}

fn validate(input: &MlxBuffer, output: &MlxBuffer, op_name: &str) -> Result<()> {
    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op_name}: input must have at least one element"
        )));
    }
    if n % (QDQ_BLOCK_SIZE as usize) != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op_name}: input element count ({n}) must be divisible by block size ({})",
            QDQ_BLOCK_SIZE
        )));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "{op_name}: output element count {} != input element count {}",
            output.element_count(),
            n
        )));
    }
    if input.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{op_name}: only f32 supported; got input={} output={}",
            input.dtype(),
            output.dtype()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    /// CPU oracle re-implementing `quantize_row_q4_0 + dequantize_row_q4_0`
    /// from hf2q's q_legacy.rs.  Mirrored here because mlx-native is
    /// upstream of hf2q and cannot depend on it; downstream hf2q tests
    /// will re-validate against the canonical q_legacy implementation.
    fn qdq_q4_0_cpu_oracle(input: &[f32]) -> Vec<f32> {
        const QK: usize = 32;
        assert!(input.len() % QK == 0);
        let mut out = vec![0f32; input.len()];
        for blk_i in 0..(input.len() / QK) {
            let block = &input[blk_i * QK..(blk_i + 1) * QK];
            // Find signed value at position with largest |.| — `>` so ties pick LEFT.
            let mut amax = 0.0f32;
            let mut max = 0.0f32;
            for &v in block {
                let av = v.abs();
                if av > amax {
                    amax = av;
                    max = v;
                }
            }
            let d = max / -8.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            // f16 round-trip — dequant reads d as f16 (q_legacy.rs:358 + d() at decode).
            let d_h = half::f16::from_f32(d).to_f32();
            for (j, &v) in block.iter().enumerate() {
                // CPU uses `((v * id + 8.5) as i32).clamp(0, 15)`.
                // `as i32` on f32 is truncate-toward-zero; for the
                // intended domain (v * id + 8.5 ∈ [0, 16) by construction
                // when v ∈ [-8d, 8d]) this is equivalent to floor.
                let scaled = v * id + 8.5;
                let q = (scaled as i32).clamp(0, 15);
                out[blk_i * QK + j] = (q - 8) as f32 * d_h;
            }
        }
        out
    }

    fn qdq_q8_0_cpu_oracle(input: &[f32]) -> Vec<f32> {
        const QK: usize = 32;
        assert!(input.len() % QK == 0);
        let mut out = vec![0f32; input.len()];
        for blk_i in 0..(input.len() / QK) {
            let block = &input[blk_i * QK..(blk_i + 1) * QK];
            let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            let d_h = half::f16::from_f32(d).to_f32();
            for (j, &v) in block.iter().enumerate() {
                // CPU uses `(v * id).round() as i32` then clamp(-128, 127).
                // `f32::round` is half-away-from-zero.
                let q = ((v * id).round() as i32).clamp(-128, 127);
                out[blk_i * QK + j] = (q as f32) * d_h;
            }
        }
        out
    }

    /// Run the GPU dispatch on a single host slice and read back the result.
    /// Uses `device.alloc_buffer + as_mut_slice + as_slice` directly (no
    /// GraphExecutor needed for a single-kernel unit test).
    fn run_qdq_q4_0(input: &[f32]) -> Vec<f32> {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let n_bytes = input.len() * std::mem::size_of::<f32>();
        let mut in_buf = device
            .alloc_buffer(n_bytes, DType::F32, vec![input.len()])
            .expect("alloc input");
        let out_buf = device
            .alloc_buffer(n_bytes, DType::F32, vec![input.len()])
            .expect("alloc output");
        // Populate input
        {
            let slice: &mut [f32] = in_buf.as_mut_slice().expect("input as_mut_slice");
            slice.copy_from_slice(input);
        }
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("command_encoder");
        dispatch_qdq_q4_0_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
        )
        .expect("dispatch_qdq_q4_0_f32");
        encoder.commit_and_wait().expect("commit_and_wait");
        out_buf.as_slice::<f32>().expect("output as_slice").to_vec()
    }

    fn run_qdq_q8_0(input: &[f32]) -> Vec<f32> {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let n_bytes = input.len() * std::mem::size_of::<f32>();
        let mut in_buf = device
            .alloc_buffer(n_bytes, DType::F32, vec![input.len()])
            .expect("alloc input");
        let out_buf = device
            .alloc_buffer(n_bytes, DType::F32, vec![input.len()])
            .expect("alloc output");
        {
            let slice: &mut [f32] = in_buf.as_mut_slice().expect("input as_mut_slice");
            slice.copy_from_slice(input);
        }
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("command_encoder");
        dispatch_qdq_q8_0_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
        )
        .expect("dispatch_qdq_q8_0_f32");
        encoder.commit_and_wait().expect("commit_and_wait");
        out_buf.as_slice::<f32>().expect("output as_slice").to_vec()
    }

    fn assert_byte_identical(label: &str, gpu: &[f32], cpu: &[f32]) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            // q→dq is deterministic — bit-equality must hold.
            if g.to_bits() != c.to_bits() {
                panic!(
                    "{label}: bit-mismatch at index {i}: gpu={} (0x{:08x}) cpu={} (0x{:08x})",
                    g,
                    g.to_bits(),
                    c,
                    c.to_bits()
                );
            }
        }
    }

    #[test]
    fn qdq_q4_0_byte_identical_one_block() {
        // Linear ramp through the representable range — exercises the
        // amax-tie boundary, exact-halfway rounding cases, and zero.
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.137).collect();
        let gpu = run_qdq_q4_0(&input);
        let cpu = qdq_q4_0_cpu_oracle(&input);
        assert_byte_identical("q4_0 single-block ramp", &gpu, &cpu);
    }

    #[test]
    fn qdq_q4_0_byte_identical_random_multi_block() {
        // 8 blocks of 32 = 256 elements with varied magnitudes per block,
        // including a mostly-zero block and a block with a single outlier.
        let mut input = Vec::with_capacity(256);
        for blk in 0..8 {
            let scale = (blk as f32 + 1.0) * 0.5;
            for i in 0..32 {
                let v = ((i as f32 * 17.0 + blk as f32 * 31.0).sin()) * scale;
                input.push(v);
            }
        }
        let gpu = run_qdq_q4_0(&input);
        let cpu = qdq_q4_0_cpu_oracle(&input);
        assert_byte_identical("q4_0 multi-block sin", &gpu, &cpu);
    }

    #[test]
    fn qdq_q4_0_zero_block() {
        let input = vec![0.0_f32; 32];
        let gpu = run_qdq_q4_0(&input);
        let cpu = qdq_q4_0_cpu_oracle(&input);
        assert_byte_identical("q4_0 all-zero block", &gpu, &cpu);
        assert!(gpu.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn qdq_q4_0_single_outlier_block() {
        // 31 zeros + 1 large value: amax = the outlier, max = signed outlier,
        // d = outlier / -8, all 31 zeros quantize to q=8 (the zero point).
        let mut input = vec![0.0_f32; 32];
        input[7] = 5.25;
        let gpu = run_qdq_q4_0(&input);
        let cpu = qdq_q4_0_cpu_oracle(&input);
        assert_byte_identical("q4_0 single outlier", &gpu, &cpu);
    }

    #[test]
    fn qdq_q4_0_negative_amax_block() {
        // The amax-bearing element is negative — verifies sign tracking
        // through the tree reduction (the LEFT-tie semantics of `>`).
        let mut input: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.05).collect();
        input[3] = -2.0; // make this the unique amax
        let gpu = run_qdq_q4_0(&input);
        let cpu = qdq_q4_0_cpu_oracle(&input);
        assert_byte_identical("q4_0 negative amax", &gpu, &cpu);
    }

    #[test]
    fn qdq_q8_0_byte_identical_one_block() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.137).collect();
        let gpu = run_qdq_q8_0(&input);
        let cpu = qdq_q8_0_cpu_oracle(&input);
        assert_byte_identical("q8_0 single-block ramp", &gpu, &cpu);
    }

    #[test]
    fn qdq_q8_0_byte_identical_random_multi_block() {
        let mut input = Vec::with_capacity(256);
        for blk in 0..8 {
            let scale = (blk as f32 + 1.0) * 0.5;
            for i in 0..32 {
                let v = ((i as f32 * 17.0 + blk as f32 * 31.0).sin()) * scale;
                input.push(v);
            }
        }
        let gpu = run_qdq_q8_0(&input);
        let cpu = qdq_q8_0_cpu_oracle(&input);
        assert_byte_identical("q8_0 multi-block sin", &gpu, &cpu);
    }

    #[test]
    fn qdq_q8_0_zero_block() {
        let input = vec![0.0_f32; 32];
        let gpu = run_qdq_q8_0(&input);
        let cpu = qdq_q8_0_cpu_oracle(&input);
        assert_byte_identical("q8_0 all-zero block", &gpu, &cpu);
        assert!(gpu.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn qdq_q8_0_signed_extremes() {
        // Pin the half-away-from-zero rounding contract: for q at exactly
        // ±0.5, round AWAY from zero (per Rust's f32::round).
        let mut input = vec![0.0_f32; 32];
        // Crafted so v * id is exactly ±0.5 at j=1: amax = 1.0, d = 1/127,
        // so v[1] = 0.5/127 should round to q=1 (not 0).
        input[0] = 1.0;
        input[1] = 0.5 / 127.0;
        input[2] = -0.5 / 127.0;
        let gpu = run_qdq_q8_0(&input);
        let cpu = qdq_q8_0_cpu_oracle(&input);
        assert_byte_identical("q8_0 signed extremes", &gpu, &cpu);
    }

    #[test]
    fn input_size_must_be_block_aligned() {
        let device = MlxDevice::new().expect("MlxDevice");
        // 33 elements → not block-aligned.
        let in_buf = device
            .alloc_buffer(33 * 4, DType::F32, vec![33])
            .expect("alloc");
        let out_buf = device
            .alloc_buffer(33 * 4, DType::F32, vec![33])
            .expect("alloc");
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("command_encoder");
        let err = dispatch_qdq_q4_0_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
        )
        .expect_err("must reject non-block-aligned input");
        let msg = format!("{err}");
        assert!(
            msg.contains("divisible by block size"),
            "wrong error message: {msg}"
        );
    }

    #[test]
    fn output_size_must_match_input() {
        let device = MlxDevice::new().expect("MlxDevice");
        let in_buf = device
            .alloc_buffer(32 * 4, DType::F32, vec![32])
            .expect("alloc input");
        let out_buf = device
            .alloc_buffer(64 * 4, DType::F32, vec![64])
            .expect("alloc output");
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("command_encoder");
        let err = dispatch_qdq_q4_0_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
        )
        .expect_err("must reject mismatched output size");
        let msg = format!("{err}");
        assert!(
            msg.contains("output element count"),
            "wrong error message: {msg}"
        );
    }
}
