//! ADR-022 Phase 1 P1.7 — `mul_mv_ext` r1 family for Q5_1 + IQ4_NL.
//!
//! Wraps the eight Metal kernels in `shaders/mul_mv_ext.metal`:
//!
//!   `kernel_mul_mv_ext_<q>_f32_r1_<r1>` for q ∈ {q5_1, iq4_nl},
//!   r1 ∈ {2, 3, 4, 5}.
//!
//! The host dispatcher mirrors llama.cpp's
//! `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2080-2152`:
//!   - `nsg = 2` (constant)
//!   - `nxpsg = 16` if `K % 256 == 0 && M < 3`,
//!     else `8` if `K % 128 == 0`,
//!     else `4`
//!   - `r1ptg` selected by m: m=2→2, m∈{3,6}→3, m∈{4,7,8}→4, m=5→5
//!   - threadgroups = (N/r0ptg, M/r1ptg, batch)
//!   - threads-per-tg = (32, nsg, 1)
//!
//! Falls within the public dispatcher's m=2..8 batch range. ADR-022 P1.7
//! ports it for Q5_1 + IQ4_NL only; Phase 4 will extend the family across
//! Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::GgmlType;

/// Host-side parameters for [`mul_mv_ext_dispatch`].
///
/// Buffer layout contract:
///   - `weight` (= src0): row-major `[N, blocks_per_row]` GGUF blocks.
///   - `input`  (= src1): row-major `[batch, M, K]` f32. K = ne00.
///   - `output` (= dst):  row-major `[batch, M, N]` f32. N = ne01 / ne0.
///
/// `r2`, `r3` model llama.cpp's batch-broadcast (default 1, 1).
#[derive(Debug, Clone, Copy)]
pub struct MulMvExtParams {
    /// M — number of src1 rows (small batch, must be ∈ [2, 8]).
    pub m: u32,
    /// N — number of weight rows (output dim).
    pub n: u32,
    /// K — contract dim (input dim, must be divisible by 32).
    pub k: u32,
    /// Batch-broadcast factor for src0 vs src1 (typical 1).
    pub batch: u32,
    /// GGUF weight type. Phase 1 supports Q5_1 + IQ4_NL only; other types
    /// return `MlxError::InvalidArgument`.
    pub ggml_type: GgmlType,
}

/// GPU args struct — must match `hf2q_mul_mv_ext_args` in
/// `shaders/mul_mv_ext.metal` byte-for-byte.
///
/// llama.cpp's C layout puts an int32 triple before u64 fields, then more
/// int32 + u64, ending with two i16. The Metal-side struct's natural
/// alignment matches this with padding inserted after `ne02` (4-byte pad
/// before nb00 to reach 8-byte alignment). We model that explicitly so
/// `bytemuck::Pod` is happy and the byte layout matches the GPU struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MulMvExtGpuArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    _pad0: u32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    _pad1: u32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    // Trailing pad: largest member u64 (align 8) → struct size must be a
    // multiple of 8. Without this, bytemuck::Pod's no-padding contract
    // refuses the type. Match the implicit C tail padding the Metal
    // compiler emits for the MSL struct.
    _pad2: u32,
}

/// Pick `nxpsg` per llama.cpp's `ggml-metal-ops.cpp:2094-2100`.
fn pick_nxpsg(k: u32, m: u32) -> i32 {
    if k % 256 == 0 && m < 3 {
        16
    } else if k % 128 == 0 {
        8
    } else {
        4
    }
}

/// Pick `r1ptg` per llama.cpp's `ggml-metal-ops.cpp:2107-2120` switch.
/// Returns `Err(InvalidArgument)` for unsupported m values.
fn pick_r1ptg(m: u32) -> Result<i32> {
    match m {
        2 => Ok(2),
        3 | 6 => Ok(3),
        4 | 7 | 8 => Ok(4),
        5 => Ok(5),
        other => Err(MlxError::InvalidArgument(format!(
            "mul_mv_ext: unsupported m {} (peer mapping covers 2..=8 only)",
            other
        ))),
    }
}

/// Compose the kernel name from ggml_type + r1ptg, matching the metal
/// shader's `[[host_name(...)]]` attributes.
///
/// Phase 1 (P1.7): Q5_1 + IQ4_NL × r1∈{2,3,4,5}.
/// Phase 4: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K × r1∈{2,3,4,5}.
fn kernel_name(ggml_type: GgmlType, r1ptg: i32) -> Result<&'static str> {
    Ok(match (ggml_type, r1ptg) {
        (GgmlType::Q5_1, 2) => "kernel_mul_mv_ext_q5_1_f32_r1_2",
        (GgmlType::Q5_1, 3) => "kernel_mul_mv_ext_q5_1_f32_r1_3",
        (GgmlType::Q5_1, 4) => "kernel_mul_mv_ext_q5_1_f32_r1_4",
        (GgmlType::Q5_1, 5) => "kernel_mul_mv_ext_q5_1_f32_r1_5",
        (GgmlType::IQ4_NL, 2) => "kernel_mul_mv_ext_iq4_nl_f32_r1_2",
        (GgmlType::IQ4_NL, 3) => "kernel_mul_mv_ext_iq4_nl_f32_r1_3",
        (GgmlType::IQ4_NL, 4) => "kernel_mul_mv_ext_iq4_nl_f32_r1_4",
        (GgmlType::IQ4_NL, 5) => "kernel_mul_mv_ext_iq4_nl_f32_r1_5",
        (GgmlType::Q4_0, 2) => "kernel_mul_mv_ext_q4_0_f32_r1_2",
        (GgmlType::Q4_0, 3) => "kernel_mul_mv_ext_q4_0_f32_r1_3",
        (GgmlType::Q4_0, 4) => "kernel_mul_mv_ext_q4_0_f32_r1_4",
        (GgmlType::Q4_0, 5) => "kernel_mul_mv_ext_q4_0_f32_r1_5",
        (GgmlType::Q8_0, 2) => "kernel_mul_mv_ext_q8_0_f32_r1_2",
        (GgmlType::Q8_0, 3) => "kernel_mul_mv_ext_q8_0_f32_r1_3",
        (GgmlType::Q8_0, 4) => "kernel_mul_mv_ext_q8_0_f32_r1_4",
        (GgmlType::Q8_0, 5) => "kernel_mul_mv_ext_q8_0_f32_r1_5",
        (GgmlType::Q4_K, 2) => "kernel_mul_mv_ext_q4_K_f32_r1_2",
        (GgmlType::Q4_K, 3) => "kernel_mul_mv_ext_q4_K_f32_r1_3",
        (GgmlType::Q4_K, 4) => "kernel_mul_mv_ext_q4_K_f32_r1_4",
        (GgmlType::Q4_K, 5) => "kernel_mul_mv_ext_q4_K_f32_r1_5",
        (GgmlType::Q5_K, 2) => "kernel_mul_mv_ext_q5_K_f32_r1_2",
        (GgmlType::Q5_K, 3) => "kernel_mul_mv_ext_q5_K_f32_r1_3",
        (GgmlType::Q5_K, 4) => "kernel_mul_mv_ext_q5_K_f32_r1_4",
        (GgmlType::Q5_K, 5) => "kernel_mul_mv_ext_q5_K_f32_r1_5",
        (GgmlType::Q6_K, 2) => "kernel_mul_mv_ext_q6_K_f32_r1_2",
        (GgmlType::Q6_K, 3) => "kernel_mul_mv_ext_q6_K_f32_r1_3",
        (GgmlType::Q6_K, 4) => "kernel_mul_mv_ext_q6_K_f32_r1_4",
        (GgmlType::Q6_K, 5) => "kernel_mul_mv_ext_q6_K_f32_r1_5",
        (other_type, other_r1) => {
            return Err(MlxError::InvalidArgument(format!(
                "mul_mv_ext: no kernel for type {:?} × r1ptg {} (Phase 1+4 ports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q5_1/IQ4_NL × r1∈{{2,3,4,5}})",
                other_type, other_r1
            )));
        }
    })
}

/// Encode a `mul_mv_ext` dispatch.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
///   - `ggml_type` is not Q5_1 or IQ4_NL,
///   - `m` is outside [2, 8],
///   - `k` is not divisible by 32,
///   - any of `m`, `n`, `k`, `batch` is zero,
///   - any buffer is too small.
pub fn mul_mv_ext_dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &MulMvExtParams,
) -> Result<()> {
    if params.m == 0 || params.n == 0 || params.k == 0 || params.batch == 0 {
        return Err(MlxError::InvalidArgument(
            "mul_mv_ext: m, n, k, batch must all be > 0".into(),
        ));
    }
    // K must be divisible by the block size of the weight type.
    // Legacy 32-element types (Q4_0/Q8_0/Q5_1/IQ4_NL): k % 32 == 0.
    // K-quants (Q4_K/Q5_K/Q6_K): k % 256 == 0.
    let block_qk = params.ggml_type.block_values();
    if params.k % block_qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "mul_mv_ext: k ({}) must be divisible by block QK ({}) for {:?}",
            params.k, block_qk, params.ggml_type
        )));
    }

    let r1ptg = pick_r1ptg(params.m)?;
    let nxpsg = pick_nxpsg(params.k, params.m);
    let nsg: i32 = 2;
    let nypsg = 32 / nxpsg;
    let r0ptg = nypsg * nsg;

    let kname = kernel_name(params.ggml_type, r1ptg)?;

    // PSO compile keyed on (kname, FC_mul_mv_nsg, FC_mul_mv_nxpsg).
    let pipeline = registry
        .get_pipeline_with_constants(
            kname,
            device.metal_device(),
            &[],
            &[(600, nsg), (601, nxpsg)],
        )?
        .clone();

    // Buffer-size validation. Use the block-aware formula so K-quants
    // (256-element blocks) work alongside legacy 32-element types.
    let block_bytes_per_row =
        (params.k as usize / block_qk as usize) * (params.ggml_type.block_bytes() as usize);
    let weight_required = (params.n as usize) * block_bytes_per_row;
    if weight.byte_len() < weight_required {
        return Err(MlxError::InvalidArgument(format!(
            "mul_mv_ext: weight buffer too small: {} < {} bytes",
            weight.byte_len(),
            weight_required
        )));
    }
    let input_required = (params.batch as usize)
        * (params.m as usize)
        * (params.k as usize)
        * DType::F32.size_of();
    if input.byte_len() < input_required {
        return Err(MlxError::InvalidArgument(format!(
            "mul_mv_ext: input buffer too small: {} < {} bytes",
            input.byte_len(),
            input_required
        )));
    }
    let output_required = (params.batch as usize)
        * (params.m as usize)
        * (params.n as usize)
        * DType::F32.size_of();
    if output.byte_len() < output_required {
        return Err(MlxError::InvalidArgument(format!(
            "mul_mv_ext: output buffer too small: {} < {} bytes",
            output.byte_len(),
            output_required
        )));
    }

    // GPU args. nb01 = bytes per weight row; nb00 = block_bytes (1 block per
    // QK4_0=32 elements). The args mirror llama.cpp's:
    //   nb00 = ggml_type_size(weight)            (single block)
    //   nb01 = ggml_row_size(weight, K)           (full row)
    //   nb02 = nb01 * N                           (single batch)
    //   nb10 = sizeof(float) = 4
    //   nb11 = K * sizeof(float)                  (single src1 row)
    //   nb12 = nb11 * M                           (single src1 batch)
    let nb00 = params.ggml_type.block_bytes() as u64;
    let nb01 = block_bytes_per_row as u64;
    let nb02 = nb01 * params.n as u64;
    let nb10: u64 = 4;
    let nb11 = (params.k as u64) * 4;
    let nb12 = nb11 * params.m as u64;
    let args = MulMvExtGpuArgs {
        ne00: params.k as i32,
        ne01: params.n as i32,
        ne02: 1,
        _pad0: 0,
        nb00,
        nb01,
        nb02,
        nb03: nb02, // unused
        ne10: params.k as i32,
        ne11: params.m as i32,
        ne12: params.batch as i32,
        _pad1: 0,
        nb10,
        nb11,
        nb12,
        nb13: nb12, // unused
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        _pad2: 0,
    };

    use crate::encoder::{as_bytes, KernelArg};

    let args_bytes = as_bytes(&args);
    let r0_groups = ((params.n as i32) + r0ptg - 1) / r0ptg;
    let r1_groups = ((params.m as i32) + r1ptg - 1) / r1ptg;

    encoder.encode_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Bytes(args_bytes)),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
        ],
        crate::MTLSize::new(r0_groups as u64, r1_groups as u64, params.batch as u64),
        crate::MTLSize::new(32, nsg as u64, 1),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_nxpsg_matches_peer_logic() {
        // K=128, M=1 → not %256 / yes %128 → 8
        assert_eq!(pick_nxpsg(128, 1), 8);
        // K=256, M=2 → yes %256 + M<3 → 16
        assert_eq!(pick_nxpsg(256, 2), 16);
        // K=256, M=3 → yes %256 + M>=3 → 8
        assert_eq!(pick_nxpsg(256, 3), 8);
        // K=64, M=4 → not %256 / not %128 → 4
        assert_eq!(pick_nxpsg(64, 4), 4);
        // K=2816, M=2 → 2816%256==0 (11×256) AND M<3 → 16
        assert_eq!(pick_nxpsg(2816, 2), 16);
        // K=2816, M=3 → 2816%256==0 but M>=3 → fallthrough %128==0 → 8
        assert_eq!(pick_nxpsg(2816, 3), 8);
        // K=512, M=4 → 512%256==0 but M>=3 → fallthrough %128==0 → 8
        assert_eq!(pick_nxpsg(512, 4), 8);
    }

    #[test]
    fn pick_r1ptg_matches_peer_switch() {
        assert_eq!(pick_r1ptg(2).unwrap(), 2);
        assert_eq!(pick_r1ptg(3).unwrap(), 3);
        assert_eq!(pick_r1ptg(4).unwrap(), 4);
        assert_eq!(pick_r1ptg(5).unwrap(), 5);
        assert_eq!(pick_r1ptg(6).unwrap(), 3);
        assert_eq!(pick_r1ptg(7).unwrap(), 4);
        assert_eq!(pick_r1ptg(8).unwrap(), 4);
        assert!(pick_r1ptg(1).is_err());
        assert!(pick_r1ptg(9).is_err());
    }

    #[test]
    fn kernel_name_covers_all_phase1_combinations() {
        for r1 in 2..=5 {
            assert!(kernel_name(GgmlType::Q5_1, r1).is_ok());
            assert!(kernel_name(GgmlType::IQ4_NL, r1).is_ok());
        }
        // Note: Q4_0 was Phase 1's "no kernel" canary — but Phase 4
        // (commit `9ee8a28`) added Q4_0/Q8_0/Q4_K/Q5_K/Q6_K coverage,
        // so Q4_0 is now a hit, not a miss. See sibling test
        // `kernel_name_covers_all_phase4_combinations` for the Phase 4
        // coverage assertion + `kernel_name_rejects_unsupported_types`
        // for the new "no kernel" canary.
    }

    #[test]
    fn kernel_name_covers_all_phase4_combinations() {
        // Phase 4 (commit `9ee8a28`): Q4_0, Q8_0, Q4_K, Q5_K, Q6_K
        // × r1 ∈ {2, 3, 4, 5}. Pin coverage so future GgmlType
        // additions don't silently drop Phase 4 wires.
        for r1 in 2..=5 {
            assert!(kernel_name(GgmlType::Q4_0, r1).is_ok(),
                "Phase 4 Q4_0 r1={r1} must have a kernel");
            assert!(kernel_name(GgmlType::Q8_0, r1).is_ok(),
                "Phase 4 Q8_0 r1={r1} must have a kernel");
            assert!(kernel_name(GgmlType::Q4_K, r1).is_ok(),
                "Phase 4 Q4_K r1={r1} must have a kernel");
            assert!(kernel_name(GgmlType::Q5_K, r1).is_ok(),
                "Phase 4 Q5_K r1={r1} must have a kernel");
            assert!(kernel_name(GgmlType::Q6_K, r1).is_ok(),
                "Phase 4 Q6_K r1={r1} must have a kernel");
        }
    }

    #[test]
    fn kernel_name_rejects_unsupported_combinations() {
        // r1 outside [2, 5] is rejected for ALL types (including
        // Phase 1 + Phase 4 covered ones).
        assert!(kernel_name(GgmlType::Q5_1, 1).is_err(),
            "r1=1 not supported by any phase");
        assert!(kernel_name(GgmlType::Q5_1, 6).is_err(),
            "r1=6 not supported by any phase");
        assert!(kernel_name(GgmlType::Q4_0, 0).is_err(),
            "r1=0 not supported");
        assert!(kernel_name(GgmlType::Q4_0, -1).is_err(),
            "r1=-1 not supported");
    }
}
