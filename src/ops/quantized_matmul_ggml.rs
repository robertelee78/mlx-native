//! GGML block-format quantized matrix-vector multiply dispatch.
//!
//! Encodes GPU compute commands for GGML quantized mat-vec:
//!   output[row] = dot(dequant(weight[row]), input)
//!
//! Weight buffers contain raw GGML blocks — the same bytes that come from
//! GGUF mmap. No intermediate conversion.
//!
//! Supported formats: Q4_0 (4-bit), Q8_0 (8-bit), Q6_K (6-bit super-block).
//!
//! Portions derived from candle-metal-kernels v0.10.2 (Apache-2.0) and
//! llama.cpp (MIT). See src/shaders/quantized_matmul_ggml.metal for full
//! attribution.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

// ---- Block format constants ----

/// Q4_0: 32 values per block, 18 bytes per block (2 byte f16 scale + 16 bytes quants).
const QK4_0: u32 = 32;
const BLOCK_Q4_0_BYTES: u32 = 18;

/// Q8_0: 32 values per block, 34 bytes per block (2 byte f16 scale + 32 bytes quants).
const QK8_0: u32 = 32;
const BLOCK_Q8_0_BYTES: u32 = 34;

/// Q4_K: 256 values per block, 144 bytes per block.
const QK4_K: u32 = 256;
const BLOCK_Q4_K_BYTES: u32 = 144;

/// Q6_K: 256 values per block, 210 bytes per block.
const QK6_K: u32 = 256;
const BLOCK_Q6_K_BYTES: u32 = 210;

// ---- Public types ----

/// GGML quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    /// 32-bit float (unquantized). 1 element per block, 4 bytes per block.
    F32,
    /// 16-bit float (unquantized). 1 element per block, 2 bytes per block.
    F16,
    /// 4-bit quantization. 32 values per block, 18 bytes per block.
    Q4_0,
    /// 8-bit quantization. 32 values per block, 34 bytes per block.
    Q8_0,
    /// 4-bit super-block quantization. 256 values per block, 144 bytes per block.
    Q4_K,
    /// 6-bit super-block quantization. 256 values per block, 210 bytes per block.
    Q6_K,
}

impl GgmlType {
    /// Number of dequantized values per GGML block.
    pub fn block_values(self) -> u32 {
        match self {
            GgmlType::F32 => 1,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => QK4_0,
            GgmlType::Q8_0 => QK8_0,
            GgmlType::Q4_K => QK4_K,
            GgmlType::Q6_K => QK6_K,
        }
    }

    /// Number of bytes per GGML block.
    pub fn block_bytes(self) -> u32 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => BLOCK_Q4_0_BYTES,
            GgmlType::Q8_0 => BLOCK_Q8_0_BYTES,
            GgmlType::Q4_K => BLOCK_Q4_K_BYTES,
            GgmlType::Q6_K => BLOCK_Q6_K_BYTES,
        }
    }

    /// Metal kernel function name for the matrix-vector (mv) kernel
    /// — used for `m <= MM_ROUTING_THRESHOLD`.
    fn kernel_name(self) -> &'static str {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => {
                // These types do not have a direct mat-vec kernel in this module.
                // Q4_K support for mat-vec will be added separately.
                "unsupported"
            }
            GgmlType::Q4_0 => "kernel_mul_mv_q4_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mv_q8_0_f32",
            GgmlType::Q6_K => "kernel_mul_mv_q6_K_f32",
        }
    }

    /// Metal kernel function name for the matrix-matrix (mm) kernel
    /// — used for `m > MM_ROUTING_THRESHOLD`.  Ported from
    /// llama.cpp's `kernel_mul_mm_<qtype>_f32` template (ADR-011 Phase 3).
    fn mm_kernel_name(self) -> &'static str {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => "unsupported",
            GgmlType::Q4_0 => "kernel_mul_mm_q4_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_q8_0_f32",
            GgmlType::Q6_K => "kernel_mul_mm_q6_K_f32",
        }
    }

    /// Metal kernel function name for the tensor-API matrix-matrix
    /// variant (ADR-011 Phase 3 Wave P3b-tensor).  On M3+ this path uses
    /// `mpp::tensor_ops::matmul2d<>` which hits the hardware tensor cores
    /// for 2-3× the FLOP throughput of the simdgroup MMA variant.  The
    /// dispatcher falls back to `mm_kernel_name()` when the tensor
    /// pipeline fails to compile (pre-M3 hardware).
    fn mm_tensor_kernel_name(self) -> &'static str {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => "unsupported",
            GgmlType::Q4_0 => "kernel_mul_mm_q4_0_tensor_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_q8_0_tensor_f32",
            GgmlType::Q6_K => "kernel_mul_mm_q6_K_tensor_f32",
        }
    }
}

/// Cached tensor-API availability — `None` until the first mm dispatch,
/// then `Some(true)` if the tensor mm kernels compile on this device,
/// `Some(false)` if they don't (we transparently fall back to the
/// simdgroup MMA variants).  One-shot probe keeps the hot path
/// branch-free after the first layer.
static TENSOR_MM_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn probe_tensor_mm(registry: &mut KernelRegistry, device: &MlxDevice) -> bool {
    *TENSOR_MM_AVAILABLE.get_or_init(|| {
        // Attempt to compile one tensor-mm pipeline; success means the
        // Metal runtime has `<metal_tensor>` +
        // `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
        // available on this device (M3+).  Probing via Q4_0 is sufficient
        // — all three qtype variants share the same tensor_ops surface.
        let ok = registry
            .get_pipeline("kernel_mul_mm_q4_0_tensor_f32", device.metal_device())
            .is_ok();
        if std::env::var("MLX_LOG_TENSOR_PROBE").is_ok() {
            eprintln!("[mlx-native] tensor_mm probe: {}", if ok { "OK (using tensor variant)" } else { "FAILED (falling back to simdgroup MMA)" });
        }
        ok
    })
}

/// llama.cpp's `ne11_mm_min` threshold for routing between mat-vec and
/// mat-mat (see `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2046`).
/// At prefill m > 8, the mm kernel's threadgroup-staged weight tile reuse
/// beats the mv kernel's per-row DRAM re-read by 5-30x on Apple Silicon.
/// At m <= 8 the mv kernel wins on launch overhead for narrow inputs.
pub const MM_ROUTING_THRESHOLD: u32 = 8;

/// Parameters for GGML block-format quantized mat-vec.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizedMatmulParams {
    /// Number of input rows (1 for decode).
    pub m: u32,
    /// Number of output columns (weight rows).
    pub n: u32,
    /// Input dimension (weight cols before quantization).
    /// Must be divisible by the block's QK value.
    pub k: u32,
    /// GGML quantization type.
    pub ggml_type: GgmlType,
}

/// GPU-side params struct — must match the Metal shader's `GgmlMatvecParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatvecGpuParams {
    ne00: i64, // K
    ne01: i64, // N
    ne02: i64, // batch (weights)
    ne10: i64, // K
    ne12: i64, // batch (input)
    ne0: i64,  // N (output stride)
    ne1: i64,  // M
    r2: u32,   // ne12/ne02
    r3: u32,   // always 1
}

/// GPU-side params struct for the mat-mat (mm) kernel.
/// Must match `GgmlMatmulMmParams` in
/// `/opt/mlx-native/src/shaders/quantized_matmul_mm.metal`.
/// Mirrors llama.cpp's `ggml_metal_kargs_mul_mm`
/// (`ggml/src/ggml-metal/ggml-metal-impl.h:423`).
///
/// Explicit 4-byte padding is inserted between `ne12` and `nb10` so the
/// Rust struct has deterministic layout and matches the natural Metal
/// struct alignment (u64 members align to 8 bytes).  bytemuck::Pod
/// requires no implicit padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatmulMmGpuParams {
    ne00: i32,    // K
    ne02: i32,    // batch(src0), always 1 for our projections
    nb01: u64,    // bytes per weight row
    nb02: u64,    // bytes per weight batch
    nb03: u64,    // unused (always 0)
    ne12: i32,    // batch(src1), always 1
    _pad0: u32,   // align nb10 to 8
    nb10: u64,    // = sizeof(float) = 4
    nb11: u64,    // bytes per input row = K * sizeof(float)
    nb12: u64,    // bytes per input batch = M * nb11
    nb13: u64,    // unused
    ne0: i32,     // N (output stride)
    ne1: i32,     // M
    r2: i16,      // 1
    r3: i16,      // 1
    _pad1: u32,   // trailing pad so sizeof == multiple of 8 (u64 align)
}

/// Quantized matmul for GGML block format weights.
///
/// Weight buffer contains raw GGML blocks (same bytes as GGUF on disk).
/// Input is f32, output is f32.
///
/// Routes between two Metal kernels based on `m`:
///
/// - `m <= MM_ROUTING_THRESHOLD` (8) -> uses the matrix-vector kernel
///   (`kernel_mul_mv_q*_f32`).  Lower launch overhead; one output row
///   per threadgroup-block in the M axis.
/// - `m > MM_ROUTING_THRESHOLD` -> uses the matrix-matrix kernel
///   (`kernel_mul_mm_q*_f32`, ADR-011 Phase 3 port from llama.cpp).
///   Tiles the input at 64x32 and stages a dequantized weight tile into
///   threadgroup shared memory, reusing each weight block across a 32-row
///   block of inputs.  At prefill m=2455 this is ~32x less DRAM traffic.
///
/// The threshold matches llama.cpp's `ne11_mm_min = 8`
/// (ggml-metal-ops.cpp:2046).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - K is not divisible by the GGML block QK value
/// - Buffer sizes don't match expected dimensions
/// - M, K, or N are zero
pub fn quantized_matmul_ggml(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();

    // --- Validate (common to mv and mm paths) ---
    match params.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q6_K => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "quantized_matmul_ggml does not support {:?} — use a different dispatch path",
                other
            )));
        }
    }
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, and N must all be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({}) must be divisible by block QK ({})",
            params.k, qk
        )));
    }

    let blocks_per_row = params.k / qk;
    let expected_weight_bytes =
        (params.n as usize) * (blocks_per_row as usize) * (block_bytes as usize);
    if weight.byte_len() < expected_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "Weight buffer too small: expected {} bytes for {:?} [{}x{}], got {}",
            expected_weight_bytes,
            params.ggml_type,
            params.n,
            params.k,
            weight.byte_len()
        )));
    }

    let expected_input_bytes =
        (params.m as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "Input buffer too small: expected {} bytes for [{}x{}] f32, got {}",
            expected_input_bytes, params.m, params.k, input.byte_len()
        )));
    }

    let expected_output_bytes =
        (params.m as usize) * (params.n as usize) * DType::F32.size_of();
    if output.byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "Output buffer too small: expected {} bytes for [{}x{}] f32, got {}",
            expected_output_bytes, params.m, params.n, output.byte_len()
        )));
    }

    // ADR-011 Phase 3 Wave P3a — route on m threshold.
    //
    // The mm kernel stages a 64x32 weight tile into threadgroup shared
    // memory and reuses it across a 32-row block of M.  This cuts DRAM
    // weight-read bandwidth by ~32x at prefill m=2455 and delivers a
    // 5-30x per-kernel speedup over the mv path (which re-reads every
    // weight block once per M row).  The mv path is still preferable at
    // low M (decode m=1, short-prompt prefill m<=8) where launch overhead
    // dominates tile reuse savings.
    //
    // Threshold matches llama.cpp's `ne11_mm_min = 8`
    // (ggml-metal-ops.cpp:2046).  The mm kernel also requires K >= NK=32,
    // which every projection in our Gemma 4 DWQ model satisfies — guard
    // kept so any future shape smaller than 32 falls back to mv.
    if params.m > MM_ROUTING_THRESHOLD && params.k >= 32 {
        dispatch_mm(encoder, registry, device, input, weight, output, params)
    } else {
        dispatch_mv(encoder, registry, device, input, weight, output, params)
    }
}

/// Test-only helper: force the mm dispatch path.  Used by the mm parity
/// tests (`tests/test_quantized_matmul_mm.rs`).  This entry point
/// intentionally bypasses the public dispatcher's routing decision so
/// that tests can verify mm vs mv parity at every M (including the
/// m <= 8 range where the production dispatcher normally picks mv).
///
/// Not intended for production callers — use `quantized_matmul_ggml`
/// above, which routes by m.
#[doc(hidden)]
pub fn dispatch_mm_for_test(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    // Re-run common validation so this entry point is safe on its own.
    let qk = params.ggml_type.block_values();
    match params.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q6_K => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_mm_for_test does not support {:?}", other
            )));
        }
    }
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, and N must all be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({}) must be divisible by block QK ({})", params.k, qk
        )));
    }
    dispatch_mm(encoder, registry, device, input, weight, output, params)
}

/// Matrix-vector dispatch (original path, unchanged from pre-Phase-3).
/// Used for decode (m=1) and small-prompt prefills (m <= 8).
fn dispatch_mv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    let kernel_name = params.ggml_type.kernel_name();
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let gpu_params = GgmlMatvecGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: 1,
        ne10: params.k as i64,
        ne12: 1,
        ne0: params.n as i64,
        ne1: params.m as i64,
        r2: 1,
        r3: 1,
    };

    let n = params.n as usize;
    let m = params.m as usize;

    let (nth0, nth1, align) = match params.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 => (8u64, 8u64, 8usize),
        GgmlType::Q6_K => (2u64, 32u64, 2usize),
        _ => unreachable!(),
    };

    let threadgroups = metal::MTLSize::new(
        div_ceil(n, align) as u64,
        m as u64,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(nth0, nth1, 1);

    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Matrix-matrix (mm) dispatch.  ADR-011 Phase 3 Wave P3a: port of
/// llama.cpp's `kernel_mul_mm_<qtype>_f32`.  64x32 output tile, 4
/// simdgroups (128 threads), threadgroup-staged A+B with simdgroup MMA.
/// See `/opt/mlx-native/src/shaders/quantized_matmul_mm.metal`.
fn dispatch_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    // ADR-011 Phase 3 Wave P3b-tensor — prefer the tensor_ops::matmul2d
    // variant on M3+ (hardware tensor cores); fall back to the simdgroup
    // MMA kernel if the probe fails or the tensor kernel can't compile
    // on this device.
    let use_tensor = probe_tensor_mm(registry, device);
    let kernel_name = if use_tensor {
        params.ggml_type.mm_tensor_kernel_name()
    } else {
        params.ggml_type.mm_kernel_name()
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();
    let blocks_per_row = params.k / qk;
    let nb01 = (blocks_per_row as u64) * (block_bytes as u64);
    let nb11 = (params.k as u64) * DType::F32.size_of() as u64;

    let gpu_params = GgmlMatmulMmGpuParams {
        ne00: params.k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * (params.n as u64),
        nb03: 0,
        ne12: 1,
        _pad0: 0,
        nb10: DType::F32.size_of() as u64,
        nb11,
        nb12: nb11 * (params.m as u64),
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        _pad1: 0,
    };

    // Tile geometry from llama.cpp: NR0=64 (output-N per tg), NR1=32 (M per tg),
    // 4 simdgroups/tg -> 128 threads/tg.
    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;

    let threadgroups = metal::MTLSize::new(
        (params.m as u64 + NR1 - 1) / NR1,
        (params.n as u64 + NR0 - 1) / NR0,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    // Threadgroup shared memory: sa (A tile half, 64*32 = 2048 halfs = 4096 B)
    // + sb (B tile float, 32*32 = 1024 floats = 4096 B) = 8192 bytes.
    // llama.cpp allocates identical 8192 bytes; the partial-tile write-back
    // path reuses the same region (so no extra allocation needed).
    const SHMEM_BYTES: u64 = 8192;

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, SHMEM_BYTES)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ===========================================================================
// Wave P4.19 — bf16-input perm021 entry point for tensor-mm
//
// Used by the hf2q batched prefill's O-projection to consume the flash-
// attention output buffer (bf16 at [n_heads, seq_len, head_dim] physical
// layout) directly, eliminating the dedicated `permute_021_bf16_to_f32`
// dispatch that otherwise runs every layer.
//
// Semantics:
//   output[t, c] = sum_{i=0..K-1} weight[c, i] * src1_logical[t, i]
// where src1_logical[t, i] is obtained from the physical bf16 buffer at
//   src1_bf16[h * seq_len * head_dim + t * head_dim + f],  h = i / head_dim,
//                                                          f = i mod head_dim.
// K must equal n_heads * head_dim, and head_dim must be a multiple of NK=32
// (Gemma 4: head_dim ∈ {256 sliding, 512 global} — both satisfy).
//
// See /opt/mlx-native/src/shaders/quantized_matmul_mm_tensor.metal kernel
// `hf2q_mul_mm_tensor_perm021_impl` for the byte-exact equivalence proof.
// ===========================================================================

/// GPU-side params for the perm021 tensor-mm kernel — must match the
/// shader's `GgmlMatmulMmTensorPerm021Params`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatmulMmTensorPerm021GpuParams {
    ne00: i32,   // K = n_heads * head_dim
    ne02: i32,
    nb01: u64,   // bytes per weight row
    nb02: u64,
    nb03: u64,
    ne12: i32,
    _pad0: u32,
    nb10: u64,   // = sizeof(bfloat) = 2
    nb11: u64,   // unused (kept for struct symmetry)
    nb12: u64,
    nb13: u64,
    ne0: i32,    // N = hidden_size
    ne1: i32,    // M = seq_len
    r2: i16,
    r3: i16,
    // NO _pad between r3 and head_dim: Metal auto-aligns int32_t after
    // two int16_t at 2-byte boundary; the next int32_t naturally lands
    // at offset 84 (= 80 + 2 + 2).  Adding a u32 pad here would slide
    // head_dim to byte 88, mismatching the Metal struct layout and
    // causing the GPU to read head_dim = 0 (verified empirically
    // 2026-04-20: an earlier version with _pad1 produced first_token
    // 236772 instead of the expected 29294; removing the pad restored
    // byte-identity).
    head_dim: i32,
    seq_len: i32,
    // Trailing pad to bring struct size to a multiple of 8 (largest
    // member alignment = u64).  Rust's repr(C) auto-inserts this to 96
    // bytes anyway, but bytemuck::Pod rejects implicit trailing padding;
    // an explicit pad makes the derive compile and matches Metal's
    // struct size exactly.
    _pad_trailing: u32,
}

/// Params for the perm021 tensor-mm dispatch.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizedMatmulPerm021Params {
    /// M — number of rows / tokens.
    pub m: u32,
    /// N — number of output cols (= hidden_size).
    pub n: u32,
    /// K — hidden_size (= n_heads * head_dim).  Must be divisible by
    /// the block's QK and by `head_dim`.
    pub k: u32,
    /// Head dimension.  Must be a multiple of NK=32.
    pub head_dim: u32,
    /// GGML quantization type of the weight (Q4_0 or Q6_K).
    pub ggml_type: GgmlType,
}

/// Dispatch the bf16-input permuted-021 variant of the tensor-mm kernel.
///
/// `weight` is the quantized O-projection weight `[n, k]`.
/// `input_bf16` is the flash-attention output at physical layout
///   `[n_heads, seq_len, head_dim]` bf16.
/// `output` is the standard `[m, n]` f32 O-proj result.
///
/// # Errors
/// Returns `InvalidArgument` if:
/// - `ggml_type` is not Q4_0 or Q6_K
/// - `head_dim` is not a positive multiple of 32
/// - `k != n_heads * head_dim`  (we infer n_heads = k / head_dim)
/// - buffer sizes don't match the declared shapes
pub fn quantized_matmul_mm_tensor_perm021(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    weight: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulPerm021Params,
) -> Result<()> {
    let kernel_name = match params.ggml_type {
        GgmlType::Q4_0 => "kernel_mul_mm_q4_0_tensor_bf16_perm021",
        GgmlType::Q6_K => "kernel_mul_mm_q6_K_tensor_bf16_perm021",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "quantized_matmul_mm_tensor_perm021: unsupported ggml_type {:?} \
                 (only Q4_0 and Q6_K are instantiated)",
                other
            )));
        }
    };

    if params.head_dim == 0 || params.head_dim % 32 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: head_dim {} must be a positive \
             multiple of 32 (NK tile width)",
            params.head_dim
        )));
    }
    if params.k % params.head_dim != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: k ({}) must be divisible by \
             head_dim ({})",
            params.k, params.head_dim
        )));
    }

    // Input-buffer size check: n_heads * seq_len * head_dim * sizeof(bfloat).
    let n_heads = params.k / params.head_dim;
    let expected_input_bytes = (n_heads as usize) * (params.m as usize)
        * (params.head_dim as usize) * 2;
    if input_bf16.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: input_bf16 buffer too small \
             (have {}, need {})",
            input_bf16.byte_len(), expected_input_bytes
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();
    let blocks_per_row = params.k / qk;
    let nb01 = (blocks_per_row as u64) * (block_bytes as u64);

    let gpu_params = GgmlMatmulMmTensorPerm021GpuParams {
        ne00: params.k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * (params.n as u64),
        nb03: 0,
        ne12: 1,
        _pad0: 0,
        nb10: 2, // sizeof(bfloat)
        nb11: 0, // unused; B-stage computes addresses directly
        nb12: 0,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        head_dim: params.head_dim as i32,
        seq_len: params.m as i32,
        _pad_trailing: 0,
    };

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;
    const SHMEM_BYTES: u64 = 8192;

    let threadgroups = metal::MTLSize::new(
        (params.m as u64 + NR1 - 1) / NR1,
        (params.n as u64 + NR0 - 1) / NR0,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input_bf16)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, SHMEM_BYTES)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}
