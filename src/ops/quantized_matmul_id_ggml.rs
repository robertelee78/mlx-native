//! GGML block-format expert-routed (MoE) quantized matrix-vector multiply dispatch.
//!
//! Encodes a GPU compute command that performs, for each (token, expert-slot):
//!   expert_id = ids[token * top_k + slot]
//!   output[token*top_k + slot][col] = sum_k(dequant(weight[expert_id][col][k]) * input[token][k])
//!
//! This is the _id variant of quantized_matmul_ggml: same GGML block dequantization
//! but with per-token expert selection via an ids buffer, enabling fused MoE dispatch.
//!
//! Derived from candle-metal-kernels (Apache-2.0) kernel_mul_mv_id template
//! and mlx-native's quantized_matmul_ggml kernels.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::GgmlType;

// ---- GPU params struct ----

/// GPU-side params struct — must match the Metal shader's `GgmlMatvecIdParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatvecIdGpuParams {
    ne00: i64,           // K
    ne01: i64,           // N
    ne02: i64,           // 1 (unused)
    ne10: i64,           // K
    ne12: i64,           // 1 (unused)
    ne0: i64,            // N (output stride)
    ne1: i64,            // total output rows = n_tokens * top_k
    r2: u32,             // 1
    r3: u32,             // 1
    top_k: u32,          // experts per token
    n_tokens: u32,       // number of input tokens
    expert_stride: i64,  // bytes between expert weight slices
}

// ---- Public types ----

/// Parameters describing the expert-routed GGML quantized matmul dimensions.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizedMatmulIdParams {
    /// Number of input tokens.
    pub n_tokens: u32,
    /// Number of experts each token is routed to (top-k).
    pub top_k: u32,
    /// Number of output columns per expert (weight rows).
    pub n: u32,
    /// Input dimension (weight cols before quantization).
    /// Must be divisible by the GGML block QK value.
    pub k: u32,
    /// Total number of experts in the stacked weight buffer.
    pub n_experts: u32,
    /// Byte stride between expert weight slices in the stacked buffer.
    pub expert_stride: u64,
    /// GGML quantization type.
    pub ggml_type: GgmlType,
}

impl GgmlType {
    /// Metal kernel function name for the mat-vec `_id` variant.
    fn id_kernel_name(self) -> &'static str {
        match self {
            GgmlType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
            GgmlType::Q6_K => "kernel_mul_mv_id_q6_K_f32",
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => "unsupported",
        }
    }

    /// Metal kernel function name for the mat-mat `_id` variant (ADR-011
    /// Phase 3 Wave P3a port of llama.cpp's `kernel_mul_mm_id_<q>_f32`).
    fn id_mm_kernel_name(self) -> &'static str {
        match self {
            GgmlType::Q4_0 => "kernel_mul_mm_id_q4_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_id_q8_0_f32",
            GgmlType::Q6_K => "kernel_mul_mm_id_q6_K_f32",
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => "unsupported",
        }
    }

    /// Tensor-API variant of the mm_id kernel (ADR-011 Phase 3 Wave
    /// P3b-tensor).  Dispatcher falls back to `id_mm_kernel_name()` when
    /// the tensor pipeline probe fails on pre-M3 hardware.
    fn id_mm_tensor_kernel_name(self) -> &'static str {
        match self {
            GgmlType::Q4_0 => "kernel_mul_mm_id_q4_0_tensor_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_id_q8_0_tensor_f32",
            GgmlType::Q6_K => "kernel_mul_mm_id_q6_K_tensor_f32",
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => "unsupported",
        }
    }
}

/// One-shot probe for mm_id tensor-API availability.  Cached separately
/// from the dense-mm probe in quantized_matmul_ggml.rs because these are
/// distinct shader files; whichever runs first pays its own compile cost.
static TENSOR_MM_ID_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn probe_tensor_mm_id(registry: &mut KernelRegistry, device: &MlxDevice) -> bool {
    *TENSOR_MM_ID_AVAILABLE.get_or_init(|| {
        let ok = registry
            .get_pipeline("kernel_mul_mm_id_q4_0_tensor_f32", device.metal_device())
            .is_ok();
        if std::env::var("MLX_LOG_TENSOR_PROBE").is_ok() {
            eprintln!("[mlx-native] tensor_mm_id probe: {}", if ok { "OK (using tensor variant for MoE)" } else { "FAILED (falling back to simdgroup MMA)" });
        }
        ok
    })
}

/// Encode an expert-routed GGML quantized matrix-vector multiply.
///
/// Weight buffer contains raw GGML blocks stacked as `[n_experts, N, packed_K]`.
/// Input is f32 `[n_tokens, K]`, output is f32 `[n_tokens * top_k, N]`.
/// The `ids` buffer `[n_tokens * top_k]` u32 selects which expert to use for
/// each (token, slot) pair.
///
/// # Arguments
///
/// * `encoder`  -- Command encoder to record the dispatch into.
/// * `registry` -- Kernel registry (compiles shader on first call).
/// * `device`   -- Metal device.
/// * `input`    -- f32 input buffer, shape `[n_tokens, K]`.
/// * `weight`   -- Stacked GGML block weight buffer, `[n_experts, N, packed_K]`.
/// * `ids`      -- u32 expert index buffer, shape `[n_tokens * top_k]`.
/// * `output`   -- f32 output buffer, shape `[n_tokens * top_k, N]`.
/// * `params`   -- Dimensions and quantization parameters.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - K is not divisible by the GGML block QK value
/// - Buffer sizes don't match expected dimensions
/// - Any dimension is zero
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();

    // --- Validate dimensions ---
    if params.n_tokens == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml: n_tokens, K, and N must all be > 0".into(),
        ));
    }
    if params.top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml: top_k must be > 0".into(),
        ));
    }
    if params.n_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml: n_experts must be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: K ({}) must be divisible by block QK ({})",
            params.k, qk
        )));
    }

    // --- Validate buffer sizes ---
    let expected_input_bytes =
        (params.n_tokens as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: input buffer too small: expected {} bytes for [{} x {}] f32, got {}",
            expected_input_bytes, params.n_tokens, params.k, input.byte_len()
        )));
    }

    let blocks_per_row = params.k / qk;
    let per_expert_bytes =
        (params.n as usize) * (blocks_per_row as usize) * (block_bytes as usize);

    // Validate expert_stride is sane
    if params.expert_stride < per_expert_bytes as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: expert_stride ({}) < per_expert_bytes ({})",
            params.expert_stride, per_expert_bytes
        )));
    }

    let total_weight_bytes = per_expert_bytes * (params.n_experts as usize);
    if weight.byte_len() < total_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: weight buffer too small: expected {} bytes for {} experts, got {}",
            total_weight_bytes, params.n_experts, weight.byte_len()
        )));
    }

    let total_rows = (params.n_tokens as usize) * (params.top_k as usize);
    let expected_ids_bytes = total_rows * DType::U32.size_of();
    if ids.byte_len() < expected_ids_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: ids buffer too small: expected {} bytes for [{} * {}] u32, got {}",
            expected_ids_bytes, params.n_tokens, params.top_k, ids.byte_len()
        )));
    }

    let expected_output_bytes = total_rows * (params.n as usize) * DType::F32.size_of();
    if output.byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: output buffer too small: expected {} bytes for [{} x {}] f32, got {}",
            expected_output_bytes, total_rows, params.n, output.byte_len()
        )));
    }

    // ADR-011 Phase 3 — route on n_tokens threshold.
    //
    // At prefill with n_tokens > 8, dispatch the two-stage mm_id kernels
    // (map0 + mm).  Each expert's weight tile is staged once to
    // threadgroup shmem per 32-row block of that expert's routed tokens
    // — identical to the dense mm dispatcher's win but per-expert.
    //
    // P3b-tensor.2 — extended to top_k=1 (Gemma 4's MoE down call).
    // Without this the down call's 19,640-row matmul falls back to mv_id
    // and re-reads each expert's weights once per row — ~50% of prefill
    // wall time burnt on weight re-reads.  Today we have ne20_1 and
    // ne20_8 instantiations; other top_k values still fall back to mv_id.
    //
    // Falls back to the mv_id path for:
    //   * decode (n_tokens <= 8)
    //   * top_k values without a map0 instantiation
    //   * K < 32 (mm tile requires NK=32)
    if params.n_tokens > (MM_ID_ROUTING_THRESHOLD as u32)
        && (params.top_k == 1 || params.top_k == 8)
        && params.k >= 32
    {
        return dispatch_id_mm(
            encoder, registry, device, input, weight, ids, output, params,
        );
    }

    dispatch_id_mv(encoder, registry, device, input, weight, ids, output, params)
}

/// Same contract as `quantized_matmul_id_ggml`, but takes caller-owned
/// `IdMmScratch` so batched-prefill dispatches avoid the per-call
/// `MTLDevice.newBufferWithLength:` allocations the auto entry point
/// incurs (ADR-011 Phase 3 Wave P3b — "scratch pooling").
///
/// When the dispatch routes to the mv_id path (decode / top_k != 8 /
/// K < 32), the scratch is not touched — it is only used on the mm_id
/// path.  Callers may over-size the scratch once per prefill and share
/// it across every mm_id call in the forward pass.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_pooled(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    // Mirror the validation + routing logic from `quantized_matmul_id_ggml`
    // so the pooled path has identical correctness invariants.  (We keep
    // the two entry points separate rather than extracting a shared inner
    // because the scratch is only relevant on the mm_id branch — lifting
    // scratch into the mv branch would add unused parameters.)
    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();

    if params.n_tokens == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml_pooled: n_tokens, K, and N must all be > 0".into(),
        ));
    }
    if params.top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml_pooled: top_k must be > 0".into(),
        ));
    }
    if params.n_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml_pooled: n_experts must be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml_pooled: K ({}) must be divisible by block QK ({})",
            params.k, qk
        )));
    }

    let expected_input_bytes =
        (params.n_tokens as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml_pooled: input buffer too small: expected {} bytes for [{} x {}] f32, got {}",
            expected_input_bytes, params.n_tokens, params.k, input.byte_len()
        )));
    }

    let blocks_per_row = params.k / qk;
    let per_expert_bytes =
        (params.n as usize) * (blocks_per_row as usize) * (block_bytes as usize);

    if params.expert_stride < per_expert_bytes as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml_pooled: expert_stride ({}) < per_expert_bytes ({})",
            params.expert_stride, per_expert_bytes
        )));
    }

    let total_weight_bytes = per_expert_bytes * (params.n_experts as usize);
    if weight.byte_len() < total_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml_pooled: weight buffer too small: expected {} bytes for {} experts, got {}",
            total_weight_bytes, params.n_experts, weight.byte_len()
        )));
    }

    let total_rows = (params.n_tokens as usize) * (params.top_k as usize);
    let expected_ids_bytes = total_rows * DType::U32.size_of();
    if ids.byte_len() < expected_ids_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml_pooled: ids buffer too small: expected {} bytes for [{} * {}] u32, got {}",
            expected_ids_bytes, params.n_tokens, params.top_k, ids.byte_len()
        )));
    }

    let expected_output_bytes = total_rows * (params.n as usize) * DType::F32.size_of();
    if output.byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml_pooled: output buffer too small: expected {} bytes for [{} x {}] f32, got {}",
            expected_output_bytes, total_rows, params.n, output.byte_len()
        )));
    }

    // P3b-tensor.2 — accept top_k ∈ {1, 8} (Gemma 4's MoE down/gate_up).
    if params.n_tokens > (MM_ID_ROUTING_THRESHOLD as u32)
        && (params.top_k == 1 || params.top_k == 8)
        && params.k >= 32
    {
        return dispatch_id_mm_pooled(
            encoder, registry, device, input, weight, ids, output,
            scratch, params,
        );
    }

    dispatch_id_mv(encoder, registry, device, input, weight, ids, output, params)
}

/// The n_tokens threshold at which `quantized_matmul_id_ggml` switches
/// from the mv_id kernel to the mm_id kernel.  Matches llama.cpp's
/// `ne11_mm_min = 8` (ggml-metal-ops.cpp:2046).
pub const MM_ID_ROUTING_THRESHOLD: u32 = 8;

/// Matrix-vector `_id` dispatch (decode path, unchanged from pre-Phase-3).
#[allow(clippy::too_many_arguments)]
fn dispatch_id_mv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let total_rows = (params.n_tokens as usize) * (params.top_k as usize);

    let kernel_name = params.ggml_type.id_kernel_name();
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let gpu_params = GgmlMatvecIdGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: 1,
        ne10: params.k as i64,
        ne12: 1,
        ne0: params.n as i64,
        ne1: total_rows as i64,
        r2: 1,
        r3: 1,
        top_k: params.top_k,
        n_tokens: params.n_tokens,
        expert_stride: params.expert_stride as i64,
    };

    let (nth0, nth1, align) = match params.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 => (8u64, 8u64, 8usize),
        GgmlType::Q6_K => (2u64, 32u64, 2usize),
        GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K => {
            return Err(MlxError::InvalidArgument(format!(
                "quantized_matmul_id_ggml does not support {:?}",
                params.ggml_type
            )));
        }
    };

    let n = params.n as usize;
    let m = total_rows;

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
            (3, KernelArg::Buffer(ids)),
            (4, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Caller-owned scratch for the `_id` mm path's map0 stage.
///
/// Holds the two small buffers map0 writes and mm_id reads
/// (`htpe`: `[n_experts]` per-expert routed-token count, `hids`:
/// `[n_experts, n_tokens]` per-expert routed-token list).  Passing one
/// instance through every mm_id call in a prefill amortises what would
/// otherwise be two Metal allocations per MoE layer — ~60 allocations
/// per Gemma 4 prefill.
///
/// Size the scratch for the largest `(n_experts, n_tokens)` pair the
/// session will dispatch; callers use `IdMmScratch::alloc(dev, n_experts,
/// max_n_tokens)` once at prefill start.  Smaller subsequent dispatches
/// reuse the same buffers (kernel only touches the first
/// `n_experts * n_tokens` u32s).
pub struct IdMmScratch {
    pub htpe: MlxBuffer,
    pub hids: MlxBuffer,
    n_experts_cap: u32,
    n_tokens_cap: u32,
}

impl IdMmScratch {
    /// Allocate scratch sized to `n_experts * max_n_tokens` u32s.
    pub fn alloc(
        device: &MlxDevice,
        n_experts: u32,
        max_n_tokens: u32,
    ) -> Result<Self> {
        let htpe = device.alloc_buffer(
            (n_experts as usize) * DType::U32.size_of(),
            DType::U32,
            vec![n_experts as usize],
        )?;
        let hids = device.alloc_buffer(
            (n_experts as usize) * (max_n_tokens as usize) * DType::U32.size_of(),
            DType::U32,
            vec![n_experts as usize, max_n_tokens as usize],
        )?;
        Ok(Self {
            htpe,
            hids,
            n_experts_cap: n_experts,
            n_tokens_cap: max_n_tokens,
        })
    }

    fn check_capacity(&self, n_experts: u32, n_tokens: u32) -> Result<()> {
        if n_experts > self.n_experts_cap {
            return Err(MlxError::InvalidArgument(format!(
                "IdMmScratch: n_experts ({}) > cap ({})",
                n_experts, self.n_experts_cap,
            )));
        }
        if n_tokens > self.n_tokens_cap {
            return Err(MlxError::InvalidArgument(format!(
                "IdMmScratch: n_tokens ({}) > cap ({})",
                n_tokens, self.n_tokens_cap,
            )));
        }
        Ok(())
    }
}

/// Matrix-matrix `_id` dispatch using caller-owned scratch (ADR-011 Phase
/// 3 Wave P3b).
#[allow(clippy::too_many_arguments)]
fn dispatch_id_mm_pooled(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    scratch.check_capacity(params.n_experts, params.n_tokens)?;

    // Translate the dispatcher-facing params to the mm_id internal dispatch
    // shape.  Same fields; different type keeps the public mv params from
    // becoming mm-specific.
    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: params.n_tokens,
        top_k: params.top_k,
        n: params.n,
        k: params.k,
        n_experts: params.n_experts,
        expert_stride: params.expert_stride,
        ggml_type: params.ggml_type,
    };

    dispatch_id_mm_for_test(
        encoder, registry, device,
        input, weight, ids,
        &mut scratch.htpe, &mut scratch.hids, output, &dispatch,
    )
}

/// Matrix-matrix `_id` dispatch that allocates scratch on every call.
///
/// Retained for the auto-allocating `quantized_matmul_id_ggml` entry
/// point (tests, non-prefill callers); the pooled entry point
/// `quantized_matmul_id_ggml_pooled` is preferred for batched prefill.
#[allow(clippy::too_many_arguments)]
fn dispatch_id_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let mut scratch = IdMmScratch::alloc(device, params.n_experts, params.n_tokens)?;
    dispatch_id_mm_pooled(
        encoder, registry, device,
        input, weight, ids, output,
        &mut scratch, params,
    )
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ============================================================================
// ADR-011 Phase 3 Wave P3a: `_id` matrix-matrix (mm) path.
//
// Ports llama.cpp's `kernel_mul_mm_id_map0_ne20_<N>` + `kernel_mul_mm_id_<q>_f32`
// two-stage dispatch.  Used for MoE projections at prefill — instead of
// re-reading each expert's weight blocks once per routed (token, slot) pair,
// the mm kernel stages a 64x32 expert weight tile into threadgroup shared
// memory and reuses it across a 32-row block of the expert's routed tokens.
//
// The preprocessing step (`map0`) is what lets mm work for MoE: it
// regroups the flat `[n_tokens, top_k]` ids table into per-expert routed
// token lists so each mm tile is homogeneous in its choice of expert
// weight slab.  Without map0, consecutive M-rows in a tile could route to
// different experts, defeating weight reuse.
//
// Same staging strategy as Wave P3a Commit 1 (non-id): the kernel exists,
// tests verify correctness, but the public `quantized_matmul_id_ggml`
// dispatcher is NOT rerouted yet — tests call `dispatch_id_mm_for_test`.
// ============================================================================

/// Host-side params for the `_id` mm path's `map0` preprocessor.
///
/// Matches `GgmlMatmulIdMm_Map0Params` in
/// `/opt/mlx-native/src/shaders/quantized_matmul_id_mm.metal`.  Explicit
/// 4-byte trailing padding so the struct aligns to 8 (u64).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlIdMmMap0GpuParams {
    ne10: i32,       // unused, kept for struct symmetry
    ne11: i32,       // n_expert_used (bcast, == ne20)
    nb11: u64,       // unused
    nb12: u64,       // unused
    ne21: i32,       // n_tokens
    ne20: i32,       // n_expert_used (top_k)
    nb21: u64,       // bytes per token in the ids table (= ne20 * sizeof(i32))
}

/// Host-side params for the `_id` mm kernel.
///
/// Matches `GgmlMatmulIdMm_MmParams` in
/// `/opt/mlx-native/src/shaders/quantized_matmul_id_mm.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlIdMmMmGpuParams {
    ne00: i32,   // K
    ne02: i32,   // n_experts
    nb01: u64,   // bytes per weight row (within one expert's slab)
    nb02: u64,   // bytes per expert weight slab (= nb01 * N)
    nb03: u64,
    ne11: i32,   // n_expert_used (bcast)
    _pad0: u32,
    nb10: u64,   // = sizeof(float)
    nb11: u64,   // bytes per input row (= K * 4)
    nb12: u64,   // bytes per input batch (= n_tokens * nb11)
    nb13: u64,
    ne20: i32,   // n_expert_used (top_k)
    ne21: i32,   // n_tokens
    ne0: i32,    // N (per-expert output rows)
    ne1: i32,    // batch stride (== ne20 for our packed layout)
    r2: i16,
    r3: i16,
    _pad1: u32,
}

/// Parameters for the `_id` mm dispatch (scratch-buffer sized view).
#[derive(Debug, Clone, Copy)]
pub struct GgmlIdMmDispatchParams {
    /// Number of input tokens.
    pub n_tokens: u32,
    /// Number of experts each token is routed to (top-k).
    pub top_k: u32,
    /// Number of output columns per expert (weight rows).
    pub n: u32,
    /// Input dimension (weight cols before quantization).
    pub k: u32,
    /// Total experts in the stacked weight buffer.
    pub n_experts: u32,
    /// Byte stride between expert weight slices in the stacked buffer.
    pub expert_stride: u64,
    /// GGML quantization type.
    pub ggml_type: GgmlType,
}

impl GgmlIdMmDispatchParams {
    /// Bytes required for the `htpe` scratch buffer (per-expert routed count).
    pub fn htpe_bytes(&self) -> usize {
        (self.n_experts as usize) * DType::U32.size_of()
    }

    /// Bytes required for the `hids` scratch buffer (per-expert routed-token list).
    /// Layout: `[n_experts, n_tokens]` int32 row-major.
    pub fn hids_bytes(&self) -> usize {
        (self.n_experts as usize) * (self.n_tokens as usize) * DType::U32.size_of()
    }
}

/// Test-only helper: force the `_id` mm two-stage dispatch path.
///
/// Runs `kernel_mul_mm_id_map0_ne20_<top_k>` followed by
/// `kernel_mul_mm_id_<qtype>_f32`.
///
/// Input:
///   * `input`   — f32 input rows `[n_tokens, K]`.
///   * `weight`  — stacked expert weights `[n_experts, N, packed_K]`.
///   * `ids`     — flat expert-id table `[n_tokens, top_k]` viewed as
///                 i32 (u32 is byte-equivalent in this range).
///   * `output`  — f32 output `[n_tokens, top_k, N]` row-major.
///
/// Scratch (caller-allocated, zero-init not required):
///   * `htpe`    — `[n_experts]` u32 (per-expert count).
///   * `hids`    — `[n_experts, n_tokens]` i32 (per-expert routed list).
///
/// Not intended for production callers — the public `quantized_matmul_id_ggml`
/// entry point stays on the mv path until the follow-up commit wires the
/// m > 8 threshold.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_id_mm_for_test(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    htpe: &mut MlxBuffer,
    hids: &mut MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlIdMmDispatchParams,
) -> Result<()> {
    let qk = params.ggml_type.block_values();

    // ---- Validate common shapes ----
    match params.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q6_K => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_id_mm_for_test does not support {:?}", other
            )));
        }
    }
    if params.n_tokens == 0 || params.k == 0 || params.n == 0
        || params.top_k == 0 || params.n_experts == 0
    {
        return Err(MlxError::InvalidArgument(
            "n_tokens, K, N, top_k, n_experts must all be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({}) must be divisible by block QK ({})", params.k, qk
        )));
    }

    // Match the top_k template instantiations available in the shader.
    // P3b-tensor.2 — added ne20_1 alongside ne20_8 (Gemma 4 MoE down).
    if params.top_k != 1 && params.top_k != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_id_mm_for_test: top_k {} has no map0 instantiation (need 1 or 8)",
            params.top_k
        )));
    }

    let blocks_per_row = params.k / qk;
    let block_bytes = params.ggml_type.block_bytes();
    let per_expert_bytes =
        (params.n as usize) * (blocks_per_row as usize) * (block_bytes as usize);

    if (params.expert_stride as usize) < per_expert_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "expert_stride ({}) < per_expert_bytes ({})",
            params.expert_stride, per_expert_bytes
        )));
    }

    if weight.byte_len() < per_expert_bytes * params.n_experts as usize {
        return Err(MlxError::InvalidArgument(
            "dispatch_id_mm_for_test: weight buffer too small".into(),
        ));
    }
    if input.byte_len()
        < (params.n_tokens as usize) * (params.k as usize) * DType::F32.size_of()
    {
        return Err(MlxError::InvalidArgument(
            "dispatch_id_mm_for_test: input buffer too small".into(),
        ));
    }
    let total_rows = (params.n_tokens as usize) * (params.top_k as usize);
    if ids.byte_len() < total_rows * DType::U32.size_of() {
        return Err(MlxError::InvalidArgument(
            "dispatch_id_mm_for_test: ids buffer too small".into(),
        ));
    }
    if output.byte_len() < total_rows * (params.n as usize) * DType::F32.size_of() {
        return Err(MlxError::InvalidArgument(
            "dispatch_id_mm_for_test: output buffer too small".into(),
        ));
    }
    if htpe.byte_len() < params.htpe_bytes() {
        return Err(MlxError::InvalidArgument(
            "dispatch_id_mm_for_test: htpe buffer too small".into(),
        ));
    }
    if hids.byte_len() < params.hids_bytes() {
        return Err(MlxError::InvalidArgument(
            "dispatch_id_mm_for_test: hids buffer too small".into(),
        ));
    }

    // ---- Stage 1: map0 — build per-expert routed-token lists ----
    //
    // Dispatch: 1 threadgroup of `n_experts` threads.  Shared memory:
    // `n_experts * top_k * sizeof(uint16)` staging area.
    //
    // ADR-011 Phase 3 Wave P3b-tensor.2 — pick the map0 instantiation
    // whose `ne20` template arg matches our top_k.  Gemma 4 needs both:
    // `ne20_8` for the gate_up call (top_k=8) and `ne20_1` for the
    // down call (top_k=1, where each output row routes to a single
    // expert).  Without ne20_1 the top_k=1 caller falls back to mv_id
    // and re-reads each expert's weights once per (seq_len*top_k) row —
    // ~50% of prefill time wasted on weight re-reads.
    let map0_kernel_name = match params.top_k {
        1 => "kernel_mul_mm_id_map0_ne20_1",
        8 => "kernel_mul_mm_id_map0_ne20_8",
        other => return Err(MlxError::InvalidArgument(format!(
            "dispatch_id_mm_for_test: no map0 instantiation for top_k={}",
            other
        ))),
    };
    let map0_pipeline = registry.get_pipeline(map0_kernel_name, device.metal_device())?;

    let map0_params = GgmlIdMmMap0GpuParams {
        ne10: params.n.try_into().map_err(|_| {
            MlxError::InvalidArgument("N out of i32 range".into())
        })?,
        ne11: params.top_k as i32,
        nb11: 0,
        nb12: 0,
        ne21: params.n_tokens as i32,
        ne20: params.top_k as i32,
        nb21: (params.top_k as u64) * (DType::U32.size_of() as u64),
    };

    let map0_shmem =
        (params.n_experts as u64) * (params.top_k as u64) * std::mem::size_of::<u16>() as u64;
    let map0_threadgroups = metal::MTLSize::new(1, 1, 1);
    let map0_threads = metal::MTLSize::new(params.n_experts as u64, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        map0_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&map0_params))),
            (1, KernelArg::Buffer(ids)),
            (2, KernelArg::Buffer(htpe)),
            (3, KernelArg::Buffer(hids)),
        ],
        &[(0, map0_shmem)],
        map0_threadgroups,
        map0_threads,
    );

    // Memory barrier: the mm kernel reads htpe + hids, map0 wrote them.
    // Without this, Metal's concurrent-dispatch compute encoder lets the
    // two dispatches overlap — mm would read zeros (all-expert early-exit).
    // llama.cpp does the same via `ggml_metal_op_concurrency_reset`
    // (ggml-metal-ops.cpp:2353).
    encoder.memory_barrier();

    // ---- Stage 2: mm_id — matmul with the per-expert lists ----
    //
    // ADR-011 Phase 3 Wave P3b-tensor — prefer the tensor_ops::matmul2d
    // mm_id variant on M3+.  The probe caches the decision after the
    // first dispatch; subsequent calls are branch-free.
    let use_tensor = probe_tensor_mm_id(registry, device);
    let mm_kernel_name = if use_tensor {
        params.ggml_type.id_mm_tensor_kernel_name()
    } else {
        params.ggml_type.id_mm_kernel_name()
    };
    let mm_pipeline = registry.get_pipeline(mm_kernel_name, device.metal_device())?;

    let nb01 = (blocks_per_row as u64) * (block_bytes as u64);
    let row_bytes = (params.k as u64) * (DType::F32.size_of() as u64);

    // Input layout: `[n_tokens, K]` f32 flat.  There is ONE input row per
    // token (shared across all top_k slots), so:
    //   * nb11 (slot stride) = 0 — the kernel advances by `i11 * nb11`
    //     inside the K loop; zero means every slot reads the same token row.
    //   * nb12 (token stride) = K * 4.
    //
    // This differs from llama.cpp's upstream MUL_MAT_ID where `src1` has
    // shape `[K, n_expert_used, n_tokens]` (pre-replicated per slot),
    // making `nb11 = K * 4` and `nb12 = top_k * K * 4` there.  Our mv_id
    // port uses the flat `[n_tokens, K]` layout and so does mm_id.
    let mm_params = GgmlIdMmMmGpuParams {
        ne00: params.k as i32,
        ne02: params.n_experts as i32,
        nb01,
        nb02: params.expert_stride,
        nb03: 0,
        ne11: params.top_k as i32,
        _pad0: 0,
        nb10: DType::F32.size_of() as u64,
        nb11: 0,             // no slot dim in our input
        nb12: row_bytes,     // per-token stride
        nb13: 0,
        ne20: params.top_k as i32,
        ne21: params.n_tokens as i32,
        ne0: params.n as i32,
        ne1: params.top_k as i32,
        r2: 1,
        r3: 1,
        _pad1: 0,
    };

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;

    let mm_threadgroups = metal::MTLSize::new(
        (params.n_tokens as u64 + NR1 - 1) / NR1,
        (params.n as u64 + NR0 - 1) / NR0,
        params.n_experts as u64,
    );
    let mm_threads = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    const MM_SHMEM_BYTES: u64 = 8192;

    encoder.encode_threadgroups_with_args_and_shared(
        mm_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&mm_params))),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(htpe)),
            (4, KernelArg::Buffer(hids)),
            (5, KernelArg::Buffer(output)),
        ],
        &[(0, MM_SHMEM_BYTES)],
        mm_threadgroups,
        mm_threads,
    );

    Ok(())
}
