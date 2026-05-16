//! RMS Normalization GPU dispatch.
//!
//! Computes: `x * rsqrt(mean(x^2) + eps) * weight`
//!
//! The mean is computed over the last dimension.  eps=1e-6 is the standard
//! value for Gemma 4.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::{CapturedOpKind, CommandEncoder, DispatchRecord};
use crate::env_flags::cached_env_default_true;
use std::sync::atomic::AtomicI8;

// ADR-029: cached hot-path env-flag gates for rms_norm
// dispatchers. rms_norm is called multiple times per layer (attn_norm,
// ffn_norm, post-attn-norm, post-ffn-norm, final norm) — many hits per token.
static CACHED_RMS_NORM_V2: AtomicI8 = AtomicI8::new(-1);
static CACHED_FUSED_POST_FF_NORM2_V2: AtomicI8 = AtomicI8::new(-1);
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the RMS norm kernels (embedded at compile time).
pub static RMS_NORM_SHADER_SOURCE: &str = include_str!("../shaders/rms_norm.metal");

/// Register RMS norm shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("rms_norm_f32", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_f16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_bf16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_no_scale_bf16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_no_scale_f32", RMS_NORM_SHADER_SOURCE);
    // Fused RMS norm + elementwise multiply (Phase 4e.2)
    registry.register_source("rms_norm_mul_f32", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_mul_f16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_mul_bf16", RMS_NORM_SHADER_SOURCE);
}

/// Select the fused RMS norm + multiply kernel name based on input dtype.
fn fused_rms_norm_mul_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("rms_norm_mul_f32"),
        DType::F16 => Ok("rms_norm_mul_f16"),
        DType::BF16 => Ok("rms_norm_mul_bf16"),
        _ => Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul unsupported dtype: {}",
            dtype
        ))),
    }
}

/// Dispatch an RMS normalization operation on the GPU.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm sources registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (f32, f16, or bf16).
/// * `weight`     - Weight buffer of shape `[dim]` (same dtype as input; f32 for f32/f16 kernels, bf16 for bf16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - Input dtype is not f32, f16, or bf16.
/// - Input element count does not match rows * dim.
pub fn dispatch_rms_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }

    // hf2q ADR-030 — defense-in-depth: kernel selection below
    // gates on input.dtype() ONLY, and each rms_norm_{f32,f16,bf16}
    // Metal kernel declares its weight buffer with the matching dtype
    // (`device const float*`, `half*`, or `bfloat*`).  Passing a weight
    // buffer of a DIFFERENT dtype than input silently mis-strides the
    // weight reads + over-reads past the buffer end (iter-106 bug
    // signature: BF16 buffer read as F32 → bit-misinterpretation +
    // OOB).  Validate up front so callers see a clear error instead
    // of corrupted hidden states downstream.  Allow input.dtype() ==
    // weight.dtype() == output.dtype() invariant (matches sigmoid_mul
    // and silu_mul's existing dtype-coherence checks).
    if input.dtype() != weight.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm dtype mismatch: input={} != weight={} (kernel rms_norm_{} \
             reads weight at input-dtype stride; mismatched buffer dtypes cause \
             silent bit-misinterpretation + OOB reads — see hf2q ADR-030 iter-106)",
            input.dtype(), weight.dtype(),
            match input.dtype() {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => "?",
            },
        )));
    }
    if input.dtype() != output.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm dtype mismatch: input={} != output={}",
            input.dtype(), output.dtype(),
        )));
    }

    // ADR-028 —float4 + simd_sum variant for F32 path.
    // Requires `dim % 4 == 0`; falls back to scalar when not.
    //
    // ADR-028 default-flipped to ON (operator REFRAME #2).
    // Opt out with `HF2Q_RMS_NORM_V2=0` / `=false` / `=off`.
    let use_v2 = matches!(input.dtype(), DType::F32)
        && (dim % 4 == 0)
        && cached_env_default_true(&CACHED_RMS_NORM_V2, "HF2Q_RMS_NORM_V2");

    let kernel_name = if use_v2 {
        "rms_norm_f32_v2"
    } else {
        match input.dtype() {
            DType::F32 => "rms_norm_f32",
            DType::F16 => "rms_norm_f16",
            DType::BF16 => "rms_norm_bf16",
            _ => {
                return Err(MlxError::InvalidArgument(format!(
                    "RMS norm unsupported dtype: {}",
                    input.dtype()
                )));
            }
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // One threadgroup per row.  Threadgroup size must be a power of 2.
    //
    // ADR-028 iter-360 — tested raising cap from 256 to 1024 (peer pattern):
    //   FALSIFIED at -0.5 to -0.7%.  Wider TG hurts concurrent-TG-per-SM count.
    // ADR-028 iter-361 — tested right-sizing tg to float4 elem count for V2
    //   (so dim=256 per-head norms get tg=64 instead of 256, no idle threads):
    //   FALSIFIED at -0.4%.  Apple Metal compiler must already dedup idle
    //   threads OR per-thread launch overhead amortizes better at tg=256.
    // Production: tg = min(256, dim.next_power_of_two()) — both falsifications
    // documented; future iters skip both directions.
    let mut tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    if use_v2 && tg_size < 32 {
        tg_size = 32;
    }

    // Threadgroup shared memory: v2 only needs one float per simdgroup
    // (≤ 32 SGs for tg_size ≤ 1024 → ≤ 128 bytes).  Scalar path uses
    // tg_size floats for the tree reduction.
    let shared_mem_bytes = if use_v2 {
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    };

    // Tag for the fusion pass (Phase 4e.2): RMS norm can fuse with a
    // subsequent elementwise multiply.
    encoder.set_op_kind(CapturedOpKind::RmsNorm);

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, weight),
            (2, output),
            (3, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// ADR-029 — pre-bake the per-(dtype, rows, dim)
/// `dispatch_rms_norm` dispatch into a `DispatchRecord`.
///
/// Mirrors `dispatch_rms_norm`'s exact kernel-selection + geometry +
/// shared-memory logic so a fast-path `dispatch_record` call produces a
/// byte-identical Metal command stream.  Bakes once per (dtype, rows,
/// dim) combination; subsequent calls skip:
///   - `KernelRegistry::get_pipeline` HashMap lookup (~30 ns)
///   - `MTLSize::new` × 2 (~5 ns)
///   - `dim.next_power_of_two()` + bit-twiddling (~5 ns)
///   - `set_op_kind` indirection (~5 ns) — baked into the record
///
/// On gemma4 APEX-Q5_K_M decode, hidden-size rms_norm dispatches fire
/// ~120 times/token (pre-FF norm + pre-FF norm 2 + router norm +
/// post-FF norm 1, each × 30 layers) — closer to Step 1d's coverage
/// class than Step 1e/1e2, so this substep is potentially measurable
/// above the σ=0.2% bench floor.
///
/// # Bake-time invariants
///
/// - Pipeline = `rms_norm_f32_v2` when `dtype==F32 && dim%4==0 &&
///   HF2Q_RMS_NORM_V2` enabled (default-on); else
///   `rms_norm_{f32,f16,bf16}`.  Returns `Err` on unsupported dtype.
/// - `tg_size = min(256, dim.next_power_of_two())` — falsifications
///   noted at iter-360 (cap raised to 1024 = -0.5/-0.7%) and iter-361
///   (right-size to float4 = -0.4%); production keeps 256.
/// - V2 path uses `(tg_size / 32).max(1) * 4` bytes shmem; scalar path
///   uses `tg_size * 4` bytes (one float per thread for the tree-sum).
/// - `op_kind = CapturedOpKind::RmsNorm` for the fusion pass (Phase 4e.2).
/// - Bindings: input=0, weight=1, output=2, params_buf=3.
/// - threadgroups = `(rows, 1, 1)`, threads_per_tg = `(tg_size, 1, 1)`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if dtype is unsupported (not f32/f16/bf16)
/// or if pipeline lookup fails.
pub fn build_rms_norm_decode_record(
    registry: &mut crate::kernel_registry::KernelRegistry,
    device: &metal::DeviceRef,
    dtype: DType,
    rows: u32,
    dim: u32,
) -> Result<Option<DispatchRecord>> {
    if rows == 0 || dim == 0 {
        return Ok(None); // bake-time guard mirrors dispatch_rms_norm's runtime check
    }

    // Mirror `dispatch_rms_norm`'s kernel-selection logic.
    let use_v2 = matches!(dtype, DType::F32)
        && (dim % 4 == 0)
        && cached_env_default_true(&CACHED_RMS_NORM_V2, "HF2Q_RMS_NORM_V2");

    let kernel_name = if use_v2 {
        "rms_norm_f32_v2"
    } else {
        match dtype {
            DType::F32 => "rms_norm_f32",
            DType::F16 => "rms_norm_f16",
            DType::BF16 => "rms_norm_bf16",
            _ => return Ok(None), // unsupported — caller falls through
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?.clone();

    let mut tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    if use_v2 && tg_size < 32 {
        tg_size = 32;
    }

    let shared_mem_bytes = if use_v2 {
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    };

    let threadgroups = MTLSize::new(rows as u64, 1, 1);
    let threads_per_tg = MTLSize::new(tg_size, 1, 1);

    Ok(Some(DispatchRecord {
        pipeline,
        threadgroups,
        threads_per_tg,
        threadgroup_mem: vec![(0, shared_mem_bytes)],
        params_bytes: Vec::new(),  // params_buf is a runtime buffer at slot 3, not inline bytes
        params_slot: 0,             // ignored when params_bytes is empty
        buffer_slots: vec![0, 1, 2, 3], // input, weight, output, params_buf
        op_kind: CapturedOpKind::RmsNorm,
        kernel_name: kernel_name.to_string(),
    }))
}

/// Dispatch a fused 3-output RMS normalization (f32 only).
///
/// Computes RMS(input) ONCE, then applies three different per-element
/// weight vectors to produce three outputs:
///   output_a = input * rsqrt(mean(input^2) + eps) * weight_a
///   output_b = input * rsqrt(mean(input^2) + eps) * weight_b
///   output_c = input * rsqrt(mean(input^2) + eps) * weight_c
///
/// Replaces three separate `dispatch_rms_norm` calls when the input is
/// the same.  Saves 2 dispatches and reads `input` once instead of
/// three times.  Wave P4.9.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm_f32_triple registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Shared input buffer of shape `[rows, dim]` (f32).
/// * `weight_a`   - First weight vector `[dim]` (f32).
/// * `weight_b`   - Second weight vector `[dim]` (f32).
/// * `weight_c`   - Third weight vector `[dim]` (f32).
/// * `output_a`   - First output buffer `[rows, dim]` (f32).
/// * `output_b`   - Second output buffer `[rows, dim]` (f32).
/// * `output_c`   - Third output buffer `[rows, dim]` (f32).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
pub fn dispatch_rms_norm_f32_triple(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    weight_a: &MlxBuffer,
    weight_b: &MlxBuffer,
    weight_c: &MlxBuffer,
    output_a: &MlxBuffer,
    output_b: &MlxBuffer,
    output_c: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm triple: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    for (name, buf) in [
        ("input", input),
        ("output_a", output_a),
        ("output_b", output_b),
        ("output_c", output_c),
    ] {
        if buf.element_count() != expected {
            return Err(MlxError::InvalidArgument(format!(
                "RMS norm triple: {} element count {} != rows({}) * dim({})",
                name, buf.element_count(), rows, dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "RMS norm triple: {} must be f32, got {}",
                name, buf.dtype()
            )));
        }
    }
    for (name, buf) in [
        ("weight_a", weight_a),
        ("weight_b", weight_b),
        ("weight_c", weight_c),
    ] {
        if buf.element_count() != dim as usize {
            return Err(MlxError::InvalidArgument(format!(
                "RMS norm triple: {} element count {} != dim({})",
                name, buf.element_count(), dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "RMS norm triple: {} must be f32, got {}",
                name, buf.dtype()
            )));
        }
    }

    let pipeline = registry.get_pipeline("rms_norm_f32_triple", device)?;

    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, weight_a),
            (2, weight_b),
            (3, weight_c),
            (4, output_a),
            (5, output_b),
            (6, output_c),
            (7, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch the Wave P4.18 fused post-attention norm+add + triple pre-FF norm.
///
/// Replaces the two dispatches
///   (a) `fused_norm_add_f32(hidden, attn_out, post_attn_w -> pf_residual)`
///   (b) `rms_norm_f32_triple(pf_residual, w_a/b/c -> out_a/b/c)`
/// with one kernel that:
///   1. computes rms_inv over attn_out
///   2. writes `residual_out = hidden + attn_out * rms_inv * post_attn_w` AND
///      accumulates sum(residual_out^2) in the same pass
///   3. computes rms_inv over the residual stream
///   4. reads `residual_out` back, applies three weight vectors, writes 3 outputs
///
/// Eliminates one write+read of the residual buffer (~60 MB / layer @ pp2455
/// x 30 layers = ~1.8 GB of memory traffic) and one dispatch per layer.
///
/// # Arguments
/// * `encoder`      - Command encoder to record the dispatch into.
/// * `registry`     - Kernel registry (must have fused_post_attn_triple_norm_f32).
/// * `device`       - Metal device for pipeline compilation.
/// * `hidden`       - Pre-attention residual stream, f32 `[rows, dim]`.
/// * `attn_out`     - Attention O-proj output, f32 `[rows, dim]`.
/// * `post_attn_w`  - Post-attention layernorm weight, f32 `[dim]`.
/// * `weight_a`     - First pre-FF weight (e.g. MLP norm), f32 `[dim]`.
/// * `weight_b`     - Second pre-FF weight (e.g. MoE input norm), f32 `[dim]`.
/// * `weight_c`     - Third pre-FF weight (e.g. router norm), f32 `[dim]`.
/// * `residual_out` - Output: residual stream (hidden + normed_attn), f32 `[rows, dim]`.
/// * `output_a/b/c` - Triple-normed outputs, f32 `[rows, dim]`.
/// * `eps`          - RMS epsilon.
/// * `rows`         - Number of rows (= seq_len).
/// * `dim`          - Model hidden size.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_post_attn_triple_norm_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    hidden:       &MlxBuffer,
    attn_out:     &MlxBuffer,
    post_attn_w:  &MlxBuffer,
    weight_a:     &MlxBuffer,
    weight_b:     &MlxBuffer,
    weight_c:     &MlxBuffer,
    residual_out: &MlxBuffer,
    output_a:     &MlxBuffer,
    output_b:     &MlxBuffer,
    output_c:     &MlxBuffer,
    eps:          f32,
    rows:         u32,
    dim:          u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_post_attn_triple_norm_f32: rows and dim must be > 0".into(),
        ));
    }
    let expected = (rows as usize) * (dim as usize);
    for (name, buf) in [
        ("hidden",       hidden),
        ("attn_out",     attn_out),
        ("residual_out", residual_out),
        ("output_a",     output_a),
        ("output_b",     output_b),
        ("output_c",     output_c),
    ] {
        if buf.element_count() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_post_attn_triple_norm_f32: {} size {} < rows({}) * dim({})",
                name, buf.element_count(), rows, dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "fused_post_attn_triple_norm_f32: {} must be f32, got {}",
                name, buf.dtype()
            )));
        }
    }
    for (name, buf) in [
        ("post_attn_w", post_attn_w),
        ("weight_a",    weight_a),
        ("weight_b",    weight_b),
        ("weight_c",    weight_c),
    ] {
        if buf.element_count() != dim as usize {
            return Err(MlxError::InvalidArgument(format!(
                "fused_post_attn_triple_norm_f32: {} size {} != dim({})",
                name, buf.element_count(), dim
            )));
        }
    }

    // ADR-028: V2 (float4 + simd_sum) variant when dim % 4 == 0.
    // Default-OFF; opt-in via HF2Q_FUSED_TRIPLE_NORM_V2=1.  V1 was iter-186
    // (regressed -1.0% on decode); V2 saves 12 barriers/dispatch.
    let use_v2 = (dim % 4 == 0)
        && std::env::var("HF2Q_FUSED_TRIPLE_NORM_V2").ok().as_deref() == Some("1");
    let kernel_name = if use_v2 {
        "fused_post_attn_triple_norm_f32_v2"
    } else {
        "fused_post_attn_triple_norm_f32"
    };
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let tg_size = std::cmp::min(256u32, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = if use_v2 {
        // V2: 1 float per simdgroup (max 32 SGs at tg=1024).  Min 32 floats
        // so simd_sum's sgitg-indexed write is never OOB at partial-warp tgs.
        let n_sg = std::cmp::max(1u64, tg_size / 32);
        std::cmp::max(32, n_sg) * 4
    } else {
        tg_size * 4
    };

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params { eps: f32, dim: u32 }
    let params = Params { eps, dim };

    use super::encode_helpers::{as_bytes, KernelArg};
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0,  KernelArg::Buffer(hidden)),
            (1,  KernelArg::Buffer(attn_out)),
            (2,  KernelArg::Buffer(post_attn_w)),
            (3,  KernelArg::Buffer(weight_a)),
            (4,  KernelArg::Buffer(weight_b)),
            (5,  KernelArg::Buffer(weight_c)),
            (6,  KernelArg::Buffer(residual_out)),
            (7,  KernelArg::Buffer(output_a)),
            (8,  KernelArg::Buffer(output_b)),
            (9,  KernelArg::Buffer(output_c)),
            (10, KernelArg::Bytes(as_bytes(&params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// ADR-028 — Dispatch fused post-FF norm 2 + end-of-layer FINAL.
///
/// Fuses the gemma4 layer-end pair into a single kernel:
///   (a) mlp_down = attn_out + norm(moe_accum, w2)
///   (b) hidden   = (residual + norm(mlp_down, w3)) * layer_scalar
///
/// Bisect-confirmed +2.7% throughput on gemma4 default (iter-208 measured
/// 0.34 ms savings from eliminating the (b) launch).
///
/// # Arguments
/// * `encoder`      - Command encoder.
/// * `registry`     - Must have `fused_post_ff_norm2_endlayer_f32` registered.
/// * `device`       - Metal device.
/// * `attn_out`     - Post-attention residual stream, f32 [rows, dim].
/// * `moe_accum`    - MoE expert weighted sum, f32 [rows, dim].
/// * `residual`     - Pre-attention residual, f32 [rows, dim].
/// * `w2`           - post_feedforward_layernorm_2 weight, f32 [dim].
/// * `w3`           - post_feedforward_layernorm weight, f32 [dim].
/// * `layer_scalar` - Layer-scaling factor, f32 [1] (broadcast) or [dim] (per-channel).
/// * `mlp_down`     - Output: intermediate (also written for downstream compat), f32 [rows, dim].
/// * `hidden`       - Output: final layer output, f32 [rows, dim].
/// * `eps`          - RMS epsilon.
/// * `rows`         - Number of rows.
/// * `dim`          - Model hidden size.
/// * `scalar_is_vector` - True iff layer_scalar has dim elements (not 1).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_post_ff_norm2_endlayer_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    attn_out:     &MlxBuffer,
    moe_accum:    &MlxBuffer,
    residual:     &MlxBuffer,
    w2:           &MlxBuffer,
    w3:           &MlxBuffer,
    layer_scalar: &MlxBuffer,
    mlp_down:     &MlxBuffer,
    hidden:       &MlxBuffer,
    eps:          f32,
    rows:         u32,
    dim:          u32,
    scalar_is_vector: bool,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_post_ff_norm2_endlayer_f32: rows and dim must be > 0".into(),
        ));
    }
    let expected = (rows as usize) * (dim as usize);
    for (name, buf) in [
        ("attn_out",  attn_out),
        ("moe_accum", moe_accum),
        ("residual",  residual),
        ("mlp_down",  mlp_down),
        ("hidden",    hidden),
    ] {
        if buf.element_count() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_post_ff_norm2_endlayer_f32: {} size {} < rows({}) * dim({})",
                name, buf.element_count(), rows, dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "fused_post_ff_norm2_endlayer_f32: {} must be f32, got {}",
                name, buf.dtype()
            )));
        }
    }
    for (name, buf) in [("w2", w2), ("w3", w3)] {
        if buf.element_count() != dim as usize {
            return Err(MlxError::InvalidArgument(format!(
                "fused_post_ff_norm2_endlayer_f32: {} size {} != dim({})",
                name, buf.element_count(), dim
            )));
        }
    }
    let expected_scalar = if scalar_is_vector { dim as usize } else { 1 };
    if layer_scalar.element_count() < expected_scalar {
        return Err(MlxError::InvalidArgument(format!(
            "fused_post_ff_norm2_endlayer_f32: layer_scalar size {} < expected {}",
            layer_scalar.element_count(), expected_scalar
        )));
    }

    // ADR-028: V2 path (float4 + simd_sum) — same math, 75% fewer
    // barriers per dispatch (4 vs 16 at tg=256).  Requires `dim % 4 == 0`
    // (gemma4 hidden=3584 ✓).  Default-ON since parity-tested byte-identical
    // (max_abs=0, max_rel=0 at gemma4 hidden_dim across both scalar_is_vector
    // modes — see test_fused_post_ff_norm2_endlayer_v2_parity.rs) AND
    // measured +1.5% on gemma4 hybrid decode (74.9 → 76.0) + +0.7-1.0% on
    // legacy decode (73.7 → 74.2).  Opt-out via
    // `HF2Q_FUSED_POST_FF_NORM2_V2=0` / `=false` / `=off`.
    let use_v2 = (dim % 4 == 0) && cached_env_default_true(&CACHED_FUSED_POST_FF_NORM2_V2, "HF2Q_FUSED_POST_FF_NORM2_V2");
    let kernel_name = if use_v2 {
        "fused_post_ff_norm2_endlayer_f32_v2"
    } else {
        "fused_post_ff_norm2_endlayer_f32"
    };
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let tg_size = std::cmp::min(256u32, dim.next_power_of_two()) as u64;
    // V2 needs only one float per simdgroup for cross-SG reduction
    // (≤ 32 SGs at tg ≤ 1024 → ≤ 128 bytes).  V1 uses tg_size floats for
    // the tree reduction.
    let shared_mem_bytes = if use_v2 {
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    };

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        eps: f32,
        dim: u32,
        scalar_is_vector: u32,
    }
    let params = Params {
        eps,
        dim,
        scalar_is_vector: if scalar_is_vector { 1 } else { 0 },
    };

    use super::encode_helpers::{as_bytes, KernelArg};
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(attn_out)),
            (1, KernelArg::Buffer(moe_accum)),
            (2, KernelArg::Buffer(residual)),
            (3, KernelArg::Buffer(w2)),
            (4, KernelArg::Buffer(w3)),
            (5, KernelArg::Buffer(layer_scalar)),
            (6, KernelArg::Buffer(mlp_down)),
            (7, KernelArg::Buffer(hidden)),
            (8, KernelArg::Bytes(as_bytes(&params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch the fully-fused MoE-wsum + Path A end-of-layer kernel.
///
/// Replaces the dispatch chain:
///   1. moe_weighted_sum: moe_down_id_out × routing_weights → moe_accum
///   2. fused_post_ff_norm2_endlayer_v2: attn_out + moe_accum + residual + ...
///                                       → mlp_down + hidden
///
/// With ONE dispatch using V2 (float4 + simd_sum) reduction pattern.
/// Saves 1 dispatch/layer + 1 moe_accum global memory round-trip.
///
/// # Arguments
/// * `expert_outputs`  - MoE down outputs, f32 [rows, top_k, dim].
/// * `routing_weights` - Routing weights, f32 [rows, top_k].
/// * `attn_out`        - Post-attention residual stream, f32 [rows, dim].
/// * `residual`        - Pre-attention residual, f32 [rows, dim].
/// * `w2`              - post_feedforward_layernorm_2 weight, f32 [dim].
/// * `w3`              - post_feedforward_layernorm weight, f32 [dim].
/// * `layer_scalar`    - Layer scaling, f32 [1] or [dim].
/// * `mlp_down`        - Output: combined MLP+MoE post-norm, f32 [rows, dim].
/// * `hidden`          - Output: final layer hidden, f32 [rows, dim].
/// * `eps`             - RMS epsilon.
/// * `rows`            - Number of rows.
/// * `dim`             - Hidden size (must be % 4 == 0 for V2).
/// * `top_k`           - MoE top-K.
/// * `scalar_is_vector`- True iff layer_scalar is per-channel.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_moe_wsum_post_ff_norm2_endlayer_f32_v2(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    expert_outputs:  &MlxBuffer,
    routing_weights: &MlxBuffer,
    attn_out:        &MlxBuffer,
    residual:        &MlxBuffer,
    w2:              &MlxBuffer,
    w3:              &MlxBuffer,
    layer_scalar:    &MlxBuffer,
    mlp_down:        &MlxBuffer,
    hidden:          &MlxBuffer,
    eps:             f32,
    rows:            u32,
    dim:             u32,
    top_k:           u32,
    scalar_is_vector: bool,
) -> Result<()> {
    if rows == 0 || dim == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_moe_wsum_post_ff_norm2_endlayer_v2: rows, dim, top_k must be > 0".into(),
        ));
    }
    if dim % 4 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_moe_wsum_post_ff_norm2_endlayer_v2: dim {} must be % 4 == 0", dim
        )));
    }
    let row_elems = (rows as usize) * (dim as usize);
    for (name, buf) in [
        ("attn_out", attn_out),
        ("residual", residual),
        ("mlp_down", mlp_down),
        ("hidden",   hidden),
    ] {
        if buf.element_count() < row_elems {
            return Err(MlxError::InvalidArgument(format!(
                "{}: size {} < rows({}) * dim({})", name, buf.element_count(), rows, dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "{}: must be f32, got {}", name, buf.dtype()
            )));
        }
    }
    let exp_required = (rows as usize) * (top_k as usize) * (dim as usize);
    if expert_outputs.element_count() < exp_required {
        return Err(MlxError::InvalidArgument(format!(
            "expert_outputs: size {} < rows({}) * top_k({}) * dim({})",
            expert_outputs.element_count(), rows, top_k, dim
        )));
    }
    let w_required = (rows as usize) * (top_k as usize);
    if routing_weights.element_count() < w_required {
        return Err(MlxError::InvalidArgument(format!(
            "routing_weights: size {} < rows({}) * top_k({})",
            routing_weights.element_count(), rows, top_k
        )));
    }
    for (name, buf) in [("w2", w2), ("w3", w3)] {
        if buf.element_count() != dim as usize {
            return Err(MlxError::InvalidArgument(format!(
                "{}: size {} != dim({})", name, buf.element_count(), dim
            )));
        }
    }
    let expected_scalar = if scalar_is_vector { dim as usize } else { 1 };
    if layer_scalar.element_count() < expected_scalar {
        return Err(MlxError::InvalidArgument(format!(
            "layer_scalar: size {} < expected {}",
            layer_scalar.element_count(), expected_scalar
        )));
    }

    let pipeline = registry.get_pipeline("fused_moe_wsum_post_ff_norm2_endlayer_f32_v2", device)?;

    let tg_size = std::cmp::min(256u32, dim.next_power_of_two()) as u64;
    // Threadgroup memory: max(32, n_sg) (SG scratch) + dim (sum_buf) floats.
    let n_sg = (tg_size / 32).max(1);
    let sg_scratch_floats = n_sg.max(32);
    let shared_mem_bytes = (sg_scratch_floats + dim as u64) * 4;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        eps: f32,
        dim: u32,
        top_k: u32,
        scalar_is_vector: u32,
    }
    let params = Params {
        eps,
        dim,
        top_k,
        scalar_is_vector: if scalar_is_vector { 1 } else { 0 },
    };

    use super::encode_helpers::{as_bytes, KernelArg};
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(expert_outputs)),
            (1, KernelArg::Buffer(routing_weights)),
            (2, KernelArg::Buffer(attn_out)),
            (3, KernelArg::Buffer(residual)),
            (4, KernelArg::Buffer(w2)),
            (5, KernelArg::Buffer(w3)),
            (6, KernelArg::Buffer(layer_scalar)),
            (7, KernelArg::Buffer(mlp_down)),
            (8, KernelArg::Buffer(hidden)),
            (9, KernelArg::Bytes(as_bytes(&params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Dispatch an RMS normalization without learned scale (bf16 only).
///
/// Computes: `output = x * rsqrt(mean(x^2) + eps)` — no weight multiplication.
/// Used for per-head V normalization in Gemma 4.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm_no_scale_bf16 registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (bf16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
pub fn dispatch_rms_norm_no_scale_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm no_scale: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale: output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }

    // hf2q ADR-030 —defense-in-depth dtype check.  This
    // dispatcher hardcodes the BF16 kernel; passing F32 buffers would
    // mis-stride reads/writes.
    if input.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_no_scale_bf16: input must be BF16, got {}",
            input.dtype(),
        )));
    }
    if output.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_no_scale_bf16: output must be BF16, got {}",
            output.dtype(),
        )));
    }

    let pipeline = registry.get_pipeline("rms_norm_no_scale_bf16", device)?;

    // One threadgroup per row.  Threadgroup size must be a power of 2
    // for the tree reduction to work correctly.
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;

    // Threadgroup shared memory: tg_size floats for the reduction.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch an RMS normalization without learned scale (f32).
///
/// Computes: `output = x * rsqrt(mean(x^2) + eps)` -- no weight multiplication.
/// Used for per-head V normalization in Gemma 4 when activations are f32.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm_no_scale_f32 registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (f32).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
pub fn dispatch_rms_norm_no_scale_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm no_scale f32: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale f32: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale f32: output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }

    // hf2q ADR-030 —defense-in-depth dtype check.  This
    // dispatcher hardcodes the F32 kernel; passing BF16/F16 buffers
    // would mis-stride reads/writes.
    if input.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_no_scale_f32: input must be F32, got {}",
            input.dtype(),
        )));
    }
    if output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_no_scale_f32: output must be F32, got {}",
            output.dtype(),
        )));
    }

    // ADR-028 —float4 + simd_sum variant.  Default-ON since
    // iter-326 (operator REFRAME #2).  Opt out with
    // `HF2Q_RMS_NORM_V2=0` / `=false` / `=off`.
    let use_v2 = (dim % 4 == 0) && cached_env_default_true(&CACHED_RMS_NORM_V2, "HF2Q_RMS_NORM_V2");
    let kernel_name = if use_v2 {
        "rms_norm_no_scale_f32_v2"
    } else {
        "rms_norm_no_scale_f32"
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // ADR-028 iter-361 — see `dispatch_rms_norm` comment above; right-sizing
    // tg to float4 element count was FALSIFIED at -0.4%.  Keep tg = min(256,
    // dim.next_power_of_two()) — same as the legacy path.
    let mut tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    if use_v2 && tg_size < 32 {
        tg_size = 32;
    }
    let shared_mem_bytes = if use_v2 {
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    };

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch a fused RMS normalization + elementwise multiply.
///
/// Computes: `output = (input * rsqrt(mean(input^2) + eps) * weight) * scale`
///
/// This replaces the two-dispatch pattern:
///   1. `rms_norm(input, weight) -> tmp`
///   2. `elementwise_mul(tmp, scale) -> output`
///
/// with a single kernel pass, eliminating one barrier and one global memory
/// round-trip for the intermediate `tmp` buffer.
///
/// # Arguments
///
/// * `encoder`      - Command encoder to record the dispatch into.
/// * `registry`     - Kernel registry.
/// * `device`       - Metal device for pipeline compilation.
/// * `input`        - Input buffer of shape `[rows, dim]`.
/// * `norm_weight`  - Norm weight buffer of shape `[dim]`.
/// * `scale_weight` - Scale (MUL operand) buffer of shape `[rows, dim]`.
/// * `output`       - Output buffer of shape `[rows, dim]`.
/// * `params_buf`   - Params buffer containing `[eps, dim]` as f32.
/// * `rows`         - Number of rows.
/// * `dim`          - Dimension of the last axis.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rms_norm_mul(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    scale_weight: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "Fused RMS norm+mul: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }

    // hf2q ADR-030 —defense-in-depth dtype-coherence check.
    // fused_rms_norm_mul_{f32,bf16} kernels declare norm_weight, scale,
    // and output at the input-dtype stride.  Mismatched buffers would
    // silently mis-stride (iter-106 signature).  Same pattern as the
    // iter-110 dispatch_rms_norm guard.
    if norm_weight.dtype() != input.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul dtype mismatch: input={} != norm_weight={} \
             (kernel rms_norm_mul_{} reads norm_weight at input-dtype stride)",
            input.dtype(), norm_weight.dtype(),
            match input.dtype() {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => "?",
            },
        )));
    }
    if scale_weight.dtype() != input.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul dtype mismatch: input={} != scale={}",
            input.dtype(), scale_weight.dtype(),
        )));
    }
    if output.dtype() != input.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul dtype mismatch: input={} != output={}",
            input.dtype(), output.dtype(),
        )));
    }

    let kernel_name = fused_rms_norm_mul_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, norm_weight),
            (2, scale_weight),
            (3, output),
            (4, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
