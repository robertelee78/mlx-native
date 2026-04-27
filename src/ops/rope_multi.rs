//! Multi-section Rotary Position Embedding with optional interleaved mode.
//!
//! Used by Qwen3.5 / Qwen3.6 full-attention layers (ADR-013 Decision 10).
//! Both MROPE (`mode = 8`) and IMROPE (`mode = 40`) share a kernel; only the
//! sector-to-axis mapping differs.
//!
//! # Spec (summary)
//!
//! For every pair `p ∈ [0, rope_dim/2)`:
//! 1. `sector = p mod (s0 + s1 + s2 + s3)`
//! 2. Pick axis based on `mode`:
//!    * `Mrope`: contiguous sections — `sector ∈ [0, s0)` → axis 0, etc.
//!    * `Imrope`: `sector % 3` cycling — `sector % 3 == 0 && sector < 3*s0`
//!      → axis 0; `== 1 && sector < 3*s1` → axis 1; `== 2 && sector < 3*s2`
//!      → axis 2; else axis 3.
//! 3. `theta = position[axis] * freq_base^(-2p/rope_dim)`
//! 4. Rotate pair `(x[p], x[p + head_dim/2])` by that angle (NeoX indexing).
//!
//! Pairs `p ≥ rope_dim/2` pass through unchanged (partial-rotary-factor).
//!
//! # Positions layout
//!
//! The `positions` buffer is an `int32` array of length `4 * seq_len`:
//! first `seq_len` entries are the time-axis positions, next `seq_len` are
//! the height-axis, then width, then extra. For Qwen3.5 text, all four
//! axes are set to the token's 1D position.

use std::cell::RefCell;
use std::collections::HashMap;

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static ROPE_MULTI_SHADER_SOURCE: &str = include_str!("../shaders/rope_multi.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("rope_multi_f32", ROPE_MULTI_SHADER_SOURCE);
    registry.register_source("rope_multi_bf16", ROPE_MULTI_SHADER_SOURCE);
}

/// MROPE variant. Wire-level values match the ggml `GGML_ROPE_TYPE_*` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RopeMultiMode {
    /// Standard multi-section RoPE; contiguous sections.
    Mrope = 8,
    /// Interleaved multi-section RoPE; `sector % 3` cycles through 3 axes.
    /// Used by Qwen3.5 / Qwen3.6.
    Imrope = 40,
}

/// Shape + config for a rope_multi dispatch.
#[derive(Debug, Clone, Copy)]
pub struct RopeMultiParams {
    pub head_dim: u32,
    pub rope_dim: u32, // must be <= head_dim; must be even
    pub n_heads: u32,
    pub seq_len: u32,
    pub freq_base: f32,
    pub mode: RopeMultiMode,
    /// Section counts `[s0, s1, s2, s3]`. Sum should be `rope_dim / 2` for
    /// full coverage; the kernel tolerates smaller sums (sector wraps).
    pub sections: [u32; 4],
}

fn validate(
    p: &RopeMultiParams,
    input: &MlxBuffer,
    output: &MlxBuffer,
    positions: &MlxBuffer,
) -> Result<()> {
    if p.head_dim == 0 || p.rope_dim == 0 || p.n_heads == 0 || p.seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "rope_multi: head_dim, rope_dim, n_heads, seq_len must all be > 0".into(),
        ));
    }
    if p.head_dim % 2 != 0 || p.rope_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(
            "rope_multi: head_dim and rope_dim must be even".into(),
        ));
    }
    if p.rope_dim > p.head_dim {
        return Err(MlxError::InvalidArgument(
            "rope_multi: rope_dim must be <= head_dim".into(),
        ));
    }
    if !p.freq_base.is_finite() || p.freq_base <= 0.0 {
        return Err(MlxError::InvalidArgument(format!(
            "rope_multi: freq_base must be finite and positive, got {}",
            p.freq_base
        )));
    }

    let n_rows = (p.seq_len as usize) * (p.n_heads as usize);
    let elements = n_rows * (p.head_dim as usize);
    if input.element_count() != elements {
        return Err(MlxError::InvalidArgument(format!(
            "rope_multi: input element count {} != seq_len({}) * n_heads({}) * head_dim({}) = {}",
            input.element_count(),
            p.seq_len,
            p.n_heads,
            p.head_dim,
            elements
        )));
    }
    if output.element_count() != elements {
        return Err(MlxError::InvalidArgument(format!(
            "rope_multi: output element count {} != {}",
            output.element_count(),
            elements
        )));
    }
    if input.dtype() != output.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "rope_multi: input/output dtype mismatch {} vs {}",
            input.dtype(),
            output.dtype()
        )));
    }

    let expected_positions = 4 * (p.seq_len as usize);
    if positions.element_count() != expected_positions {
        return Err(MlxError::InvalidArgument(format!(
            "rope_multi: positions length {} != 4 * seq_len({}) = {}",
            positions.element_count(),
            p.seq_len,
            expected_positions
        )));
    }
    match positions.dtype() {
        DType::I32 | DType::U32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "rope_multi: positions must be i32 or u32 (got {})",
                other
            )));
        }
    }

    Ok(())
}

/// Dispatch a rope_multi operation.
///
/// The caller must upload:
/// - `params_buf`: float4 `[freq_base, head_dim, rope_dim, 0]`.
/// - `rope_params_buf`: uint4 `[n_heads, mode_code, seq_len, 0]`. The
///   `mode_code` is the `u32` underlying [`RopeMultiMode`].
/// - `sections_buf`: uint4 `[s0, s1, s2, s3]`.
/// - `positions`: int32 array of length `4 * seq_len`.
///
/// The helper [`build_rope_multi_buffers`] constructs all three small buffers
/// in one call for callers that do not already keep them pooled.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_multi(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    positions: &MlxBuffer,
    params_buf: &MlxBuffer,
    rope_params_buf: &MlxBuffer,
    sections_buf: &MlxBuffer,
    p: RopeMultiParams,
) -> Result<()> {
    validate(&p, input, output, positions)?;

    let kernel_name = match input.dtype() {
        DType::F32 => "rope_multi_f32",
        DType::BF16 => "rope_multi_bf16",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "rope_multi: unsupported dtype {}",
                other
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let half_dim = p.head_dim / 2;
    let n_rows = p.seq_len * p.n_heads;

    // Grid: (half_dim, n_rows). Every thread writes a NeoX pair.
    let grid = MTLSize::new(half_dim as u64, n_rows as u64, 1);

    let tg_x = std::cmp::min(half_dim, 256).max(1);
    let remain = (256u32 / tg_x).max(1);
    let tg_y = std::cmp::min(n_rows, remain).max(1);
    let tg = MTLSize::new(tg_x as u64, tg_y as u64, 1);

    encoder.encode(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
            (3, positions),
            (4, rope_params_buf),
            (5, sections_buf),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// Pre-built triple of small parameter buffers for a `rope_multi` dispatch.
///
/// Held in the per-thread [`ROPE_PACK_CACHE`] so callers that issue
/// repeated dispatches with stable shape (the qwen35 / qwen36 decode hot
/// path: identical `head_dim`, `rope_dim`, `n_heads`, `seq_len=1`,
/// `freq_base`, `mode`, `sections` every step) skip the per-call
/// allocation triplet (~208 µs/token measured on M5 Max in
/// `cfa-20260426-adr015-wave2a-p3aprime`).  Decode-out-of-scope cases
/// (variable `seq_len`) populate one entry per `seq_len` value seen,
/// then reuse on re-encounter.
pub struct RopeMultiBufferPack {
    pub params_buf: MlxBuffer,
    pub rope_params_buf: MlxBuffer,
    pub sections_buf: MlxBuffer,
}

/// Cache key for [`ROPE_PACK_CACHE`].  Includes the [`MlxDevice`] pointer
/// so two consecutive sessions with different devices (e.g. model swap)
/// never share entries — required for correctness, not just isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RopeMultiCacheKey {
    device_ptr: usize,
    head_dim: u32,
    rope_dim: u32,
    n_heads: u32,
    seq_len: u32,
    freq_base_bits: u32,
    mode: u32,
    sections: [u32; 4],
}

impl RopeMultiCacheKey {
    fn from_params(device: &crate::MlxDevice, p: &RopeMultiParams) -> Self {
        Self {
            device_ptr: device as *const _ as usize,
            head_dim: p.head_dim,
            rope_dim: p.rope_dim,
            n_heads: p.n_heads,
            seq_len: p.seq_len,
            freq_base_bits: p.freq_base.to_bits(),
            mode: p.mode as u32,
            sections: p.sections,
        }
    }
}

thread_local! {
    /// Per-thread cache of pre-built `rope_multi` parameter buffers,
    /// keyed by [`RopeMultiCacheKey`].  Built lazily on first
    /// [`dispatch_rope_multi_cached`] call for a given key, retained for
    /// the thread's lifetime (cleared by [`clear_rope_pack_cache`] in
    /// tests / on explicit model unload).
    static ROPE_PACK_CACHE: RefCell<HashMap<RopeMultiCacheKey, RopeMultiBufferPack>> =
        RefCell::new(HashMap::new());
}

/// Clear the thread-local rope-multi pack cache.
///
/// Useful between model loads (the cache key includes the device pointer
/// so old entries can never be returned for a new device, but they leak
/// memory until cleared) and in test suites that swap mocked devices.
pub fn clear_rope_pack_cache() {
    ROPE_PACK_CACHE.with(|cell| cell.borrow_mut().clear());
}

/// Inspect the current pack-cache size — diagnostic only.
pub fn rope_pack_cache_len() -> usize {
    ROPE_PACK_CACHE.with(|cell| cell.borrow().len())
}

/// Dispatch a `rope_multi` operation, reusing pre-built parameter
/// buffers from the per-thread cache.
///
/// Functionally equivalent to [`dispatch_rope_multi`] preceded by
/// [`build_rope_multi_buffers`], but the small param/rope_params/sections
/// buffers (3 × 16 bytes each) are built once per
/// `(device, head_dim, rope_dim, n_heads, seq_len, freq_base, mode,
/// sections)` tuple and reused on every subsequent call.  See
/// `docs/ADR-015-mlx-native-single-cb-decode.md` §"P3a' live profile pass"
/// rank-4 finding — `build_rope_multi_buffers` per-call alloc was
/// measured at 208 µs/token on the qwen3.6-27b-dwq46 dense-FFN-Q hot
/// path (16 FullAttn layers × 2 calls/layer = 32 calls/token, each
/// allocating 3 fresh `MlxBuffer`s via Mach IPC).  This helper closes
/// that residual.
///
/// Bit-exact to the per-call form: identical kernel, identical inputs,
/// only the small parameter triplet is sourced from the cache.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_multi_cached(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &crate::MlxDevice,
    input: &MlxBuffer,
    output: &MlxBuffer,
    positions: &MlxBuffer,
    p: RopeMultiParams,
) -> Result<()> {
    let key = RopeMultiCacheKey::from_params(device, &p);
    ROPE_PACK_CACHE.with(|cell| {
        let mut map = cell.borrow_mut();
        if !map.contains_key(&key) {
            let (params_buf, rope_params_buf, sections_buf) =
                build_rope_multi_buffers(device, p)?;
            map.insert(
                key,
                RopeMultiBufferPack {
                    params_buf,
                    rope_params_buf,
                    sections_buf,
                },
            );
        }
        let pack = map
            .get(&key)
            .expect("inserted above if missing; cache is single-threaded");
        dispatch_rope_multi(
            encoder,
            registry,
            device.metal_device(),
            input,
            output,
            positions,
            &pack.params_buf,
            &pack.rope_params_buf,
            &pack.sections_buf,
            p,
        )
    })
}

/// Convenience: build all three small parameter buffers given a [`RopeMultiParams`].
///
/// Returns `(params_buf, rope_params_buf, sections_buf)`.
pub fn build_rope_multi_buffers(
    device: &crate::MlxDevice,
    p: RopeMultiParams,
) -> Result<(MlxBuffer, MlxBuffer, MlxBuffer)> {
    let mut params = device.alloc_buffer(4 * 4, DType::F32, vec![4])?;
    {
        let s = params.as_mut_slice::<f32>()?;
        s[0] = p.freq_base;
        s[1] = p.head_dim as f32;
        s[2] = p.rope_dim as f32;
        s[3] = 0.0;
    }
    let mut rope_params = device.alloc_buffer(4 * 4, DType::U32, vec![4])?;
    {
        let s = rope_params.as_mut_slice::<u32>()?;
        s[0] = p.n_heads;
        s[1] = p.mode as u32;
        s[2] = p.seq_len;
        s[3] = 0;
    }
    let mut sections = device.alloc_buffer(4 * 4, DType::U32, vec![4])?;
    {
        let s = sections.as_mut_slice::<u32>()?;
        s[0] = p.sections[0];
        s[1] = p.sections[1];
        s[2] = p.sections[2];
        s[3] = p.sections[3];
    }
    Ok((params, rope_params, sections))
}
