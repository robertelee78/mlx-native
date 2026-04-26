//! 2-D NeoX RoPE for ViT vision towers (Gemma 4 Vision).
//!
//! The head_dim splits in half: the first half rotates by `pos_x` and the
//! second half by `pos_y`, each NeoX-style with its own d-axis schedule.
//!
//! Mirrors `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:46-91`:
//!
//! ```text
//! first  = ggml_rope_ext(view[0..n_dim/2],   pos_x, n_dim/2, NEOX, ...)
//! second = ggml_rope_ext(view[n_dim/2..],    pos_y, n_dim/2, NEOX, ...)
//! cur    = ggml_concat(first, second, dim=0)
//! ```
//!
//! The kernel performs both rotations in one dispatch — each thread
//! handles ONE pair from the first half AND ONE pair from the second half
//! (so the grid has `d_quarter = head_dim / 4` threads in the X axis).
//!
//! # Constraints
//!
//! - `head_dim % 4 == 0` (so `d_half` and `d_quarter` are both integers
//!   and a clean NeoX pair-split exists in each half).
//! - `n_heads, seq_len > 0`.
//! - `pos_x` / `pos_y` lengths match `seq_len`.
//!
//! # Notes
//!
//! Unlike `rope_neox_*`, this kernel uses `d_half` (not full `head_dim`)
//! as the theta denominator, matching the candle and ggml references where
//! each half is its own rotation domain with `n_dims = head_dim / 2`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the vision 2-D RoPE kernels (embedded at compile time).
pub static VISION_2D_ROPE_SHADER_SOURCE: &str =
    include_str!("../shaders/vision_2d_rope.metal");

/// Register vision_2d_rope shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("vision_2d_rope_f32", VISION_2D_ROPE_SHADER_SOURCE);
    registry.register_source("vision_2d_rope_bf16", VISION_2D_ROPE_SHADER_SOURCE);
}

/// Dispatch a 2-D NeoX RoPE for a ViT vision tower.
///
/// # Arguments
///
/// * `encoder`     - Command encoder to record the dispatch into.
/// * `registry`    - Kernel registry (must have vision_2d_rope sources registered).
/// * `device`      - Metal device for pipeline compilation.
/// * `input`       - Input buffer of shape `[seq_len * n_heads, head_dim]` (f32 or bf16).
/// * `output`      - Output buffer (same shape and dtype as input).
/// * `params_buf`  - Params buffer containing `[theta, head_dim_f, n_heads_f, 0]` as f32.
/// * `pos_x`       - Positions buffer for first-half rotation (u32, shape `[seq_len]`).
/// * `pos_y`       - Positions buffer for second-half rotation (u32, shape `[seq_len]`).
/// * `seq_len`     - Number of sequence positions (= num_patches for ViT).
/// * `n_heads`     - Number of attention heads (per-batch row).
/// * `head_dim`    - Dimension of each head (must be divisible by 4).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `head_dim` is not divisible by 4.
/// - Any zero-sized dim.
/// - Buffers are too small.
/// - Input dtype is not F32 or BF16.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_vision_2d_rope(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    pos_x: &MlxBuffer,
    pos_y: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<()> {
    if head_dim == 0 || seq_len == 0 || n_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "vision_2d_rope: head_dim, seq_len, n_heads must all be > 0".into(),
        ));
    }
    if head_dim % 4 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "vision_2d_rope: head_dim ({}) must be divisible by 4 (need clean d_half/d_quarter split)",
            head_dim
        )));
    }

    let n_rows = (seq_len as usize) * (n_heads as usize);
    let elements = n_rows * (head_dim as usize);
    if input.element_count() != elements {
        return Err(MlxError::InvalidArgument(format!(
            "vision_2d_rope: input element count {} != seq_len({}) * n_heads({}) * head_dim({}) = {}",
            input.element_count(),
            seq_len,
            n_heads,
            head_dim,
            elements
        )));
    }
    if output.element_count() != elements {
        return Err(MlxError::InvalidArgument(format!(
            "vision_2d_rope: output element count {} != {}",
            output.element_count(),
            elements
        )));
    }
    if input.dtype() != output.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "vision_2d_rope: input/output dtype mismatch {} vs {}",
            input.dtype(),
            output.dtype()
        )));
    }

    let expected_pos = seq_len as usize;
    if pos_x.element_count() != expected_pos {
        return Err(MlxError::InvalidArgument(format!(
            "vision_2d_rope: pos_x length {} != seq_len {}",
            pos_x.element_count(),
            seq_len
        )));
    }
    if pos_y.element_count() != expected_pos {
        return Err(MlxError::InvalidArgument(format!(
            "vision_2d_rope: pos_y length {} != seq_len {}",
            pos_y.element_count(),
            seq_len
        )));
    }
    match pos_x.dtype() {
        DType::U32 | DType::I32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "vision_2d_rope: pos_x must be u32 or i32 (got {})",
                other
            )));
        }
    }
    match pos_y.dtype() {
        DType::U32 | DType::I32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "vision_2d_rope: pos_y must be u32 or i32 (got {})",
                other
            )));
        }
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "vision_2d_rope_f32",
        DType::BF16 => "vision_2d_rope_bf16",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "vision_2d_rope: unsupported dtype {}",
                other
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let d_quarter = head_dim / 4;
    let n_rows_u32 = n_rows as u32;

    // Grid: (d_quarter, n_rows) — each thread does ONE pair in each half.
    let tg_x = std::cmp::min(64, d_quarter as u64).max(1);
    let tg_y = std::cmp::min(4, n_rows_u32 as u64).max(1);

    encoder.encode(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
            (3, pos_x),
            (4, pos_y),
        ],
        MTLSize::new(d_quarter as u64, n_rows_u32 as u64, 1),
        MTLSize::new(tg_x, tg_y, 1),
    );

    Ok(())
}

/// Convenience: build the params buffer for a `dispatch_vision_2d_rope` call.
///
/// Layout: `[theta_base, head_dim_f, n_heads_f, 0]` as f32 (matches the
/// `params` buffer the kernel reads at `buffer(2)`).
pub fn build_vision_2d_rope_params(
    device: &crate::MlxDevice,
    theta: f32,
    head_dim: u32,
    n_heads: u32,
) -> Result<MlxBuffer> {
    let mut params = device.alloc_buffer(4 * 4, DType::F32, vec![4])?;
    {
        let s = params.as_mut_slice::<f32>()?;
        s[0] = theta;
        s[1] = head_dim as f32;
        s[2] = n_heads as f32;
        s[3] = 0.0;
    }
    Ok(params)
}
