//! Decode-only fused Gated DeltaNet kernel — `simd_sum`-based variant.
//!
//! ADR-015 iter56. Mirrors llama.cpp's `kernel_gated_delta_net_f32_<NSG>`
//! threading model (32-lane simdgroup reductions, NSG state cells per
//! thread, no shared memory, no threadgroup barriers) — drop-in replacement
//! for [`super::gated_delta_net::dispatch_gated_delta_net`] when n_tokens is
//! small (decode regime, n_tokens=1).
//!
//! # When to use this vs `dispatch_gated_delta_net`
//!
//! * **Decode (n_tokens=1)** — use this. The existing
//!   `gated_delta_net_f32` uses 128-thread threadgroups + threadgroup memory
//!   + barriers, which on Apple GPU spills the 128-element private state row
//!   to threadgroup memory and bottlenecks on barrier stalls between every
//!   token. With NSG=4 simdgroup reductions the state row sits in registers
//!   (NSG=4 cells × 4 bytes = 16 bytes/thread) and reductions are
//!   single-cycle warp shuffles.
//! * **Prefill (n_tokens > 32 or so)** — keep using `dispatch_gated_delta_net`.
//!   The existing kernel's `output[i]` per-thread amortizes better when the
//!   state-update arithmetic dominates barrier overhead.
//!
//! Both kernels produce **bit-equivalent** output (modulo floating-point
//! associativity in the cross-lane reductions); the
//! `gated_delta_net_decode_parity` test in `tests/` enforces this on
//! representative shapes.
//!
//! # Buffer / params contract
//!
//! Identical to [`super::gated_delta_net::dispatch_gated_delta_net`]:
//! * `q` — pre-scaled by `1/sqrt(D_k)` (caller's responsibility)
//! * `k`, `v`, `g`, `beta`, `state_in`, `output`, `state_out` — same shapes
//! * `params_buf` — same 8-u32 layout, built via
//!   [`super::gated_delta_net::build_gated_delta_net_params`]
//!
//! # NSG selection
//!
//! `NSG = D_k / 32`. Currently supports `NSG ∈ {1, 2, 4}` covering
//! `D_k ∈ {32, 64, 128}` — the production Qwen3.5/3.6 head dim is 128 → NSG=4.
//! `D_k` MUST be a multiple of 32 and `<= 128`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::gated_delta_net::GatedDeltaNetParams;

pub static GATED_DELTA_NET_DECODE_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_decode.metal");

/// Hard cap: D_k must be ≤ 32 * MAX_NSG. NSG=4 → 128.
pub const MAX_NSG: u32 = 4;

/// Register the decode kernel under all three NSG-templated names.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "gated_delta_net_decode_f32_1",
        GATED_DELTA_NET_DECODE_SHADER_SOURCE,
    );
    registry.register_source(
        "gated_delta_net_decode_f32_2",
        GATED_DELTA_NET_DECODE_SHADER_SOURCE,
    );
    registry.register_source(
        "gated_delta_net_decode_f32_4",
        GATED_DELTA_NET_DECODE_SHADER_SOURCE,
    );
}

fn validate(
    p: &GatedDeltaNetParams,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g: &MlxBuffer,
    beta: &MlxBuffer,
    state_in: &MlxBuffer,
    output: &MlxBuffer,
    state_out: &MlxBuffer,
) -> Result<()> {
    if p.d_k == 0 || p.d_v == 0 || p.n_k_heads == 0 || p.n_v_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_decode: dims must all be > 0".into(),
        ));
    }
    if p.n_tokens == 0 || p.n_seqs == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_decode: n_tokens and n_seqs must be > 0".into(),
        ));
    }
    if p.n_v_heads % p.n_k_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: n_v_heads ({}) must be a multiple of n_k_heads ({})",
            p.n_v_heads, p.n_k_heads
        )));
    }
    if p.d_k % 32 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: D_k ({}) must be a multiple of 32 (simdgroup width)",
            p.d_k
        )));
    }
    if p.d_k / 32 > MAX_NSG {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: D_k ({}) implies NSG > {} — extend the .metal kernel",
            p.d_k, MAX_NSG
        )));
    }
    if p.d_v == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_decode: D_v must be > 0".into(),
        ));
    }
    let nsg = p.d_k / 32;
    if p.d_v % nsg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: D_v ({}) must be a multiple of NSG ({}=D_k/32)",
            p.d_v, nsg
        )));
    }

    let qk_elems = (p.d_k as usize)
        * (p.n_k_heads as usize)
        * (p.n_tokens as usize)
        * (p.n_seqs as usize);
    let v_elems = (p.d_v as usize)
        * (p.n_v_heads as usize)
        * (p.n_tokens as usize)
        * (p.n_seqs as usize);
    let scalar_elems = (p.n_v_heads as usize) * (p.n_tokens as usize) * (p.n_seqs as usize);
    let state_elems =
        (p.d_k as usize) * (p.d_v as usize) * (p.n_v_heads as usize) * (p.n_seqs as usize);

    for (name, buf, exp) in [
        ("q", q, qk_elems),
        ("k", k, qk_elems),
        ("v", v, v_elems),
        ("output", output, v_elems),
        ("g", g, scalar_elems),
        ("beta", beta, scalar_elems),
        ("state_in", state_in, state_elems),
        ("state_out", state_out, state_elems),
    ] {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode: {} must be f32 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    Ok(())
}

/// Dispatch the decode-only fused Gated DeltaNet kernel.
///
/// Equivalent math to [`super::gated_delta_net::dispatch_gated_delta_net`];
/// uses `simd_sum` reductions instead of threadgroup_barrier+shared memory.
/// Threadgroup `(32, NSG, 1)`; grid `(D_v / NSG, n_v_heads, n_seqs)`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_decode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g: &MlxBuffer,
    beta: &MlxBuffer,
    state_in: &MlxBuffer,
    output: &MlxBuffer,
    state_out: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: GatedDeltaNetParams,
) -> Result<()> {
    validate(&p, q, k, v, g, beta, state_in, output, state_out)?;

    let nsg: u32 = p.d_k / 32;
    let kernel_name = match nsg {
        1 => "gated_delta_net_decode_f32_1",
        2 => "gated_delta_net_decode_f32_2",
        4 => "gated_delta_net_decode_f32_4",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode: unsupported NSG={} (D_k={})",
                other, p.d_k
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // Threadgroup: (32, NSG, 1) = 32*NSG threads, NSG simdgroups.
    let tg = MTLSize::new(32, nsg as u64, 1);
    // Grid (in threadgroups): (D_v / NSG, n_v_heads, n_seqs).
    let grid_tgs = MTLSize::new((p.d_v / nsg) as u64, p.n_v_heads as u64, p.n_seqs as u64);

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, q),
            (1, k),
            (2, v),
            (3, g),
            (4, beta),
            (5, state_in),
            (6, output),
            (7, state_out),
            (8, params_buf),
        ],
        grid_tgs,
        tg,
    );

    Ok(())
}
