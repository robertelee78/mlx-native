//! ADR-040 §26 iter-M — GPU-side first-max argmax + threshold candidate collect.
//!
//! Replaces the host full-vocab argmax + candidate-threshold scans (~0.92ms/step
//! on the autoregressive critical path) with one GPU dispatch that reads back
//! only the per-slot top1 + the few threshold candidates. The host keeps the
//! cheap F64 rerank over those candidates (Metal has no f64). Byte-matches the
//! host `argmax_f32_first_max` (first-max, lower-index tie-break) + the finalize
//! threshold scan (logits >= top1_val - 0.5f).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static GPU_SAMPLE_SHADER_SOURCE: &str =
    include_str!("../shaders/gpu_sample_argmax_candidates.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("gpu_sample_argmax_candidates", GPU_SAMPLE_SHADER_SOURCE);
}

/// Dispatch GPU argmax+candidate-collect over `[n_slots, vocab]` logits.
///
/// Outputs (per slot): `out_top1_idx[n]`, `out_top1_val[n]`,
/// `out_cand_count[n]` (atomic u32 — total count, may exceed `cap`),
/// `out_overflow[n]` (1 if count>cap), `out_cand_ids[n*cap]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gpu_sample_argmax_candidates(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    logits: &MlxBuffer,
    out_top1_idx: &MlxBuffer,
    out_top1_val: &MlxBuffer,
    out_cand_count: &MlxBuffer,
    out_overflow: &MlxBuffer,
    out_cand_ids: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_slots: u32,
    vocab: u32,
    cap: u32,
) -> Result<()> {
    if n_slots == 0 || vocab == 0 || cap == 0 {
        return Err(MlxError::InvalidArgument(
            "gpu_sample: n_slots, vocab, cap must all be > 0".into(),
        ));
    }
    if logits.element_count() < (n_slots * vocab) as usize {
        return Err(MlxError::InvalidArgument(format!(
            "gpu_sample: logits {} < n_slots*vocab {}",
            logits.element_count(),
            n_slots * vocab
        )));
    }
    if out_cand_ids.element_count() < (n_slots * cap) as usize {
        return Err(MlxError::InvalidArgument(
            "gpu_sample: out_cand_ids too small".into(),
        ));
    }

    let pipeline = registry.get_pipeline("gpu_sample_argmax_candidates", device)?;

    // Power-of-two threadgroup for the tree reduction; 1024 (each thread scans
    // ~256 cols at vocab=262144).
    let tg_size: u64 = std::cmp::min(1024, vocab.next_power_of_two() as u64).max(1);
    let float_shared = tg_size * 4;
    let uint_shared = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, logits),
            (1, out_top1_idx),
            (2, out_top1_val),
            (3, out_cand_count),
            (4, out_overflow),
            (5, out_cand_ids),
            (6, params_buf),
        ],
        &[(0, float_shared), (1, uint_shared)],
        MTLSize::new(n_slots as u64, 1, 1), // one threadgroup per slot
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
