//! FP32 embedding-table lookup with reverse-mode autograd backward.
//!
//! Used by hf2q's ADR-020 Track 1 multi-layer model on GpuTape (iter-11d).
//!
//! Forward: `output[b, h] = embedding[ids[b], h]`
//! Backward: `d_embedding[id, h] = Σ_{b: ids[b] == id} dy[b, h]`
//!
//! The existing `shaders/embedding.metal` covers QUANTIZED 4-bit/6-bit
//! lookup for inference; this module is the FP32-everywhere variant
//! needed by the autograd tape.
//!
//! The backward kernel is O(vocab × hidden × batch) — fine for the
//! test fixtures (vocab ≤ a few hundred); production-scale
//! performance (vocab=150k+) is a follow-up optimization (atomic
//! float adds or sort-segment-sum).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static EMBEDDING_AUTOGRAD_SHADER_SOURCE: &str =
    include_str!("../shaders/embedding_autograd.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("embedding_lookup_f32", EMBEDDING_AUTOGRAD_SHADER_SOURCE);
    registry.register_source(
        "embedding_scatter_add_f32",
        EMBEDDING_AUTOGRAD_SHADER_SOURCE,
    );
}

/// Encode `output[b, h] = embedding[ids[b], h]`.
///
/// `ids` element type must be u32 (kernel reads as `uint32_t`).  Out-of-range
/// IDs (≥ vocab) silently produce 0.0 instead of OOB reads.
///
/// `params_buf` must be at least 8 bytes (2 × u32: vocab, hidden).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_embedding_lookup_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    embedding: &MlxBuffer,
    ids: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    vocab: u32,
    hidden: u32,
    batch: u32,
) -> Result<()> {
    if vocab == 0 || hidden == 0 || batch == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_lookup_f32: vocab/hidden/batch must all be > 0".into(),
        ));
    }
    if embedding.element_count() != (vocab as usize) * (hidden as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_lookup_f32: embedding element count {} != vocab({vocab}) * hidden({hidden})",
            embedding.element_count(),
        )));
    }
    // ids buffer is u32; element_count counts u32 elements.
    if ids.element_count() != batch as usize {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_lookup_f32: ids element count {} != batch ({batch})",
            ids.element_count()
        )));
    }
    if output.element_count() != (batch as usize) * (hidden as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_lookup_f32: output element count {} != batch({batch}) * hidden({hidden})",
            output.element_count(),
        )));
    }
    if embedding.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_lookup_f32: embedding/output dtype must be f32; got {} / {}",
            embedding.dtype(),
            output.dtype()
        )));
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_lookup_f32: params_buf too small (need 8 bytes for 2×u32, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("embedding_lookup_f32", device)?;
    encoder.encode(
        pipeline,
        &[(0, embedding), (1, ids), (2, output), (3, params_buf)],
        MTLSize::new(hidden as u64, batch as u64, 1),
        MTLSize::new(
            std::cmp::min(hidden as u64, 32),
            std::cmp::min(batch as u64, 8),
            1,
        ),
    );
    Ok(())
}

/// Encode the embedding backward (scatter-add).
///
/// `d_embedding` MUST be pre-zeroed by caller — the kernel writes
/// each cell exactly once with the accumulated upstream contribution.
///
/// `params_buf` must be at least 12 bytes (3 × u32: vocab, hidden, batch).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_embedding_scatter_add_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    dy: &MlxBuffer,
    ids: &MlxBuffer,
    d_embedding: &MlxBuffer,
    params_buf: &MlxBuffer,
    vocab: u32,
    hidden: u32,
    batch: u32,
) -> Result<()> {
    if vocab == 0 || hidden == 0 || batch == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_scatter_add_f32: vocab/hidden/batch must all be > 0".into(),
        ));
    }
    if dy.element_count() != (batch as usize) * (hidden as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_scatter_add_f32: dy element count {} != batch({batch}) * hidden({hidden})",
            dy.element_count(),
        )));
    }
    if ids.element_count() != batch as usize {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_scatter_add_f32: ids element count {} != batch ({batch})",
            ids.element_count()
        )));
    }
    if d_embedding.element_count() != (vocab as usize) * (hidden as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_scatter_add_f32: d_embedding element count {} != vocab({vocab}) * hidden({hidden})",
            d_embedding.element_count(),
        )));
    }
    if dy.dtype() != DType::F32 || d_embedding.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_scatter_add_f32: dy/d_embedding dtype must be f32; got {} / {}",
            dy.dtype(),
            d_embedding.dtype()
        )));
    }
    if params_buf.byte_len() < 12 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_scatter_add_f32: params_buf too small (need 12 bytes for 3×u32, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("embedding_scatter_add_f32", device)?;
    encoder.encode(
        pipeline,
        &[(0, dy), (1, ids), (2, d_embedding), (3, params_buf)],
        MTLSize::new(hidden as u64, vocab as u64, 1),
        MTLSize::new(
            std::cmp::min(hidden as u64, 32),
            std::cmp::min(vocab as u64, 8),
            1,
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn cpu_lookup(embedding: &[f32], ids: &[u32], hidden: usize) -> Vec<f32> {
        let mut out = vec![0f32; ids.len() * hidden];
        for (b, &id) in ids.iter().enumerate() {
            let id = id as usize;
            for h in 0..hidden {
                out[b * hidden + h] = embedding[id * hidden + h];
            }
        }
        out
    }

    fn cpu_scatter_add(dy: &[f32], ids: &[u32], vocab: usize, hidden: usize) -> Vec<f32> {
        let mut d_embed = vec![0f32; vocab * hidden];
        for (b, &id) in ids.iter().enumerate() {
            let id = id as usize;
            for h in 0..hidden {
                d_embed[id * hidden + h] += dy[b * hidden + h];
            }
        }
        d_embed
    }

    fn run_lookup(embedding: &[f32], ids: &[u32], vocab: usize, hidden: usize) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let batch = ids.len();
        let mut e_buf = device
            .alloc_buffer(vocab * hidden * 4, DType::F32, vec![vocab, hidden])
            .expect("alloc embedding");
        e_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(embedding);
        let mut id_buf = device
            .alloc_buffer(batch * 4, DType::U32, vec![batch])
            .expect("alloc ids");
        id_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(ids);
        let out_buf = device
            .alloc_buffer(batch * hidden * 4, DType::F32, vec![batch, hidden])
            .expect("alloc out");
        let mut params = device
            .alloc_buffer(8, DType::F32, vec![2])
            .expect("alloc params");
        params.as_mut_slice::<u32>().unwrap()[..2]
            .copy_from_slice(&[vocab as u32, hidden as u32]);

        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_embedding_lookup_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &e_buf,
            &id_buf,
            &out_buf,
            &params,
            vocab as u32,
            hidden as u32,
            batch as u32,
        )
        .expect("dispatch lookup");
        encoder.commit_and_wait().expect("commit");
        out_buf.as_slice::<f32>().unwrap().to_vec()
    }

    fn run_scatter_add(
        dy: &[f32],
        ids: &[u32],
        vocab: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let batch = ids.len();
        let mut dy_buf = device
            .alloc_buffer(batch * hidden * 4, DType::F32, vec![batch, hidden])
            .expect("alloc dy");
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(dy);
        let mut id_buf = device
            .alloc_buffer(batch * 4, DType::U32, vec![batch])
            .expect("alloc ids");
        id_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(ids);
        // alloc_buffer is zero-fill (ADR-015 iter61a).
        let de_buf = device
            .alloc_buffer(vocab * hidden * 4, DType::F32, vec![vocab, hidden])
            .expect("alloc d_embedding");
        let mut params = device
            .alloc_buffer(12, DType::F32, vec![3])
            .expect("alloc params");
        params.as_mut_slice::<u32>().unwrap()[..3]
            .copy_from_slice(&[vocab as u32, hidden as u32, batch as u32]);

        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_embedding_scatter_add_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &dy_buf,
            &id_buf,
            &de_buf,
            &params,
            vocab as u32,
            hidden as u32,
            batch as u32,
        )
        .expect("dispatch scatter_add");
        encoder.commit_and_wait().expect("commit");
        de_buf.as_slice::<f32>().unwrap().to_vec()
    }

    #[test]
    fn embedding_lookup_byte_identical_to_cpu() {
        let vocab = 16;
        let hidden = 8;
        let embedding: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32) * 0.13 - 0.5)
            .collect();
        let ids: Vec<u32> = vec![3, 7, 0, 15, 5, 5, 12, 1];
        let gpu = run_lookup(&embedding, &ids, vocab, hidden);
        let cpu = cpu_lookup(&embedding, &ids, hidden);
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g.to_bits(), c.to_bits(), "mismatch at {i}");
        }
    }

    #[test]
    fn embedding_lookup_handles_repeated_ids() {
        // Same ID appearing multiple times in batch.  Output rows must
        // all match the embedding row exactly.
        let vocab = 8;
        let hidden = 4;
        let embedding: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32) * 0.7)
            .collect();
        let ids: Vec<u32> = vec![5, 5, 5, 5];
        let gpu = run_lookup(&embedding, &ids, vocab, hidden);
        let row5 = &embedding[5 * hidden..6 * hidden];
        for b in 0..ids.len() {
            for h in 0..hidden {
                assert_eq!(gpu[b * hidden + h].to_bits(), row5[h].to_bits());
            }
        }
    }

    #[test]
    fn embedding_scatter_add_byte_identical_to_cpu() {
        let vocab = 16;
        let hidden = 8;
        let batch = 12;
        let dy: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32) * 0.011 - 0.05)
            .collect();
        let ids: Vec<u32> = vec![3, 7, 0, 15, 5, 5, 12, 1, 5, 0, 7, 11];
        let gpu = run_scatter_add(&dy, &ids, vocab, hidden);
        let cpu = cpu_scatter_add(&dy, &ids, vocab, hidden);
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g.to_bits(), c.to_bits(), "scatter-add mismatch at {i}");
        }
    }

    #[test]
    fn embedding_scatter_add_unused_ids_are_zero() {
        // IDs 0, 4, 9, 13 are NEVER used in the batch — their rows in
        // d_embedding must remain zero.
        let vocab = 16;
        let hidden = 4;
        let batch = 6;
        let dy: Vec<f32> = (0..batch * hidden).map(|i| (i as f32) + 1.0).collect();
        let ids: Vec<u32> = vec![1, 2, 3, 5, 7, 11];
        let gpu = run_scatter_add(&dy, &ids, vocab, hidden);
        for &unused_id in &[0u32, 4, 6, 8, 9, 10, 12, 13, 14, 15] {
            for h in 0..hidden {
                assert_eq!(
                    gpu[unused_id as usize * hidden + h], 0.0,
                    "unused id {unused_id} row should be zero at h={h}"
                );
            }
        }
    }

    #[test]
    fn embedding_round_trip_lookup_then_scatter_add() {
        // Lookup by ids; then scatter-add the lookup output back —
        // the scatter sums all batch contributions back into the
        // touched rows.  For each id `i` appearing `k` times,
        // d_embedding[i] should equal `k * embedding[i]`.
        let vocab = 8;
        let hidden = 4;
        let embedding: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.5).collect();
        let ids: Vec<u32> = vec![2, 5, 2, 7, 5, 5, 2];
        // counts: 2 appears 3x, 5 appears 3x, 7 appears 1x.
        let lookup_out = run_lookup(&embedding, &ids, vocab, hidden);
        let scatter = run_scatter_add(&lookup_out, &ids, vocab, hidden);
        for id in 0..vocab {
            let count = ids.iter().filter(|&&i| i as usize == id).count();
            for h in 0..hidden {
                let expected = embedding[id * hidden + h] * (count as f32);
                let actual = scatter[id * hidden + h];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "id={id} h={h}: expected {expected} (count={count}), got {actual}"
                );
            }
        }
    }
}
