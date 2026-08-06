//! Deterministic host-precomputed inverse-frequency tables for RoPE kernels.
//!
//! Apple GPU families are allowed to return different bounded-accuracy results
//! for `pow`, even through Metal's `precise` namespace.  Multiplying that small
//! difference by a long-context position can move the final rotation far enough
//! to change model output.  Compute the tiny, shape-stable table once with the
//! host's f32 contract and cache it in a shared Metal buffer instead.

use std::cell::RefCell;
use std::collections::HashMap;

use metal::MTLResourceOptions;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::error::{MlxError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RopeFreqCacheKey {
    device_ptr: usize,
    freq_base_bits: u32,
    denominator: u32,
    pair_count: u32,
}

thread_local! {
    static ROPE_FREQ_CACHE: RefCell<HashMap<RopeFreqCacheKey, MlxBuffer>> =
        RefCell::new(HashMap::new());
}

fn build_inv_freqs(freq_base: f32, denominator: u32, pair_count: u32) -> Vec<f32> {
    (0..pair_count)
        .map(|pair| {
            let dim_ratio = (2 * pair) as f32 / denominator as f32;
            1.0_f32 / freq_base.powf(dim_ratio)
        })
        .collect()
}

/// Borrow a cached Metal buffer containing
/// `freq_base^(-2 * pair / denominator)` for `pair in 0..pair_count`.
///
/// The callback keeps the thread-local cache borrow scoped to the immediate
/// encode operation, so callers cannot retain a reference after a cache clear.
pub(crate) fn with_inv_freqs<R>(
    device: &metal::DeviceRef,
    freq_base: f32,
    denominator: u32,
    pair_count: u32,
    f: impl FnOnce(&MlxBuffer) -> R,
) -> Result<R> {
    if !freq_base.is_finite() || freq_base <= 0.0 {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE freq_base must be finite and positive, got {freq_base}"
        )));
    }
    if denominator == 0 || pair_count == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE frequency denominator and pair_count must be > 0, got {denominator} and {pair_count}"
        )));
    }

    let key = RopeFreqCacheKey {
        device_ptr: device as *const metal::DeviceRef as usize,
        freq_base_bits: freq_base.to_bits(),
        denominator,
        pair_count,
    };

    ROPE_FREQ_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !cache.contains_key(&key) {
            let inv_freqs = build_inv_freqs(freq_base, denominator, pair_count);
            let byte_len = std::mem::size_of_val(inv_freqs.as_slice());
            let metal_buf = device.new_buffer_with_data(
                inv_freqs.as_ptr().cast(),
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            );
            cache.insert(
                key,
                MlxBuffer::from_raw(metal_buf, DType::F32, vec![pair_count as usize]),
            );
        }
        Ok(f(cache.get(&key).expect(
            "RoPE inverse-frequency cache entry inserted above",
        )))
    })
}

/// Release this thread's tiny RoPE inverse-frequency buffers.
pub fn clear_cache() {
    ROPE_FREQ_CACHE.with(|cell| cell.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::build_inv_freqs;

    #[test]
    fn host_schedule_matches_f32_rope_definition() {
        let got = build_inv_freqs(1_000_000.0, 16, 8);
        for (pair, &freq) in got.iter().enumerate() {
            let ratio = (2 * pair) as f32 / 16.0_f32;
            assert_eq!(freq, 1.0_f32 / 1_000_000.0_f32.powf(ratio));
        }
    }
}
