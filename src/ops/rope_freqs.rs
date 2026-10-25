//! Deterministic host-precomputed inverse-frequency tables for RoPE kernels.
//!
//! Apple GPU families are allowed to return different bounded-accuracy results
//! for `pow`, even through Metal's `precise` namespace.  Multiplying that small
//! difference by a long-context position can move the final rotation far enough
//! to change model output.  Compute the tiny, shape-stable table once with the
//! same host f32 contract as the CPU oracle and cache it in a shared Metal
//! buffer instead.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

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

static ROPE_FREQ_CACHE: OnceLock<Mutex<HashMap<RopeFreqCacheKey, MlxBuffer>>> = OnceLock::new();

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
/// The process-lifetime cache retains every returned Metal buffer even if the
/// encoding thread exits or an unretained command encoder moves to another
/// thread before commit. The callback receives a cheap Arc-backed clone so the
/// global mutex is not held while encoding.
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

    let inv_freqs = {
        let mut cache = ROPE_FREQ_CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|_| {
                MlxError::CommandBufferError("RoPE frequency cache lock poisoned".into())
            })?;
        crate::root_authority::healthy()?;
        cached_or_create(
            &mut cache,
            key,
            |value| value.known_root_clone().check_device(device),
            |cache| {
                let byte_len = (pair_count as usize)
                    .checked_mul(4)
                    .ok_or_else(|| MlxError::InvalidArgument("RoPE table size overflow".into()))?;
                if byte_len as u64 > device.max_buffer_length()
                    || pair_count.checked_mul(2).is_none()
                {
                    return Err(MlxError::InvalidArgument(
                        "RoPE table exceeds checked extent".into(),
                    ));
                }
                let mut values = None;
                let root = crate::root_authority::shared_with(device, byte_len, true, || {
                    // A failed miss spends both historical reservations, no key inserted.
                    cache.try_reserve(1).map_err(|_| {
                        MlxError::InvalidArgument("RoPE cache reservation failed".into())
                    })?;
                    values = Some(build_inv_freqs(freq_base, denominator, pair_count));
                    crate::device::raw_shared_buffer(device, byte_len)
                })?;
                let publication = crate::root_authority::publication_guard();
                root.check_device(device)?;
                let values =
                    values.expect("successful raw creation follows host table preparation");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        values.as_ptr().cast::<u8>(),
                        root.metal().contents().cast::<u8>(),
                        byte_len,
                    );
                }
                let buffer = MlxBuffer::from_known(root, DType::F32, vec![pair_count as usize])?;
                Ok((buffer, publication))
            },
            |publication| {
                crate::root_authority::cache_published();
                publication.complete();
            },
        )?
    };
    Ok(f(&inv_freqs))
}

// Actual cache hit/create/publication traversal, also exercised without Metal.
fn cached_or_create<K: Eq + std::hash::Hash, V: Clone, G>(
    cache: &mut HashMap<K, V>,
    key: K,
    check: impl FnOnce(&V) -> Result<()>,
    create: impl FnOnce(&mut HashMap<K, V>) -> Result<(V, G)>,
    published: impl FnOnce(G),
) -> Result<V> {
    if let Some(value) = cache.get(&key) {
        check(value)?;
        return Ok(value.clone());
    }
    let (value, guard) = create(cache)?;
    cache.insert(key, value.clone());
    published(guard);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{build_inv_freqs, with_inv_freqs, ROPE_FREQ_CACHE};

    fn cache_len() -> usize {
        match ROPE_FREQ_CACHE
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
            .lock()
        {
            Ok(cache) => cache.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    #[test]
    fn host_schedule_matches_f32_rope_definition() {
        let got = build_inv_freqs(1_000_000.0, 16, 8);
        for (pair, &freq) in got.iter().enumerate() {
            let ratio = (2 * pair) as f32 / 16.0_f32;
            assert_eq!(freq, 1.0_f32 / 1_000_000.0_f32.powf(ratio));
        }
    }

    #[test]
    fn process_cache_survives_encoding_thread_exit() {
        let before = cache_len();
        let worker = std::thread::spawn(|| {
            let Ok(device) = crate::MlxDevice::new() else {
                return false;
            };
            with_inv_freqs(device.metal_device(), 123_456.75_f32, 14, 7, |_| ()).is_ok()
        });
        assert!(matches!(worker.join(), Ok(true)));
        assert_eq!(cache_len(), before + 1);
    }
    #[test]
    fn root_history_rope_actual_cache_traversal_failed_miss_no_insert_and_hit_no_charge() {
        use crate::root_authority::{state::DeviceKey, Core, RootCreationLimits};
        let core = Core::default();
        let key = DeviceKey {
            object: 1,
            registry: 1,
        };
        {
            let mut s = core.lock().unwrap();
            s.install(RootCreationLimits {
                max_attempted_requested_bytes: 64,
                max_root_attempts: 2,
                max_raw_constructor_attempts: 2,
                max_registered_devices: 1,
                max_rope_cache_entries: 2,
            })
            .unwrap();
            s.register(key).unwrap();
        }
        let mut cache = std::collections::HashMap::<u32, u32>::new();
        let failed = super::cached_or_create(
            &mut cache,
            7,
            |_| Ok(()),
            |_| {
                core.create::<u32>(key, 32, true, || {
                    Err(crate::MlxError::InvalidArgument("nil".into()))
                })
            },
            |_| panic!("must not publish"),
        );
        assert!(failed.is_err());
        assert!(cache.is_empty());
        assert_eq!(core.lock().unwrap().cache_slots, 1);
        let value = super::cached_or_create(
            &mut cache,
            7,
            |_| Ok(()),
            |cache| {
                core.create(key, 32, true, || {
                    cache.try_reserve(1).unwrap();
                    Ok(42)
                })
            },
            |_| core.lock().unwrap().published_cache += 1,
        )
        .unwrap();
        assert_eq!(value, 42);
        let value = super::cached_or_create(
            &mut cache,
            7,
            |_| core.health(),
            |_| -> crate::Result<(u32, ())> { panic!("hit must not create") },
            |_| panic!("hit must not publish"),
        )
        .unwrap();
        assert_eq!(value, 42);
        let s = core.lock().unwrap();
        assert_eq!(
            (s.bytes, s.roots, s.cache_slots, s.published_cache),
            (64, 2, 2, 1)
        );
    }
}
