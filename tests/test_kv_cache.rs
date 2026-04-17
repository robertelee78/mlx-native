//! Tests for the KV cache GPU copy kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::kv_cache_copy;
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    kv_cache_copy::register(&mut registry);
    (device, registry)
}

/// Convert f32 to bf16 raw bytes (2 bytes LE).
fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let bf16_bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    bf16_bits.to_le_bytes()
}

/// Convert bf16 raw bytes to f32.
fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bf16_bits = u16::from_le_bytes(bytes);
    f32::from_bits((bf16_bits as u32) << 16)
}

/// Write bf16 values into a raw byte buffer.
fn write_bf16_values(buf: &mut [u8], values: &[f32]) {
    for (i, &v) in values.iter().enumerate() {
        let bytes = f32_to_bf16_bytes(v);
        buf[i * 2] = bytes[0];
        buf[i * 2 + 1] = bytes[1];
    }
}

/// Read bf16 values from a raw byte buffer as f32.
// Note: read_bf16_values is available if needed for future tests.
// fn read_bf16_values(buf: &[u8], count: usize) -> Vec<f32> {
//     (0..count)
//         .map(|i| bf16_bytes_to_f32([buf[i * 2], buf[i * 2 + 1]]))
//         .collect()
// }

#[test]
fn test_kv_cache_copy_linear() {
    let (device, mut registry) = setup();

    let n_new: u32 = 4;
    let row_size: u32 = 128; // n_kv_heads * head_dim
    let cache_cap: u32 = 64;
    let write_pos: u32 = 0;

    let total_src = n_new as usize * row_size as usize;
    let total_cache = cache_cap as usize * row_size as usize;

    // Generate source data (bf16)
    let src_f32: Vec<f32> = (0..total_src).map(|i| (i as f32) * 0.01).collect();
    let mut src_bytes = vec![0u8; total_src * 2];
    write_bf16_values(&mut src_bytes, &src_f32);

    // Allocate GPU buffers
    let mut src_buf = device
        .alloc_buffer(total_src * 2, DType::BF16, vec![total_src])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<u8>()
        .expect("write src")
        .copy_from_slice(&src_bytes);

    let mut cache_buf = device
        .alloc_buffer(total_cache * 2, DType::BF16, vec![total_cache])
        .expect("alloc cache");
    // Zero the cache
    for b in cache_buf.as_mut_slice::<u8>().expect("write cache").iter_mut() {
        *b = 0;
    }

    let mut encoder = device.command_encoder().expect("encoder");
    kv_cache_copy::dispatch_kv_cache_copy(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        write_pos,
        row_size,
        n_new,
        cache_cap,
        false, // not sliding
    )
    .expect("dispatch_kv_cache_copy");

    encoder.commit_and_wait().expect("commit_and_wait");

    // Read back cache and verify
    let cache_bytes: Vec<u8> = cache_buf.as_slice::<u8>().expect("read cache").to_vec();

    // The first n_new rows of the cache should match the source
    for i in 0..total_src {
        let src_val = bf16_bytes_to_f32([src_bytes[i * 2], src_bytes[i * 2 + 1]]);
        let cache_val = bf16_bytes_to_f32([cache_bytes[i * 2], cache_bytes[i * 2 + 1]]);
        assert!(
            (src_val - cache_val).abs() < 1e-10,
            "linear copy: bitwise mismatch at element {}: src_bf16={}, cache_bf16={} (src_f32={}, cache_f32={})",
            i, src_val, cache_val,
            src_f32[i], cache_val,
        );
    }

    // Remaining cache rows should still be zero
    for i in total_src..total_cache {
        let val = bf16_bytes_to_f32([cache_bytes[i * 2], cache_bytes[i * 2 + 1]]);
        assert!(
            val == 0.0,
            "linear copy: cache element {} should be 0.0, got {}",
            i, val
        );
    }
}

#[test]
fn test_kv_cache_copy_sliding_wrap() {
    let (device, mut registry) = setup();

    let n_new: u32 = 3;
    let row_size: u32 = 64;
    let cache_cap: u32 = 4; // small cache to force wrapping
    let write_pos: u32 = 6; // seq_pos > cache_cap => wrapping

    let total_src = n_new as usize * row_size as usize;
    let total_cache = cache_cap as usize * row_size as usize;

    // Source data
    let src_f32: Vec<f32> = (0..total_src).map(|i| ((i + 1) as f32) * 0.1).collect();
    let mut src_bytes = vec![0u8; total_src * 2];
    write_bf16_values(&mut src_bytes, &src_f32);

    // Pre-fill cache with known values
    let cache_init: Vec<f32> = (0..total_cache).map(|i| -(i as f32) * 0.01).collect();
    let mut cache_init_bytes = vec![0u8; total_cache * 2];
    write_bf16_values(&mut cache_init_bytes, &cache_init);

    let mut src_buf = device
        .alloc_buffer(total_src * 2, DType::BF16, vec![total_src])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<u8>()
        .expect("write src")
        .copy_from_slice(&src_bytes);

    let mut cache_buf = device
        .alloc_buffer(total_cache * 2, DType::BF16, vec![total_cache])
        .expect("alloc cache");
    cache_buf
        .as_mut_slice::<u8>()
        .expect("write cache")
        .copy_from_slice(&cache_init_bytes);

    let mut encoder = device.command_encoder().expect("encoder");
    kv_cache_copy::dispatch_kv_cache_copy(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        write_pos,
        row_size,
        n_new,
        cache_cap,
        true, // sliding
    )
    .expect("dispatch_kv_cache_copy sliding");

    encoder.commit_and_wait().expect("commit_and_wait");

    // CPU reference: for each token t in [0..n_new), the write position is
    // (write_pos + t) % cache_cap.  Each token's data is row_size bf16 elements.
    let cache_bytes: Vec<u8> = cache_buf.as_slice::<u8>().expect("read cache").to_vec();

    for t in 0..n_new as usize {
        let cache_row = ((write_pos as usize) + t) % (cache_cap as usize);
        for j in 0..row_size as usize {
            let src_elem = t * row_size as usize + j;
            let cache_elem = cache_row * row_size as usize + j;

            let src_val = bf16_bytes_to_f32([src_bytes[src_elem * 2], src_bytes[src_elem * 2 + 1]]);
            let cache_val = bf16_bytes_to_f32([cache_bytes[cache_elem * 2], cache_bytes[cache_elem * 2 + 1]]);

            assert!(
                (src_val - cache_val).abs() < 1e-10,
                "sliding wrap: mismatch at token {} elem {} (cache_row={}): src={}, cache={}",
                t, j, cache_row, src_val, cache_val,
            );
        }
    }
}

#[test]
fn test_kv_cache_copy_empty() {
    let (device, mut registry) = setup();

    let src_buf = device
        .alloc_buffer(2, DType::BF16, vec![1])
        .expect("alloc src");
    let cache_buf = device
        .alloc_buffer(2, DType::BF16, vec![1])
        .expect("alloc cache");

    let mut encoder = device.command_encoder().expect("encoder");
    // n_new=0 should be a no-op, not an error
    let result = kv_cache_copy::dispatch_kv_cache_copy(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        0,  // write_pos
        64, // row_size
        0,  // n_new
        64, // cache_cap
        false,
    );
    assert!(result.is_ok(), "n_new=0 should succeed (no-op)");
}

#[test]
fn test_kv_cache_copy_global_overflow_error() {
    let (device, mut registry) = setup();

    let src_buf = device
        .alloc_buffer(256, DType::BF16, vec![128])
        .expect("alloc src");
    let cache_buf = device
        .alloc_buffer(256, DType::BF16, vec![128])
        .expect("alloc cache");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = kv_cache_copy::dispatch_kv_cache_copy(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        3,  // write_pos
        1,  // row_size
        2,  // n_new
        4,  // cache_cap = 4, write_pos(3) + n_new(2) = 5 > 4
        false, // global (not sliding)
    );

    assert!(result.is_err(), "Should error on global cache overflow");
}

/// Ring-wrap behaviour of the multi-position seq prefill kernel
/// (`kv_cache_copy_seq_f32`). Exercises hf2q Gate A: batched prefill on a
/// prompt longer than a sliding-window layer's capacity must ring-wrap
/// writes so the last `capacity` source tokens survive in modular-slot
/// order, matching the layout equivalent sequential decode appends would
/// produce.
#[test]
fn test_kv_cache_copy_seq_f32_sliding_ring_wrap() {
    let (device, mut registry) = setup();

    let n_heads: u32 = 2;
    let head_dim: u32 = 4;
    let capacity: u32 = 4; // small: forces wrap when n_tokens > 4
    let n_tokens: u32 = 8; // two full rings

    let src_elems = n_tokens as usize * n_heads as usize * head_dim as usize;
    let cache_elems = n_heads as usize * capacity as usize * head_dim as usize;

    // Encode (tok, head, elem) into the source value so we can tell which
    // token survived at which slot after ring-wrap.
    let mut src_f32 = vec![0f32; src_elems];
    for t in 0..n_tokens as usize {
        for h in 0..n_heads as usize {
            for e in 0..head_dim as usize {
                let idx = t * (n_heads as usize * head_dim as usize)
                    + h * head_dim as usize + e;
                // Token 0 => 1000.0, token 1 => 1001.0, etc., offset by head+elem
                src_f32[idx] = 1000.0 + t as f32 + 0.1 * h as f32 + 0.01 * e as f32;
            }
        }
    }
    let src_bytes: Vec<u8> = src_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut src_buf = device
        .alloc_buffer(src_elems * 4, DType::F32, vec![src_elems])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<u8>()
        .expect("write src")
        .copy_from_slice(&src_bytes);

    let mut cache_buf = device
        .alloc_buffer(cache_elems * 4, DType::F32, vec![cache_elems])
        .expect("alloc cache");
    for b in cache_buf.as_mut_slice::<u8>().expect("zero cache").iter_mut() {
        *b = 0;
    }

    // Sliding-window contract: host passes the last `capacity` source tokens
    // via `src_tok_offset`, eliminating the intra-dispatch race that naive
    // modular writes would have.
    let src_tok_offset = n_tokens - capacity; // 8 - 4 = 4
    let dispatch_n_tokens = capacity;
    let dispatch_seq_pos_start = src_tok_offset;

    let mut encoder = device.command_encoder().expect("encoder");
    kv_cache_copy::dispatch_kv_cache_copy_seq_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        n_heads,
        head_dim,
        capacity,
        dispatch_seq_pos_start,
        dispatch_n_tokens,
        src_tok_offset,
    )
    .expect("dispatch_kv_cache_copy_seq_f32");
    encoder.commit_and_wait().expect("commit_and_wait");

    let cache_slice: &[f32] = cache_buf.as_slice::<f32>().expect("read cache");

    // For each slot s in [0..capacity), the surviving source token is the
    // one at absolute position `src_tok_offset + t` where
    // `(dispatch_seq_pos_start + t) % capacity == s`. With our inputs this
    // resolves uniquely: slot s holds source token (src_tok_offset + s).
    for h in 0..n_heads as usize {
        for s in 0..capacity as usize {
            let t = src_tok_offset as usize + s; // unique surviving source token
            for e in 0..head_dim as usize {
                let cache_idx = h * capacity as usize * head_dim as usize
                    + s * head_dim as usize + e;
                let expected = 1000.0 + t as f32 + 0.1 * h as f32 + 0.01 * e as f32;
                let got = cache_slice[cache_idx];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "ring-wrap mismatch: head={} slot={} elem={} \
                     expected token {} value {}, got {}",
                    h, s, e, t, expected, got,
                );
            }
        }
    }
}

/// Non-wrap behaviour: when `seq_pos_start + n_tokens <= capacity` the
/// seq kernel must behave linearly (mod is identity). Global (non-sliding)
/// layers rely on this — their capacity is always `>= prompt + decode`.
#[test]
fn test_kv_cache_copy_seq_f32_no_wrap() {
    let (device, mut registry) = setup();

    let n_heads: u32 = 2;
    let head_dim: u32 = 4;
    let capacity: u32 = 8;
    let n_tokens: u32 = 3; // well below capacity
    let seq_pos_start: u32 = 0;

    let src_elems = n_tokens as usize * n_heads as usize * head_dim as usize;
    let cache_elems = n_heads as usize * capacity as usize * head_dim as usize;

    let mut src_f32 = vec![0f32; src_elems];
    for t in 0..n_tokens as usize {
        for h in 0..n_heads as usize {
            for e in 0..head_dim as usize {
                let idx = t * (n_heads as usize * head_dim as usize)
                    + h * head_dim as usize + e;
                src_f32[idx] = 2000.0 + t as f32 + 0.1 * h as f32 + 0.01 * e as f32;
            }
        }
    }
    let src_bytes: Vec<u8> = src_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut src_buf = device
        .alloc_buffer(src_elems * 4, DType::F32, vec![src_elems])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<u8>()
        .expect("write src")
        .copy_from_slice(&src_bytes);

    // Pre-fill cache with a sentinel so we can verify untouched slots stay untouched.
    let sentinel = -9999.0f32;
    let mut cache_buf = device
        .alloc_buffer(cache_elems * 4, DType::F32, vec![cache_elems])
        .expect("alloc cache");
    {
        let dst: &mut [f32] = cache_buf
            .as_mut_slice::<f32>()
            .expect("prefill cache");
        for v in dst.iter_mut() { *v = sentinel; }
    }

    let mut encoder = device.command_encoder().expect("encoder");
    kv_cache_copy::dispatch_kv_cache_copy_seq_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        n_heads,
        head_dim,
        capacity,
        seq_pos_start,
        n_tokens,
        0, // src_tok_offset: no skipping for the linear no-wrap case
    )
    .expect("dispatch_kv_cache_copy_seq_f32");
    encoder.commit_and_wait().expect("commit_and_wait");

    let cache_slice: &[f32] = cache_buf.as_slice::<f32>().expect("read cache");

    // Slots 0..n_tokens should hold the corresponding source token;
    // slots n_tokens..capacity should still be the sentinel.
    for h in 0..n_heads as usize {
        for s in 0..capacity as usize {
            for e in 0..head_dim as usize {
                let cache_idx = h * capacity as usize * head_dim as usize
                    + s * head_dim as usize + e;
                let got = cache_slice[cache_idx];
                if s < n_tokens as usize {
                    let expected = 2000.0 + s as f32 + 0.1 * h as f32 + 0.01 * e as f32;
                    assert!(
                        (got - expected).abs() < 1e-4,
                        "no-wrap: head={} slot={} elem={} expected {}, got {}",
                        h, s, e, expected, got,
                    );
                } else {
                    assert!(
                        (got - sentinel).abs() < 1e-4,
                        "no-wrap: head={} slot={} elem={} should be sentinel, got {}",
                        h, s, e, got,
                    );
                }
            }
        }
    }
}
