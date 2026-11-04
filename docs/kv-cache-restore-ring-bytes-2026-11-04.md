# `dispatch_kv_cache_restore_ring_bytes`: ring pre-image restore with `src_row_offset`

## Purpose

Speculative verify forwards write window rows into a sliding-window ring of
1,024 slots. Rejected rows overwrite readable history. The caller saves the
pre-image of those slots with `dispatch_kv_cache_linearize_ring_bytes`
before the verify and, after deciding how many rows were accepted, restores
only the tail of that pre-image (the rejected rows).

## Contract

For `i < history_len` and every head `h`:

`dst[h, (logical_end − history_len + i) mod dst_capacity] = src[h, src_row_offset + i]`

- `src` is the staging buffer, `src_capacity` rows per head; `dst` is the
  ring, `dst_capacity` slots per head; rows are `row_bytes` opaque bytes
  (F16 K, packed V codes, F32 norms).
- No other byte of `dst` is written.
- Refused before dispatch: `src_row_offset + history_len > src_capacity`,
  `history_len > dst_capacity`, `logical_end < history_len`, zero
  capacities, and either view's `data_byte_len()` below
  `n_heads × capacity × row_bytes` (the view length, not the whole Metal
  buffer, so a per-slot view cannot be written past).
- `restore(offset, n) ∘ linearize(n_saved)` with the same `logical_end` and
  `n = n_saved − offset` is the identity on the `n` restored rows.

## Evidence

Apple M5 Max, macOS 27.2, release test profile.

`cargo test --release --test kv_cache_restore_ring_bytes` — 1 passed.
The test `kv_cache_restore_ring_bytes_inverts_linearize`:

- row_bytes ∈ {4, 256, 512, 1024}; heads ∈ {2, 8}; ring capacity 1,024;
  staging capacity 4; `logical_end` ∈ {600 (no wrap), 1026 (wrap across
  slot 0), 2050}; `n_saved` 1–4; `src_row_offset` 0..n_saved−1 (240 cases);
- linearize, clobber the saved slots, restore the tail, then compare the
  whole parent buffer (prefix/suffix canaries around the ring view) to the
  expected bytes: restored rows equal the original, unrestored saved rows
  keep the clobber, every other byte is unchanged; staging canaries intact;
- refusals: a ring or staging view one byte short inside a larger parent
  buffer, offset + rows past staging capacity, `logical_end < history_len`;
- runs on the default path (asserts `PrecompiledMetallib` origin) and
  re-runs itself in a child with `MLX_PRECOMPILED_METALLIB=0` (asserts
  `RuntimeSource` origin).
