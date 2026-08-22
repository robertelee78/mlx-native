# Banked TQ KV arena spike

Status: validated on Apple M5 Max at mlx-native base
`27373708e274959b6fb00aadffe499e4db673b31`.

## Hypothesis and result

A batched TQ-HB decode can preserve scalar results while its queries own
different physical capacities inside one bounded allocation. The deciding
test used capacities 16 and 64 with non-contiguous bases and compared the
batched writer and attention output with two independent scalar executions.

The hypothesis passed:

- packed writer bytes and norm bits matched the scalar calls;
- every batched attention output bit matched the corresponding scalar call;
- guard rows before, between, and after the two banks were unchanged;
- changing only the second bank changed the second result while every bit of
  the first result remained unchanged;
- an invalid high base wrote no KV bytes and produced a numerically zero
  attention row without reading outside the arena.

## Layout contract

The new layout buffers are `base_token_rows[i]`, `capacities[i]`, and the
existing `seq_pos[i]`. For query `i`, KV head `h`, and physical position `p`,
the flattened row is:

```text
row = base_token_rows[i] + h * capacities[i] + p
packed_byte = row * head_dim
norm_element = row * (head_dim / 256)
```

The kernels calculate the base, end, and element offsets with 64-bit Metal
integers. They reject `capacity == 0`, a bank end beyond the declared arena,
and a non-ring sequence position outside its bank. The existing uniform APIs
remain source-compatible: an internal `arena_token_capacity == 0` selects the
original `slot * n_kv_heads * kv_capacity` calculation.

## Offset-surface audit

The scalar, sequence, dual-stream, and fused TQ-HB encoders already bind
`MlxBuffer::slice_view` offsets and therefore work on an individual bank
without shader changes. Uniform slot multiplication existed in two full
TQ-HB batched surfaces; both now accept the bank descriptor through new
public entry points:

- `dispatch_hadamard_quantize_kv_hb_banked`
- `flash_attn_vec_tq_hb_batched_banked`

The hybrid F16-K/TQ-HB-V batched attention kernel has its own uniform-slot
calculation. It was audited but is not part of the full-TQ API implemented by
this spike; a hybrid cache cannot select banked layout until that separate
kernel and dispatcher receive the same descriptor contract and parity test.

## Growth copy primitive

`CommandEncoder::blit_copy_bytes` copies cursor-bounded raw byte ranges on the
GPU with explicit source and destination offsets. Logical slice offsets are
composed before encoding. Exact self-copy is a no-op, disjoint ranges in one
allocation are supported, and overlapping same-allocation ranges are rejected
because the Metal blit is not a `memmove` contract. Capture mode rejects a
real blit until the captured graph IR can represent blit nodes.

## Reproducible gates

The following release-mode hardware tests passed:

```text
cargo test --release --locked --test test_tq_hb_banked_arena -- --nocapture
cargo test --release --locked --test test_hadamard_quantize_kv_hb_batched_parity -- --nocapture
cargo test --release --locked --test test_flash_attn_vec_tq_hb_batched_parity -- --nocapture
cargo test --release --locked --test test_flash_attn_vec_tq_hb_batched_parity_nwg1 -- --nocapture
cargo test --release --locked --test test_buffer_blit -- --nocapture
```
