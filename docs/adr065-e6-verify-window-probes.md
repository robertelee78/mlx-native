# ADR-065 E6: verify-window equality probes (qualification record)

Story: hf2q ADR-065 E6 (hf2q issue #509). Scope: mlx-native tests only, with
no production change. The probes find out which existing primitives can serve
speculative verify windows bitwise before anyone designs a new kernel
(ADR-065 P1a–P4a).

Shapes are Qwen3.8-27B. Hidden is 5120 and FFN is 17408. Full attention has 24
query heads, 4 KV heads and head dim 256. DeltaNet has 16 key heads and 48
value heads of dim 128, with conv width 4 (10240 conv channels). Inputs come
from deterministic synthetic data. Bitwise means `f32::to_bits` equality with
zero tolerance.

## Probes

| # | Test binary | Tests | What is compared | Asserted | Recorded only |
|---|---|---|---|---|---|
| 1 | `adr065_e6_mul_mv_ext_width` | `e6_mul_mv_ext_{q4_k,q5_k,q6_k}_8_rows_match_4_rows_at_qwen38_shapes`, `e6_served_{q4_k,q5_k,q6_k}_route_8_rows_match_4_rows_at_qwen38_shapes` | 8-column dispatch vs two 4-column dispatches, every column, 8 projection shapes; `mv_ext` route (`dense_decode_mv_ext`) and default served route (Q4_K/Q6_K mN, Q5_K canonical q4x4) | bitwise | pipeline origin |
| 2 | `adr065_e6_served_kv_attention_windows` | `e6_tq_hb_batched_window_rows_match_single_rows`, `e6_tq_hb_banked_window_rows_match_single_rows`, `e6_f32_flash_attn_vec_window_rows_match_single_rows` | Window row `i` (w ∈ {2,4,8}) vs a single-row query at position `p+i`. Windows sit inside a bucket and across the `nwg` 512/513 and `nsg` 1024/1025 boundaries. Also a four-slot mixed-width cohort vs each window alone | TQ: each row vs the same row alone under the window bucket; in-bucket rows vs the served single row; cohort vs alone, all bitwise | TQ crossing rows vs the served single row, and every F32 `flash_attn_vec` row: bitwise verdict and max abs diff, bounded at 1e-3 |
| 3 | `adr065_e6_gated_delta_net_decode_seqs` | `e6_gated_delta_net_decode_four_sequences_match_single_sequences`, `e6_gated_delta_net_decode_four_sequence_windows_match_single_sequences`, `e6_gated_delta_net_decode_capture_four_sequence_windows_match_single_sequences`, `e6_gated_delta_net_decode_window_versus_sequential_tokens` | `n_seqs=4` vs four `n_seqs=1` at 1 and 4 tokens, with and without capture. A 4-token window vs four sequential 1-token decodes | Multi-seq vs single-seq output, `state_out` and `state_capture`, all bitwise. `state_capture[T-1] == state_out` | Window vs sequential tokens: bitwise verdict, bounded at 1e-4 |
| 4 | `adr065_e6_capture_restore_copy` | `e6_capture_restore_blit_and_strided_copy_match_host_rollback`, `e6_capture_restore_blit_and_transpose_match_host_rollback` | GPU restore vs a port of hf2q `HybridKvCache::rollback_la_to`: four slots, mixed ping-pong parity, accepted `t` ∈ {0,1,2,3}, all in one command buffer. Recurrent restore uses `blit_copy_bytes`. Conv re-index uses `dispatch_strided_copy_f32` or `transpose_2d` on slice views | Every u32 bit pattern (restored slots, peer slots, the non-current parity buffer) for the blit; non-NaN bit patterns for the conv kernels | Conv-kernel NaN-payload preservation (`nan_payload_bitwise`). `E6_STRICT_NAN=1` asserts it |
| 5 | `adr065_e6_weight_load_bandwidth` (`#[ignore]`, timing) | `e6_bandwidth_{mv_ext,served}_{q4_k,q5_k,q6_k}` | 16 rows as 4×4 vs 2×8 dispatches in one command buffer, 4 shapes | Outputs bitwise across groupings; weight passes > 0 | Weight passes (Σ column-group threadgroups), logical weight bytes, median GPU µs, logical and unique GB/s |

Both metallib paths: every probe checks that each pipeline it uses was built
from the selected path (`KernelRegistry::pipeline_identity`). With
`MLX_PRECOMPILED_METALLIB=0` it must be `RuntimeSource`. With the default it
must be `PrecompiledMetallib`. The observed origin goes into every record
line.

Every comparison writes one `E6_RECORD probe=… metallib=… case=… bitwise=…`
line to stderr. When `E6_RECORD_PATH` is set, the line is also appended to
that file.

Expectations derived from the kernel source. These are not results.
- `mul_mv_ext`: m=4 and m=8 share `r1ptg=4` and `nxpsg`, so each column runs
  the same arithmetic. 8 rows load the weight twice, once per column group.
- The mN routes tile 8 columns as 4+4.
- TQ-HB splits the K loop by `nwg` and `nsg`. A row whose single-row bucket
  differs from the window's bucket is therefore not expected to be bitwise.

## Run (requires the GPU lane)

```sh
scripts/adr065_e6_run_probes.sh /Users/robert/.local/state/hf2q/v0130/p4a-526
```

The script runs the correctness probes (E6 and P4a), then the bandwidth
probe, once per metallib path. It writes `records-{precompiled,runtime}.txt`,
`bandwidth-{precompiled,runtime}.txt`, the logs, and a provenance file
(source commit, test-binary sha256, embedded metallib sha256, crate version).

## Results (M5 Max, 2026-11-03)

Run at source commit `608425d` with mlx-native 0.15.16 and rustc 1.94.0.
Embedded metallib sha256 is `a4f35c4b…f961827`. Records, logs and
provenance are in the hf2q v0130 evidence directory `509-e6/`. The capture
restore test was also run with `E6_STRICT_NAN=1` on both paths.

| Primitive | Precompiled | Runtime | Serves verify windows bitwise? |
|---|---|---|---|
| `mul_mv_ext` 8 vs 4 (Q4_K/Q5_K/Q6_K, 8 shapes) | 24/24 bitwise | 24/24 bitwise | Yes |
| Served dense routes 8 vs 4 (Q4_K mN, Q5_K canonical, Q6_K mN) | 24/24 bitwise | 24/24 bitwise | Yes |
| TQ-HB batched: in-bucket, `nwg` 16→32 crossings, cohort | all bitwise | all bitwise | Yes (`nwg`-invariant) |
| TQ-HB batched: `nsg` 1→4 crossing (w4, w8) | 6/12 rows bitwise, max 7.5e-9 | same | **No**. Rows at KV length ≤1024 (single-row `nsg`=1) differ from the window's `nsg`=4. Rows above 1024 match |
| TQ-HB banked: same matrix | identical verdicts, max 4.7e-9 | same | Same as batched |
| F32 `flash_attn_vec` window (all 9 placements) | 60/60 bitwise | 60/60 bitwise | Yes |
| `gated_delta_net_decode` 4-seq vs 4×1, 1 and 4 tokens | bitwise | bitwise | Yes |
| `gated_delta_net_decode` capture 4-seq vs 4×1; window vs sequential tokens | bitwise | **KernelNotFound** | Yes on precompiled. The runtime path cannot build `gated_delta_net_decode_capture_f32_*`: `KernelRegistry::new()` does not register their source (only `ops::gated_delta_net_decode::register` does) |
| Capture restore: blit recurrent + strided copy / transpose conv | bitwise, incl. all NaN payloads (strict) | same | Yes. The existing primitives suffice (P4a) |
| 16 rows 4×4 vs 2×8 | 4 weight passes both ways, every route and codec | same | n/a |

Bandwidth: 2×8 never reduces logical weight loads. With both `mul_mv_ext` and
mN, an 8-column dispatch runs as two 4-column groups, so 16 rows read the
weight 4 times either way. Median GPU times of the two groupings are within
noise in most rows (for example Q4_K `deltanet_qkv` 313.6 vs 316.6 µs, Q6_K
served `attn_output` 263.9 vs 264.5 µs). A few 4×4 rows ran slower, up to 25%
(Q4_K `ffn_gate_up` 794 vs 598 µs precompiled), but that did not reproduce
across paths (runtime: 547 vs 539 µs). Unique-weight bandwidth was 62–123 GB/s
and logical bandwidth 248–493 GB/s. Per-case numbers are in
`bandwidth-{precompiled,runtime}.txt`.

### Runtime-path capture kernels (superseded)

The `KernelNotFound` row above was a registry gap at 0.15.16:
`KernelRegistry::new()` did not register the
`gated_delta_net_decode_capture_f32_{1,2,4}` sources. The fix (`4e32003`,
released in 0.15.17 with its test `gdn_decode_capture_runtime_source`)
registers them, and the E6 rerun on this branch covers both capture probes on
the runtime path (see "P4a results"). The fix is not a P4a dependency: the
restore primitives never used those kernels.

## P4a: DeltaNet capture restore with existing primitives (path A)

Story: hf2q ADR-065 P4a (hf2q issue #526). E6 probe 4 showed that existing
copy primitives restore captured DeltaNet state bitwise, so **P4a is
satisfied by this record without a native kernel or release**. hf2q composes
the primitives itself (ADR-065 assigns encode-only helpers to hf2q).

### Contract (`DeltaNetCaptureRestore`)

Per DeltaNet layer: `rec_cap` is an f32 view of logical shape `[E, T, S]`
(E fastest) and `conv_cap` of `[S, T, Km1, C]` (C fastest). Each item `j` is
`(s_j < S, t_j < T, rec_dst[j], conv_dst[j])`, with destination views of
exactly `E` and `C·Km1` elements. For every `e < E`, `c < C`, `k < Km1`:

```text
bits(rec_dst[j][e])          := bits(rec_cap[(s_j·T + t_j)·E + e])
bits(conv_dst[j][c·Km1 + k]) := bits(conv_cap[((s_j·T + t_j)·Km1 + k)·C + c])
```

- Equality is on raw 32-bit patterns for every value, including signaling and
  quiet NaN payloads, ±0, subnormals and ±∞. No arithmetic or flush-to-zero.
- Preconditions, validated by the caller: destination views pairwise
  disjoint and disjoint from every source; all views inside their buffers'
  logical extents; `E`, `C`, `Km1 > 0`; no u32 index overflow inside a view.
- Cohort invariance: the bytes written for item `j` depend only on
  `(rec_cap, conv_cap, s_j, t_j)`, not on the other items, their order, their
  number (1–4) or their slots. Bytes outside the destination views are
  unchanged, including peer slots and the non-current parity buffer.
- Ordering: everything is encoded into a caller-provided `CommandEncoder`;
  the caller commits and waits. Sources written by earlier work in command
  buffer order are complete before the restore reads them. A GPU fault
  surfaces as `MlxError::CommandBufferError` from the wait.
- Anchor images: a capture-layout pair with `T = S = 1`, restored with
  `(s, t) = (0, 0)`, is inside the contract. The contract covers capture →
  state only.

### Composition

| Component | Primitive | Arguments |
|---|---|---|
| Recurrent | `CommandEncoder::blit_copy_bytes` | source `rec_cap` at byte offset `(s·T+t)·E·4`, destination `rec_dst`, length `E·4` |
| Conv re-index | `ops::copy::dispatch_strided_copy_f32` | `src = conv_cap.slice_view((s·T+t)·Km1·C·4, Km1·C)`, `dst = conv_dst`, `rows = C`, `cols = Km1`, `stride_row = 1`, `stride_col = C`, so `dst[c·Km1+k] = src[k·C+c]` |

`offset_copy_f32` and `transpose_2d` are not part of the composition.

**Encoding order.** hf2q encodes every conv strided copy for all layers and
items first (one compute pass), then every recurrent blit. Each
`blit_copy_bytes` ends the active compute pass and opens its own blit
encoder. The two families have disjoint destinations and read-only sources,
so no barrier is needed between them. E6 probe 4 interleaved the families;
T-P4a-3 covers the shipped order.

**Bounds-check caveat.** `blit_copy_bytes` validates against view extents
(`data_byte_len()`) and refuses overlapping same-allocation ranges.
`dispatch_strided_copy_f32` validates against `MlxBuffer::byte_len()`, the
whole Metal allocation, so it accepts a view too small for the copy as long
as the allocation is large enough, and writes past the view (T-P4a-6 pins
this). View-exact range checking is therefore the caller's preflight.

**C2 snapshot direction.** ADR-063 C2 snapshots state into an anchor image
with the same primitives in reverse: a blit out of the parity-current
recurrent region of slot `s`, and the inverse strided copy `rows = Km1`,
`cols = C`, `stride_row = 1`, `stride_col = Km1`, giving
`img[k·C + c] = state[c·Km1 + k]`. These are tests of existing primitives,
not a contract extension.

### Tests

All run on both metallib paths and assert the pipeline origin of
`strided_copy_f32` for the selected path. Every comparison is bitwise for
every u32 class, NaN payloads included.

| ID | Test | What is checked |
|---|---|---|
| T-P4a-1 | `adr065_e6_capture_restore_copy` (E6 probe 4) | Port of hf2q `rollback_la_to`: four slots, mixed parity, `t ∈ {0..3}`, one command buffer. NaN payloads asserted with `E6_STRICT_NAN=1` |
| T-P4a-2 | `adr065_p4a_capture_restore::p4a_acceptance_classes_match_host_rollback` | Full (no item: every byte unchanged), partial (`t ∈ {1, 2}`) and zero (`t = 0`) acceptance for every slot at the Qwen3.8 shape |
| T-P4a-3 | `p4a_compute_first_order_matches_host_for_every_ordered_slot_subset` | Compute-first order, one command buffer, all 64 ordered subsets of four slots (1–4 items) at `E = 786,432, C = 10,240`, `E = 524,288, C = 8,192` (`Km1 = 3`, `T = 4`) and a 2-layer small shape. Item regions equal the host restore (the single-item runs are in the sweep); peer regions and the non-current parity buffer equal their sentinel fill |
| T-P4a-4 | `p4a_anchor_image_restore_matches_host` | `T = S = 1` image, `(0, 0)`, into every slot under both parities; all other bytes unchanged |
| T-P4a-5 | `p4a_snapshot_direction_and_round_trip_are_bitwise` | State → image for slots 1–3 under both parities (inverse strided copy, then blit out of the parity-current region), then image → sentinel state; the round trip is byte-identical and touches no other bytes |
| T-P4a-6 | `p4a_native_refusals_and_strided_bounds_caveat` | `blit_copy_bytes` refuses overlapping same-allocation ranges and source/destination ranges past `data_byte_len()`; `dispatch_strided_copy_f32` refuses zero rows or cols, and accepts (and writes) an allocation-sized copy into a smaller view |

Records use `probe=p4a_capture_restore`.

### P4a results (M5 Max, 2026-11-03)

Run with `E6_SKIP_BANDWIDTH=1 scripts/adr065_e6_run_probes.sh …/p4a-526` at
source commit `1117708` (mlx-native 0.15.17, rustc 1.94.0, embedded
metallib sha256 `a4f35c4b…f961827`). Records, logs and provenance are in the
hf2q v0130 evidence directory `p4a-526/`. Every test binary passed on both
paths (`status.txt`: all zero).

| Records | Precompiled | Runtime |
|---|---|---|
| P4a T-P4a-2..6 (`probe=p4a_capture_restore`) | 30/30 bitwise; refusals as specified | same |
| T-P4a-3 compute-first sweeps (64 ordered slot subsets each; small 2-layer, Qwen3.6, Qwen3.8 shapes) | 3/3 sweeps, 0 failed runs | same |
| T-P4a-1 capture restore (E6 probe 4), `E6_STRICT_NAN=1` | 8/8 bitwise, NaN payloads 4/4 preserved | same |
| `gated_delta_net_decode` capture: 4-seq vs 4×1; window vs sequential tokens | 12/12 bitwise; 8/8 bitwise | same (registry fix) |
| Other E6 probes | verdicts unchanged from the E6 results above, including the recorded-only TQ-HB `nsg` 1→4 crossing rows | same |

The bandwidth (timing) probe was not rerun; its E6 numbers stand.
