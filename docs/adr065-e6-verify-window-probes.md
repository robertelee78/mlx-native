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
cd /opt/mlx-native-worktrees/adr065-e6-probes
scripts/adr065_e6_run_probes.sh /Users/robert/.local/state/hf2q/v0130/509-e6
```

The script runs the correctness probes, then the bandwidth probe, once per
metallib path. It writes `records-{precompiled,runtime}.txt`,
`bandwidth-{precompiled,runtime}.txt`, the logs, and a provenance file
(source commit, test-binary sha256, embedded metallib sha256, crate version).

## Results

Not yet run. The probes are written and compiled. GPU execution on the
M5 Max waits for the GPU lane. The result tables below are filled from the
record files of that run.

| Primitive | Precompiled | Runtime | Serves verify windows bitwise? |
|---|---|---|---|
| `mul_mv_ext` 8 vs 4 (Q4_K/Q5_K/Q6_K) | pending | pending | pending |
| Served dense routes 8 vs 4 | pending | pending | pending |
| TQ-HB batched, in-bucket / crossing | pending | pending | pending |
| TQ-HB banked, in-bucket / crossing | pending | pending | pending |
| F32 `flash_attn_vec` window | pending | pending | pending |
| `gated_delta_net_decode` 4-seq vs 4×1 (± capture) | pending | pending | pending |
| Capture restore: blit + strided copy / transpose | pending | pending | pending |
| 16 rows 4×4 vs 2×8: passes, GB/s | pending | pending | n/a |
