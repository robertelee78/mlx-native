# Wave 5b.2 iter 1 — chunk-scan pipeline M5 Max baselines

**Captured:** 2026-04-27
**Hardware:** Apple M5 Max
**Build:** `cargo bench --bench bench_chunk_scan_pipeline` (release profile, criterion harness, 20 samples each, 1s warmup, 2s measurement window)
**Shape:** `B=1, T=4096, Hg=2, H=4, K=128, V=128, BT=64, NT=64` (long-prefill regime — the operating point where chunk-scan beats autoregressive)
**HEAD:** `4fd90b7` (Wave 5b.1 iter 4.5 close + iter 6 bench harness skeleton)

| Kernel | Median wall | Range (lo/hi) | % of pipeline |
|---|---:|---:|---:|
| `gated_delta_net_chunk_inter_state_bf16` | **18.674 ms** | 18.592 / 18.750 | 82.3% |
| `gated_delta_net_chunk_o_bf16` | **2.1546 ms** | 2.1488 / 2.1627 | 9.5% |
| `gated_delta_net_recompute_w_u_fwd_bf16` | **697.59 µs** | 694.27 / 700.95 | 3.1% |
| `gated_delta_net_kkt_bf16` | **526.22 µs** | 525.00 / 527.26 | 2.3% |
| `chunk_tri_solve_invert_f32` | **455.14 µs** | 453.97 / 456.17 | 2.0% |
| `chunk_local_cumsum_g_f32` | **161.27 µs** | 160.22 / 162.28 | 0.7% |
| **Total per orchestrator call** | **~22.67 ms** | | 100% |

## Optimization priority

`inter_state` dominates at 82% of pipeline cost. The kernel's outer-update
(`bh += k^T @ b_v`, BV*K f32 cells summed over BT k-tile rows = 64*32*128
= 262144 MACs per (b, h, chunk)) is a dense bf16 matmul — exactly the
workload `simdgroup_matrix<float, 8, 8>` MMA accelerates on Apple Silicon.
Target: ≥2× speedup on `inter_state` (Wave 5b.2 iter 1 RED test bar).

`chunk_o` is the second-biggest win at 9.5%; same matmul shape, same
optimization path, deferred to Wave 5b.2 iter 2.

The remaining four kernels combined are <8% of pipeline cost; deferred to
Wave 5b.2 iter 3 if they bottleneck after iter 1+2 land.

## Methodology notes

- Wall-time includes `command_buffer_commit + wait_until_completed` per
  kernel dispatch (the natural client-observable latency).
- Buffer allocation hoisted out of the `b.iter()` closure.
- First iteration warms the Metal pipeline cache.
- 20 samples × 2s window = ~600 dispatches per kernel — adequate for
  ±0.5% confidence at these shapes.

## Post-iter-1 results (2026-04-27)

Wave 5b.2 iter 1 simdgroup_matrix MMA optimization on `inter_state` —
both matmul sections (2b: `bv = u - w @ bh^T`, 2f: `bh += bv^T @ k`)
rewritten using `simdgroup_matrix<float, 8, 8>` 8×8 tile MMA on Apple
Silicon hardware tensor units.

| Metric | Pre-iter-1 (4fd90b7) | Post-iter-1 | Change |
|---|---:|---:|---:|
| `inter_state` median wall (criterion) | 18.674 ms | **1.073 ms** | **17.40× speedup** |
| `inter_state` median wall (RED test, 50 dispatches) | 18.674 ms | **1.157 ms** | **16.14× speedup** |
| Threadgroup memory | 24 KB | **20 KB** | −4 KB (bv_stage repacked f32→bf16, transposed [BV,BT]) |
| RED test bar (≥2× speedup ⇒ ≤9.4 ms) | — | PASS | margin: 8.0× under bar |
| Correctness `max_h_err` (5e-3 atol) | 0.0 | **0.0** | unchanged |
| Correctness `max_v_err` (5e-3 atol) | 0.0 | **0.0** | unchanged |
| Correctness `max_final_err` (5e-3 atol) | 2.98e-8 | **2.98e-8** | unchanged (deterministic 8×8 MMA reduction order) |
| FLA-line-255 oracle | PASS @ 0.0e0 | **PASS @ 0.0e0** | unchanged |
| Chunk-pipeline integration tests | 13/13 PASS | **13/13 PASS** | unchanged |

The 17× speedup (vs the projected ~3× theoretical) reflects the actual
arithmetic intensity reduction: the prior per-thread MAC loops were
memory-bound on threadgroup-memory bandwidth (each lane pulled the same
bh/k columns redundantly), while the simdgroup_matrix MMA shares a single
8×8 tile load across 32 lanes. The bf16 staging layout repack
(`[BT, BV]` f32 → `[BV, BT]` bf16) eliminates section 2f's transpose
overhead on the load path AND halves bv_stage's threadgroup-memory
footprint, freeing 4 KB for instruction-cache headroom.

## Section 2f-only result (prior worker, reverted)

Reference data point — for documentation only; this configuration was
reverted per the iter-1 STOP directive because section-2b's ~6 ms floor
was not addressed:

| Section 2f-only MMA (reverted) | 12.72 ms | 1.47× speedup |

The section-2b MMA reaches its full speedup because the K-reduction
(K=128 = 16 8×8 tiles) is bigger than 2f's BT-reduction (BT=64 = 8
8×8 tiles), so the simdgroup-matrix engine has more arithmetic to spread
across each shared-load. Both sections now run in MMA, eliminating the
floor.
