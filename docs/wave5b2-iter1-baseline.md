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
