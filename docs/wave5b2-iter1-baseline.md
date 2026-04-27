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

## Wave 5b.2 iter 2 results — chunk_o simdgroup_matrix MMA (2026-04-27)

Wave 5b.2 iter 2 simdgroup_matrix MMA optimization on `chunk_o` —
all THREE matmul sections rewritten using `simdgroup_matrix<float, 8, 8>`
8×8 tile MMA on Apple Silicon hardware tensor units:

* Section 2A: `bo_acc[BT, BV] += q · h^T` (K-reduction, 16 K-tiles)
* Section 2B: `bA_acc[BT, BT] += q · k^T` (FUSED into 2A's K-loop —
  same q-frag amortized across 4 BV-col + 8 BT-col MMAs = 12 MMAs/load)
* Section 2C: `bo_acc += bA_bf16 · v_new` (BT-reduction, 8 BT-tiles)

| Metric | Pre-iter-2 (cf5f420) | Post-iter-2 | Change |
|---|---:|---:|---:|
| `chunk_o` median wall (criterion) | 2.155 ms | **0.451 ms** | **4.78× speedup** |
| `chunk_o` median wall (RED test, 50 dispatches) | 2.155 ms | **0.453 ms** | **4.76× speedup** |
| Threadgroup memory | 28 KB | **8 KB** | −20 KB (bo_acc + bA_acc moved to MMA accumulator registers; only bA_stage `[BT, BT]` bf16 retained) |
| RED test bar (≥ 3× speedup ⇒ ≤ 720 µs) | — | PASS | margin: 1.59× under bar |
| Correctness `max_o_err` (5e-3 atol) | 5.96e-8 | **5.96e-8** | unchanged |
| FLA chunk_o oracle (1e-6 tol) | PASS @ 5.96e-8 | **PASS @ 5.96e-8** | unchanged |
| `chunk_gated_delta_rule_fwd` integration | max_o_err=3.815e-6 | **max_o_err=3.815e-6** | unchanged |
| Chunk-pipeline integration tests | 18/18 PASS | **18/18 PASS** | unchanged |
| `MAX_K` cap | 192 | **128** | NARROWED — K-tile count is hard-coded compile-time per iter 1.5 lesson |

### Pipeline post-iter-2 totals (criterion bench, 2026-04-27)

| Kernel | Pre-iter-2 | Post-iter-2 | Δ |
|---|---:|---:|---:|
| `inter_state` | 1.073 ms | 1.068 ms | unchanged (iter-2 did not touch inter_state) |
| `chunk_o` | **2.155 ms** | **0.451 ms** | **−1.704 ms (−4.78×)** |
| `recompute_w_u` | 0.698 ms | 0.702 ms | unchanged |
| `kkt` | 0.526 ms | 0.528 ms | unchanged |
| `tri_solve_invert` | 0.455 ms | 0.463 ms | unchanged |
| `cumsum_g` | 0.161 ms | 0.163 ms | unchanged |
| **Total per orchestrator call** | **~5.07 ms** | **~3.38 ms** | **−1.69 ms (−33%)** |

chunk_o was 41% of post-iter-1 pipeline cost; iter 2 collapsed it to 13%.
inter_state is once again the dominant kernel (32% of post-iter-2 total),
followed by recompute_w_u (21%), kkt (16%), tri_solve_invert (14%),
chunk_o (13%), cumsum_g (5%).

### Why chunk_o saw 4.78× (less than inter_state's 17.4× but still well
above the 3× target)

chunk_o has THREE matmul sections vs inter_state's TWO, so on a per-MMA
basis it should benefit MORE. The actual ratio is smaller because:

1. **Lower arithmetic intensity per K-tile.** chunk_o's section 2A is
   `[BT, BV] = [64, 32]` (8×4 = 32 8×8 frags total) vs inter_state's
   section 2b `[BT, BV] = [64, 32]` (same shape). But chunk_o's 2A only
   gets 4 frags per simdgroup (1 row-tile × 4 col-tiles), while
   inter_state's 2b gets 8 (2 row-tiles × 4 col-tiles) — so per-sg MMA
   accumulator depth is half, halving the FMA throughput coverage of
   load-issue latency.
2. **bA_stage round-trip** between sections 2B and 2C is a full BT×BT bf16
   threadgroup-memory write+read (8 KB roundtrip per (b, h, chunk, V-tile)).
   inter_state's bv_stage was BT×BV (4 KB roundtrip).
3. **Per-thread gate/mask** on bA_acc requires reading 16 g[t] values per
   lane (8 col-tiles × 2 elements) — twice the gate cost of inter_state's
   local_v gate.

These three together explain why chunk_o lands at 4.78× rather than 17×.
The ≥ 3× RED bar is comfortably cleared with 1.59× margin.

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
