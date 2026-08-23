# Dense Q4_0 exact-shape route calibration

Status: schema-4 hardware qualified; hf2q integration and publication pending
Updated: 2026-08-23
Source base: `1dc4bd3379c684f45bded1f260b9949bc1e5267a`

## Observable contract

The loaded artifact remains native Q4_0 bytes. Dense projection inputs and
outputs remain F32. Activation declares every reachable exact execution shape
and every distinct current native weight buffer. Calibration freezes one
immutable, pointer-free route plan in the model's `KernelRegistry` before the
model becomes ready. Undeclared shapes, late declarations, unsupported policy,
proof failures, timing failures, deadline exhaustion, and absent candidate
pipelines fail closed to the compatibility route or abort activation when the
required compatibility route itself cannot be proved.

The process cache owns timing distributions and their derived route only. It
never owns a weight or scratch buffer. Every activation, including A to B to A
at the same allocation address, reruns poisoned-output bitwise proofs against
the currently borrowed weight bytes before accepting cached timing.

Every `KernelRegistry` also owns a unique activation-authority ID. A route plan
is bound to that ID as well as the Metal device, build, and caller-supplied
activation epoch. A plan obtained from model A cannot be installed into model
B's fresh registry on the same device; dispatch fails closed if that invariant
is ever violated internally.

## Hypothesis and failed spike

The first implementation gave every `(M, N, K, current weight, route)` proof
its own command buffer and CPU readback. Full Cartesian coverage was correct,
but the submission protocol was not a usable activation contract under the 15
second budget.
The three preserved Cartesian receipts are:

| Profile | Native weights | Shape-weight pairs | Cold | Reactivation | Result | Receipt SHA-256 |
|---|---:|---:|---:|---:|---|---|
| 65 x 3 structural 768/3072 | 195 | 2,145 | 7,426.039 ms | 7,661.291 ms | Within budget | `667f7fed6564a8e217390b47eae6c25e5bfd733bb4687cea62562316b0632123` |
| 28 x 3 encoder 1536/8960 | 84 | 924 | 8,298.282 ms | 8,083.296 ms | Within budget | `2a050f7ab115fa906d646d12a7f51f0862645cc07e1774dcbdf1dc4c4f1df39d` |
| 64 x 3 dense 5120/17408 | 192 | 2,112 | 15,000.707 ms | 15,022.223 ms | Falsified: deadline exhausted; every candidate downgraded | `d04c47b6849f863166e9c20fcf792aba78cb3c7222d710fcf9f2fc0409f74047` |

The receipts are deliberately retained under
`target/bench-receipts/q4-activation-cartesian-*-20260823.json`; subsequent
batched-Cartesian runs use different filenames. The failed production-width
result is evidence, not a shippable route.

## Reformulated proof

An attempted factorization into one representative proof per exact `M` plus
one canonical-`M` proof per current weight was rejected before publication.
`n_mm_block_x` changes the opaque cooperative-matrix descriptor, so the
weight×`M` cross-term is not proven separable. A substring canary over template
instantiations also cannot establish semantic identity. No factorized receipt
is accepted as authority.

The accepted reformulation retains full Cartesian coverage and removes only
avoidable command-buffer/readback overhead. For each exact shape:

1. Allocate one adversarial F32 input, two guarded F32 output buffers, and one
   U32 status per distinct current weight.
2. In one command buffer, for every current weight, GPU-poison both outputs,
   explicitly barrier, encode both production routes, explicitly barrier,
   compare every logical bit plus finite/full-overwrite/guard invariants into
   that weight's status, then explicitly barrier before scratch reuse.
3. Count the command-buffer attempt before commit, wait once, and reject any
   nonzero compatibility status as a hard activation error. Any candidate
   status disables the candidate for every exact `M` of that base shape.
4. Time Direct/V2 versus Tensor64x32 once per exact `M` on the representative;
   timing never replaces any current-weight proof.

For `W` current weights and `R` row counts, production proof dispatches remain
exactly `2 * W * R`; the mechanism does not weaken correctness. Proof command
buffers fall to `R`, with `2 * W * R` auxiliary poison/compare dispatches and
one bounded scratch allocation per exact shape. Receipts expose authorized
shape-weight pairs, proof command-buffer attempts, production-route
dispatches, auxiliary dispatches, proof GPU time, peak scratch bytes, timing
attempts, and the final cleanup submission separately.

The initial production spike used 64 distinct Qwen-width weights at
`(M,N,K)=(32,17408,5120)`: 64 authorized pairs, 128 production-route plus 128
auxiliary dispatches in one proof command buffer, 5,112,192 scratch bytes, all
statuses zero, 132.821 ms cold activation and 43.891 ms timing-cache
reactivation. Its preserved receipt is
`target/bench-receipts/q4-activation-cartesian-batched-spike-expand5120-17408-m32-64w-20260823.json`
with SHA-256
`2d867ee511c684bdf11b58fa32c4475241a286770cfbdabfffe5a36683635f52`.

The complete schema-3 full-Cartesian qualification then exercised every declared
weight at every reachable exact `M`. Allocation is reported separately from
the bounded calibration call because the benchmark manufactures distinct
native weight buffers before activation:

| Profile | Native buffers / bytes | Authorized pairs | Cold / warm calibration | Cold proof GPU | Proof CB / route / aux | Peak scratch | Result | Receipt SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 64 x 3 dense 5120/17408 | 192 / 7.361 GB | 2,112 | 10,437.905 / 10,473.438 ms | 9,355.131 ms | 33 / 4,224 / 4,224 | 654.3 MB | 23 exact short-row candidates; M=2048/4096 remain V2 | `c58e72b49ba4f5495124c71213f278f465af208fe4e7521089345f4ece376d1d` |
| 28 x 3 encoder 1536/8960 | 84 / 470.7 MB | 924 | 974.115 / 754.516 ms | 661.034 ms | 33 / 1,848 / 1,848 | 318.8 MB | 19 exact candidates; zero overrun | `bcbb9899422ce830a0bd5ab1c2130323640d079b71dd1cd36df67804da2c4f5a` |
| 65 x 3 structural 768/3072 | 195 / 194.1 MB | 2,145 | 500.568 / 401.247 ms | 333.664 ms | 33 / 4,290 / 4,290 | 113.2 MB | 12 exact candidates; zero overrun | `262a83f62cb9c1008aca9f53fb09f66690e4eae1a1a973569b51cd26fc46dc4a` |

The Qwen profile allocated its 7.361 GB of synthetic native weights in
3,692.113 ms outside calibration. Cold runs add 330 timing submissions and
warm timing-cache reactivations add none; both still perform all current-byte
proofs. Every profile recorded exactly one final cleanup submission and zero
deadline overrun. These schema-3 receipts remain valid historical performance
measurements for the exact cells shown, but they predate registry-local
activation authority and the executable publication gate. They are not
schema-4 publication authority.

The source-complete schema-4 rerun on the same Apple M5 Max passed the
executable publication gate for all three complete named profiles. Each cold
shape decision used five samples per route and each new-registry reactivation
reused timing distributions only after reproving every current weight:

| Profile | Authorized pairs | Cold / warm wall | Cold proof GPU | Proof CB / route / aux | Peak scratch | Candidate cells | Receipt SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| 64 x 3 dense 5120/17408 | 2,112 | 11,025.079 / 11,256.451 ms | 9,828.049 ms | 33 / 4,224 / 4,224 | 654.3 MB | 25; all six M=2048/4096 cells V2 | `321f315f06011ecb00df3d3f04d68226e9969db637890c110006abdc44659bd8` |
| 28 x 3 encoder 1536/8960 | 924 | 993.922 / 759.621 ms | 661.261 ms | 33 / 1,848 / 1,848 | 318.8 MB | 22; all six M=2048/4096 cells V2 | `ecd41fdf6efef8dca09ec6a780d3b0353bd7f1a8b51d00ad5ba0e1fdd6ed53da` |
| 65 x 3 structural 768/3072 | 2,145 | 557.287 / 398.557 ms | 379.812 ms | 33 / 4,290 / 4,290 | 113.2 MB | 11; all six M=2048/4096 cells V2 | `ab321797845d5923924d97be5fa75c0256ddb05a055bcf01a24e5b5e9d7d3331` |

All six cold/warm receipts recorded schema 4, distinct registry authority IDs,
one cleanup submission, complete expected counts, stable exact-shape routes,
and zero deadline overrun. The common build fingerprint is
`1259d410c2ffcd5f4fad08a0e2ddf0f237d7ea45b42cd9ac7dc697577539cfee`.
This qualifies the exact named Cartesian profiles on that hardware; it is not
a universal row-count cutoff or an hf2q integration result.

## Reproducible gates

The focused correctness gate is:

```text
cargo test --locked --release --test dense_q4_auto_calibration
cargo test --locked --release --test q4_mm_tensor_64x32
cargo test --locked --test q4_benchmark_contract
```

The first two targets are tensor-hardware gates. The build enables them only
when it successfully compiles the Q4 tensor shader into the current artifact;
an older SDK that lacks the exact `<metal_tensor>` header still runs the
model-free selection, downgrade, receipt, and capability-classifier tests.
That hosted-safe path is not publication authority. Publication still requires
the complete named hardware gate above on a build that produced the tensor
shader.

The activation-tax harness fixes reachable rows to
`9,16,24,32,48,64,96,128,129,2048,4096`, uses exact production projection
widths and representative layer multiplicity, runs a cold activation and a
new-registry timing-cache reactivation, and writes the full JSON receipt:

```text
MLX_Q4_CALIBRATION_PROFILE=qwen_dense_5120_17408 \
MLX_Q4_CALIBRATION_RECEIPT_PATH=target/bench-receipts/q4-activation-cartesian-batched-qwen5120-17408-schema4-20260823.json \
cargo bench --locked --bench bench_q4_activation_calibration
```

Equivalent profiles are `encoder_1536_8960` and `structural_768_3072`.
Performance acceptance requires zero deadline overrun, bitwise proof success,
truthful submission totals, one cleanup boundary, and at least one surviving
exact-shape candidate decision. It also requires every declared physical row
width for every base shape, stable cold/reactivation decisions, and V2 at the
declared long-row cells (`M=2048/4096`). The harness now enforces this contract
before writing a receipt and exits nonzero for fallback-only, partial,
over-budget, zero-work, or count-inconsistent evidence. A timing-cache hit
never substitutes for the current-weight proofs. Layer, projection, row, and
budget environment overrides are diagnostic only: the publication gate remains
pinned to the complete named profile and 15-second ceiling, rejects reduced
runs, and never allows a budget override to expand that ceiling.

## hf2q integration contract

At model activation, hf2q groups every native Q4_0 dense projection by exact
base `(N, K, batch, input layout)`, retains every distinct layer-local borrowed
buffer in the declaration, unions the scheduler-reachable `M` set, and calls
`calibrate_dense_q4_routes` once with a new nonzero activation epoch. It waits
for calibration, verifies the receipt and frozen plan, and only then publishes
model readiness. A new model gets a new `KernelRegistry`; model A returning
after B may reuse process timing metadata but must reprove A's current buffers.
Arbitrary tails or undeclared runtime shapes remain on CompatibilityV2. The
integration must deduplicate by `(N, K)` base shape, not by codec alone, and
must never retain borrowed weight pointers in a process-global plan or cache.
