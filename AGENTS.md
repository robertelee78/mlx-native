# mlx-native contributor and agent guide

## Project identity

mlx-native is a Rust Metal compute library for ML inference on Apple Silicon.
It owns GPU buffers, command encoding, kernel registration, GGUF-backed weight
views, and reusable Metal operations. It is not a TypeScript application and
does not use Node.js for its build or runtime.

The crate targets Rust 1.82 or newer. `Cargo.toml`, `Cargo.lock`, the Rust
source, Metal shaders, tests, and checked-in technical notes are authoritative.

Start with:

- `README.md` for the public API and execution model.
- `src/device.rs`, `src/buffer.rs`, and `src/encoder.rs` for resource and
  command-buffer contracts.
- `src/kernel_registry.rs` and `src/ops/` for dispatch ownership.
- The matching shader under `src/shaders/` before changing a dispatcher.
- The focused parity or performance test under `tests/` before changing a
  kernel used by production inference.

## Relationship with hf2q

mlx-native and hf2q are related but have a strict ownership boundary:

- mlx-native owns family-neutral Metal primitives, buffer/resource safety,
  command scheduling mechanisms, GGUF tensor views, and kernel correctness.
- hf2q owns model-family graphs, chat and tool semantics, KV-cache policy,
  conversion, quantization, serving, and operator configuration.
- A family-specific optimization belongs here only when it is a genuine Metal
  primitive with a tested public contract. Policy switches and model behavior
  stay in hf2q.
- hf2q must consume a published mlx-native crate version. A local Cargo patch
  is valid for a spike, never for a landed or released hf2q result.
- When a change spans both repositories, publish and verify mlx-native first,
  then pin the exact released version in hf2q and validate from a clean tree.

## Non-negotiable correctness contracts

- Coherent output and byte/numerical parity take precedence over throughput.
- `MlxDevice::alloc_buffer` zero-fills fresh shared allocations. Metal may
  recycle pages containing prior data; code that only partially writes a
  buffer depends on this zero-fill for correctness.
- An overwrite allocation path may skip zero-fill only when its producer
  writes every byte before the buffer becomes observable. Failure paths must
  drop the allocation rather than expose partial or recycled contents.
- File-mapped GGUF weights are read-only logical views. Honor their byte
  offset and logical length; never expose page-alignment prefix/suffix bytes.
- A Metal command queue orders submitted work. It does not make an
  unsubmitted producer visible. Every cross-encoder consumer needs a proven
  submission/fence contract.
- Buffer dtype, shape, byte length, alignment, offsets, residency lifetime,
  and CPU/GPU synchronization are API invariants, not debug assertions.
- Raw byte copies must preserve every payload bit, including packed quantized
  data and NaN payloads.
- Production library code does not launch Python, MLX, or any other
  inference runtime. Reference programs are acceptable only in explicit
  benchmark/parity harnesses.
- Reference-implementation policy: code, comments, and docs do not name
  external peer engines. Say "the reference implementation"; comparison data
  and derived-kernel provenance live only in `docs/peer-benchmarks.md`
  (legal notices: `LICENSE-MIT-llamacpp`, `LICENSE-APACHE-candle`).
  Enforced in CI by `scripts/check_reference_policy.sh`.

## Working method

Apply the project Kata:

- Need it? If no, it is out of scope.
- If yes and possible, implement it. If the path is unknown, research and
  measure until there is an evidence-backed path.
- Complete the loop: hypothesis -> smallest spike/test -> reformulate from
  measurements -> update the governing note or API documentation -> implement
  -> prove correctness and performance -> commit -> push -> merge.

Before editing:

1. Inspect `git status`, current dependency/release state, relevant source and
   shader, focused tests, and any active hardware process.
2. State the observable contract and the failure path.
3. Add or strengthen the smallest parity test that can falsify the change.
4. Keep dispatch validation in Rust and kernel bounds defense in Metal.
5. Run focused tests before broad checks and real-model downstream gates.

Do not land a failed performance spike. Revert it from the integration diff
and retain its exact measurements in the relevant technical note or evidence
ledger.

## Build and validation

Use locked dependencies:

```bash
cargo check --locked --all-targets --all-features
cargo build --release --locked
cargo test --locked
```

During iteration, prefer the narrowest applicable command, for example:

```bash
cargo test --locked --test test_buffer_overwrite
cargo test --release --locked --test test_flash_attn_prefill
cargo test --release --locked --test bench_deepseek_sparse_attention -- --ignored --nocapture
```

Metal tests require Apple Silicon. A compile-only check on another target does
not prove kernel correctness, synchronization, or performance.

## Performance and parity rules

- Establish a source-bound baseline before optimization.
- Compare identical tensors, dtypes, shapes, quantization, prompts, context,
  sampling, batch geometry, and hardware state.
- Warm pipeline compilation and allocator state where the production path is
  warm; retain cold-start measurements when startup is the contract.
- Report multiple-run medians and dispersion, not a favorable single run.
- Pair wall-clock timing with GPU command-buffer timing and dispatch/sync
  counts. Host stage timers can attribute queue backpressure to the wrong
  operation.
- A lower command-buffer count is not itself a win. Accept only measured
  end-to-end improvement with unchanged parity.
- Bind claims to the exact mlx-native commit and downstream hf2q commit or
  artifact used for validation.

## Code and git hygiene

- Preserve unrelated local changes. Use an isolated worktree for experiments.
- Avoid whole-tree formatting; format only edited Rust files.
- Keep Rust dispatchers and their Metal shader changes in one coherent commit
  with parity tests.
- Never commit model artifacts, profiles, generated metallibs, target output,
  secrets, machine configuration, or local memory databases.
- Use focused conventional commits such as `fix(buffer): ...`,
  `perf(attention): ...`, and `test(metal): ...`.
- Do not add `Co-Authored-By` trailers unless the repository explicitly
  authorizes them.
- `main` is the integration branch. A local commit is not a published crate;
  a crate release is not consumable by hf2q until the registry artifact and
  checksum are verified.
