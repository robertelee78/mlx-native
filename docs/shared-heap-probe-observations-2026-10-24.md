# Bounded Shared heap observations — 2026-10-24

These ignored test controls investigate native issue #47. They leave production
allocators, pools, APIs and versions unchanged. The source base is
`33fc6de0547bd551a83991842d34cae07f5e4e5c`. The integration owner executed the
controls on Apple M5 Max, macOS 27.2 (26B5112a), SDK 27.0, 16 KiB host pages.

## Reproduction and success criteria

Run explicitly and serially on an otherwise suitable Apple Silicon host:

```sh
cargo test --locked --lib device::shared_heap_tests -- --ignored --nocapture --test-threads=1
cargo test --locked --lib device::shared_heap_gpu_tests::tracked_shared_heap_two_command_buffers_match_standalone_and_cpu -- --ignored --nocapture --test-threads=1
```

An external supervisor must enforce a process timeout for the GPU control;
its synchronous GPU waits are not a watchdog. Run the GPU control with
`MLX_UNRETAINED_REFS`, `MLX_PROFILE_DISPATCH`, `MLX_PROFILE_CB`,
`MLX_ENCODE_TRACE`, `MLX_PROFILE_BARRIERS`, `HF2Q_AUTO_BARRIER`,
`HF2Q_FORCE_SERIAL_DISPATCH` and `HF2Q_PIPELINE_TG_MULT_HINT` absent.

Exit zero alone is insufficient: unsupported configurations can explicitly
report unavailable/unknown and return without completing the control. Require
both CPU cases' full nine-phase observations, their observed hazard modes,
and their capacity-control records with `nil_observed=true`.
For GPU success require both `Q5_HEAP_GPU_COMMANDS` records with both completion
fields true, both `Q5_HEAP_GPU_ORACLE` records with exact bits equal, and the
final `Q5_HEAP_GPU_CAPABILITY` status `bounded_gpu_probe_completed` with
`exact_cpu_and_standalone_parity=true`. Preserve unavailable outcomes separately;
there is no Private or Untracked fallback.

## CPU ownership and capacity controls

Both Default and explicit Tracked Shared/Automatic configurations completed all
nine phases. Default resolved to hazard mode 1 (Untracked); explicit Tracked
resolved to mode 2 on heap and child. These are distinct controls: Default CPU
success does not establish Tracked GPU ordering.

Both requested and observed a 65,536-byte heap. The 4096-byte root had length
and allocatedSize 4096; size/alignment query was 4096/256. The root was fully
initialized before reading its known CPU pattern. Heap usedSize stayed 4096
and maxAvailable stayed 61,440 while wrapper, view, raw Metal or RecordedBinding
owners retained the root. After final resource ownership ended, usedSize was
0 and maxAvailable 65,536. Released resource fields were not queried.

A 65,537-byte request returned nil from the bounded heap; this tested heap-full
failure without host-memory exhaustion. Heap currentAllocatedSize remained
65,536 until heap release. Device currentAllocatedSize was 65,536 before heap
creation, 131,072 while retained, then 65,536 after heap release. These metrics
have overlapping scopes and must not be added as independent memory categories.

## Tiny GPU parity control

The explicit Tracked heap requested and observed 65,536 bytes. Four simultaneously
live, distinct, nonaliasable children each held 256 F32 values (1024 bytes).
The size/alignment query was 1024/256, giving a checked four-child packing of
4096 bytes. Heap and all children reported Tracked mode 2 and matching parent
identity. A standalone Shared/Tracked arm used the same device and queue.

Every logical byte was initialized: A contained signed integers -8 through 8,
B was 2, and T/O held sentinels. Both native operations were encoded before
submission. CB1 computed T=A+B; CB2 computed O=T*B. Consecutive same-queue
commits had no CPU wait between them. Both wait results were collected before
asserting success, with resources and command-buffer owners retained through
readback. All four command buffers succeeded. Both arms matched the independent
CPU oracle bit for bit for all four buffers, and matched each other exactly.
Final live-heap observations were usedSize 0 and currentAllocatedSize 65,536.

Automatic heapOffset 0 is a documented non-placement sentinel. It is neither a
placement offset nor evidence of disjoint byte intervals. The allocation
contract here uses distinct simultaneously live nonaliasable resources.

## Exact evidence identities

All values below are SHA-256. Historical default-only evidence remains separate
from the successor explicit Tracked controls.

| Evidence | SHA-256 |
| --- | --- |
| Original default-only CPU source | `1675db26200717df191c06908893308a010806f88960c733db95fb7e2783becf` |
| Original default-only CPU log | `f2bfd8bff2fd04b58307ca780bbe8cdd771576bdb09a9f86e288b42645bf53b7` |
| `src/device/shared_heap_tests.rs` | `3697b54fedba2d54cae38668dad3dc57d02ff7460cc621fcc482d2a9d31245a6` |
| Combined Default/Tracked CPU log | `0a8b7c64e5d212db7bad5b2d1c335fe32591982bce6bfb75f7b248fb9eb014ac` |
| Combined CPU observations | `8c1e5c4e5ea1f069343db1436cf355770aa425b45fbcf1d69be5e00c9fec9e7f` |
| `src/device/shared_heap_gpu_tests.rs` | `b60134f772a5e7b406392cb3df53671580ede6eba9ca20b4dc225a20e7089605` |
| `src/device.rs` module-registration source | `93dab44796089ce96d34712ae316d64700c5312c88107183dc6e990decb17cf6` |
| GPU test binary | `b8fe02f0bbc9e5460e1fe007c5b573efdd035f6d612fb60a4fd3d740c4acaa23` |
| GPU log | `6addb98e5b03b675d574243fad26071c8e05ffde511aa81c3eccbbba392fa3c7` |
| GPU observations | `72d4db98d1765697d4522df6692493f906161b70985af4f7c154ee43a8e4da7b` |

The GPU process exited zero without timeout and emitted the required completed
parity markers. Existing shader precompile warnings for `flash_attn_prefill`
and `flash_attn_train_fwd` persisted; those shaders were not exercised by this
control. This result does not qualify those paths.

## Limits

This is bounded CPU ownership/capacity evidence plus one tiny direct-binding,
same-queue GPU numerical control. It does not establish general GPU ordering,
indirect bindings, cross-queue behavior, model correctness, performance,
production-pool suitability or full Q5 admission. No models ran and no speed
claim follows. Requested heap capacity is distinct from driver/object metadata;
observed heap size and allocatedSize do not prove a universal physical
pre-allocation upper bound or OS footprint bound. Complete allocator-domain,
metadata, scratch and competing-memory bounds remain separate work.
