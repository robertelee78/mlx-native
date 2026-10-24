# File-backed Metal resource lifetime

The 0.15.2 candidate ties a no-copy Metal buffer's mmap backing to the resource's
own deallocator block. This preserves the existing public API, file offsets,
page-rounded resource lengths, logical view lengths and read-only CPU access.
No kernel arithmetic or model policy changes.

Previously only the shared MlxBuffer storage retained Arc<Mmap>. Ordinary
MlxBuffer clones and views were protected, but a safe raw Metal buffer clone or
RecordedBinding could retain the resource after that Rust storage disappeared.
The no-copy resource was created with a null deallocator, leaving no mapping
owner tied to those independent Metal references. This is a source-backed
missing lifetime guarantee, not a claim of an observed invalid GPU read or
crash.

## Bounded evidence and regression contract

Before the production correction, the six escrow tests produced the expected
four raw/recorded-owner failures: with all ordinary wrapper/caller owners gone,
Arc::strong_count was one, representing only the explicit safety escrow. The
ordinary clone/view and invalid-range controls passed. Every no-copy address
remained backed by escrow, including during assertion cleanup. The finding did
not require a GPU command or a post-drop payload read.

The current tests require independent mapping ownership while a raw Metal or
RecordedBinding owner survives, followed by escrow-only count one after the
final resource/residency/autorelease owners end, then Weak expiry when escrow
drops. The exact retained count need not be two. Default residency and explicit
per-device disabled residency are both covered without changing global
environment flags. Staged residency removal is explicitly flushed on normal
cleanup and assertion unwinding. Tests use a 4 KiB logical view at nonzero
offset, rounded to actual host pages (one 16 KiB page on the observed host),
not a hard-coded 4 KiB physical no-copy allocation.

Additional deterministic controls exercise the same production block-creation
boundary with a nil constructor result, and with a panic before resource
creation. Both require capture cleanup without relying on the deallocator
callback being invoked. These controls need no GPU allocation. The parent
integration owner records exact after-fix execution and package evidence;
adding the tests alone does not assert those results passed.

Run the witnesses serially:

```sh
cargo test --locked --lib device::mapped_backing_tests -- --nocapture --test-threads=1
```

## Capture and ABI

The callback is an Fn closure explicitly capturing an Arc<Mmap>. It neither
consumes that Arc on invocation nor frees/unmaps Metal's pointer directly; the
block dispose helper drops the capture after the last block owner. The callback
does not read payload, call Metal, log, acquire pool locks or invoke admission.
A compile-time bound checks Send+Sync for the actual closure capture. No unsafe
Send/Sync implementation is added for RcBlock.

The directly pinned block 0.1.6 dependency heap-copies ConcreteBlock and supplies
capture copy/dispose ownership. The local RcBlock remains alive throughout the
raw Objective-C call, including a nil return; normal Rust RAII releases it on
failure or unwinding. The nullable return is checked before constructing an
owned metal::Buffer, rather than using the binding's nonnullable wrapper. The
callback signature matches metal 0.33's Block<(*const c_void, NSUInteger), ()>
binding. The block never owns a Metal buffer, device or wrapper, avoiding a
resource/block cycle. Existing wrapper file_backing metadata ownership remains
as well; another Arc is not another mapping or extra resource capacity.

## Packaging and scope

Source library tests cover the witnesses. Packed CI and downloaded-registry
release gates separately select the library module, verify the source file is
present, and compare the nonzero listed test count to the source declarations.
Existing integration checks remain. The downloaded archive identity is checked
by the existing exact-package release process before testing it.

This correction does not establish unique resource-byte accounting, completed
command-buffer release, full model memory admission, numerical parity or any
speed win. In particular, it does not change the DeepSeek resource-capacity
lower bound or authorize model execution. Native publication and registry
verification must precede a reproducible downstream pin update.

## Candidate validation on Apple M5 Max

On macOS build 26B5112a, using Rust 1.94.0, the corrected candidate passed
all eight lifetime witnesses (zero ignored), four mapped-GGUF integration
checks, three command-buffer autorelease checks and two command-label checks.
The all-feature library suite passed 432 tests with two existing ignored tests.
The locked all-target/all-feature check, locked release build, reference policy
and package hygiene checks also passed. Private receipts bind source hashes
and logs; the before/after witnesses used no model or GPU dispatch. The
integration checks did exercise bounded Metal operations.

The local Metal compiler emitted existing precompile failures for
flash_attn_prefill and flash_attn_train_fwd, producing a partial metallib with
145 shaders and runtime-source fallback for two. These unrelated shader paths
were not validated by the lifetime witnesses; subsequent model gates retain
their own correctness and performance requirements. Public registry, MSRV and
CI results must be recorded separately before claiming publication readiness.
