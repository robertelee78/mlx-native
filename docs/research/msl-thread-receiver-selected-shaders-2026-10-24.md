# Explicit thread receivers for selected attention shaders

This change fixes the selected MMATile receiver address-space mismatch in
`flash_attn_prefill.metal` and its training-forward sibling. It preserves
fragment references/lvalues, indexes, math, precision, tile layouts and function
constants. It adds no casts, by-value fragment copies or MSL language downgrade.
Source base: `09ace9cb6d9ed39e1512959cefa7b7d633f1a9a8`.

## Observed compiler sequence

On Apple M5 Max, macOS27.2 (26B5112a), SDK27.0, Apple Metal32023.921
(metalfe-32023.921.6), both original files failed offline compilation with exit1,
no AIR. The diagnostic rejected binding the thread fragment reference from a
method with an unqualified/default-address-space receiver.

The first patch qualified mutable/const frag_at receivers only. Both files still
failed with exit1/no AIR: actual template instantiations identified callers with
default receivers invoking those thread-qualified accessors. The final patch
also qualifies exactly those compiler-identified MMATile methods: row_reduce,
row_bin_op, load(threadgroup), store(device), store_safe(device). Mutable methods
use thread; const methods use const thread. There are seven receiver qualifiers
per file,14total. Uninstantiated unrelated methods remain unchanged.

Both propagated files then compiled with exit0, empty diagnostic logs and nonempty
AIR (prefill98,400bytes; training38,144bytes). Linking these AIRs succeeded:
selected metallib251,012bytes, SHA256
`49e3c4eb3cc82ec706ec301b9cdd55004a40f6b151cb1f82ccf237143ff52061`.
A Rust build that tolerates missing shader files would not establish this result;
this evidence is specific to the two selected files.

| Shader | Original SHA256 | First patch SHA256 | Propagated SHA256 |
| --- | --- | --- | --- |
| prefill | `7cb04ff1e3bdedbaa40bb33c9d1e34242aff42fd89658c00bb4d4203d406ffae` | `110cc1c9ebddbe63121a75b267eb3445f08a4637ce278fc4589c098696341995` | `39fc455b5f7b1bcb18752252e281ebec673f2a5010fdc64a6f594403552e04e4` |
| training forward | `6f9f2677c3bf85505d099377af91d0d07819369f0b7bfd262b01044562706104` | `2c1a94feb3124cb2066cf8e39afdb5b1c09c0136af1970ffc24d1e64ad72007c` | `31d1a6738b570516419669664cd914e2e2f89c23b602d1f9498352910ad10ff8` |

Portable offline commands (keep source/AIR outputs outside the checkout):

```sh
env -u MACOSX_DEPLOYMENT_TARGET xcrun -sdk macosx metal -O3 -c src/shaders/flash_attn_prefill.metal -o /tmp/selected-prefill.air
env -u MACOSX_DEPLOYMENT_TARGET xcrun -sdk macosx metal -O3 -c src/shaders/flash_attn_train_fwd.metal -o /tmp/selected-train.air
xcrun -sdk macosx metallib /tmp/selected-prefill.air /tmp/selected-train.air -o /tmp/selected.metallib
```

## Forced runtime-source and numerical controls

All five exact controls passed, each in a separate supervised process with exactly
one test, zero failures/ignored and no timeout. These are actual runtime-source
results, not an inference from offline compilation or an embedded library.
The test registers the production source plus a harmless trailing comment under
the SAME function name. Changed bytes force the registry's runtime-source path;
actual pipeline origin/hash is checked before and after numerical execution.
Embedded-metallib identity must be absent. The exact runtime sources were:

- Prefill62,093bytes: `446f78817077990516c4afecd25cf28f2b88c84233ed4352d9743f25ae73056f`.
- Training43,687bytes: `4c35083428b98400c4315ea9f18de2b8c4b3f52a3ca2bbc24606571436b02709`.

| Control | Shape/result | Log SHA256 |
| --- | --- | --- |
| selected_prefill_runtime_source_required | aligned/unmasked and ragged/causal D256 specializations compiled | `d43d03d264cb46141746da1084cbd5d5f29802b87c9ecbabc4fe47acb24560e6` |
| selected_qwen38_h24_kv4_d256_causal | B1,H24,KV4,S33,D256 causal; CPU oracle passed | `2a77fee462cf2100e4b817d819e588cf08c34ed94eccb6d3d3e969446216503f` |
| selected_qwen36_h16_kv2_d256_causal | B1,H16,KV2,S33,D256 causal; CPU oracle passed | `be1a778bf76dca60d7918e284e8972949b375dd50df0d8604af91ae0c56864d9` |
| selected_train_runtime_source_required | all four public BF16 forward entrypoints compiled | `99ac8ebf4ce7e1a9bd7c067aa8efe9efb4af4a4d076b7792262a4dd569230f5c` |
| selected_train_d256_causal_output_and_lse | B1,H2,KV1,S17,D256 causal O/L CPU oracles passed | `1b44b442c12890a549447445ff443c3efc3ee52dde3265a2bdfe90ecded14664` |

Both Qwen-shaped primitive cases had max absolute error1.953e-3 and max relative
error7.092e-3, within unchanged existing atol5e-3+rtol2e-2. Training O error
1.9531e-3<=5e-3; L error2.0046e-3<=3e-3. Existing CPU references are reused;
no shader-math emulator or post-hoc tolerance change was added. Qwen head counts
match frozen artifact metadata, but inputs are initialized synthetic tensors.

Every Q/K/V byte is initialized; O starts at finite1024 and L at8192. A checked
command wait precedes CPU reads; output must be finite with no sentinel remaining.
Explicit success markers and runtime source hashes are mandatory; a zero-filter
run, skipped hardware case or warning-only build cannot substitute.

| Numeric case | Requested Metal payload bytes | Conservative scoped host+Metal packet bytes |
| --- | ---: | ---: |
| Qwen3.8 head geometry |946176 |9773248 |
| Qwen3.6 head geometry |608256 |6662272 |
| Training forward |52360 |1525528 |

Each packet is below16MiB, including conservative input/output/oracle casts,
per-query score allocation requests and source/hash allowance. They are separate
process bounds, not a combined whole-suite claim. Opaque compiler/framework/
allocator/device/pipeline/residency metadata remains Unknown; none is a physical
preallocation upper bound or memory-admission result.

## Reproduction and validation identities

Run each exact filter separately with an external process-group watchdog and the
test's profiling/capture/unretained-reference environment controls absent:

```sh
cargo test --locked --test test_flash_attn_prefill flash_attn_prefill_tests::selected_shader_runtime::selected_prefill_runtime_source_required -- --exact --nocapture --test-threads=1
cargo test --locked --test test_flash_attn_prefill flash_attn_prefill_tests::selected_shader_runtime::selected_qwen38_h24_kv4_d256_causal -- --exact --nocapture --test-threads=1
cargo test --locked --test test_flash_attn_prefill flash_attn_prefill_tests::selected_shader_runtime::selected_qwen36_h16_kv2_d256_causal -- --exact --nocapture --test-threads=1
cargo test --locked --test test_flash_attn_train flash_attn_train_tests::selected_shader_runtime::selected_train_runtime_source_required -- --exact --nocapture --test-threads=1
cargo test --locked --test test_flash_attn_train flash_attn_train_tests::selected_shader_runtime::selected_train_d256_causal_output_and_lse -- --exact --nocapture --test-threads=1
```

Locked library all-feature check passed; both selected integration targets compiled
with --no-run. Their only reported warning was the existing deliberately retained
heap field's dead-code warning; the owner was preserved. The checks are not
numerical or all-shader qualification. Pre-version locked all-target/all-feature check also passed (5.55s).
The integration owner then changed only the Cargo package version to0.15.3.
For that package candidate, all-feature library tests passed441/zero failed/eight
ignored (1.47s) and the locked release build passed (12.53s). Reference/package
hygiene and rustfmt checks on the three new support files passed. Ignored library
counts remain distinct from the five explicitly executed integration controls.
Build durations are validation records, not performance measurements. The release
also contains the previously landed optional bounded-pool API; HF has not adopted
that API merely because it accompanies this package candidate.

Exact evidence identities, SHA256:

| Evidence | SHA256 |
| --- | --- |
| Original/first-patch offline receipt | `96b236592e6557b147a53c8da2488a6eb6193365f83fc73e7f4b354929bd4708` |
| Propagated offline receipt | `1e97f2d4f68f2d8ec2c85ed593f1d22bc046ee44c8dd976a7dc1d4c6f8e4b6fd` |
| Library check log | `54fab86aa866a7a8c551e0c8f14663083c14e10d8f5e26ecb481aa73c1550712` |
| Integration --no-run log | `3d9bd2aa49509781370ff2cc8803f7e6ead2c4f07d43b6573b73e7e31f49ddab` |
| Pre-version all-target/all-feature check log | `eef2f078c597de6305f8d45fbe15f48a6f8cf9e31247c8a0bad537522d1d34d4` |
| 0.15.3 all-feature library tests log | `2d009dcb7ef7646a82864a6a96aa032f43ce4d415a72f73c13f619c2d3c48924` |
| 0.15.3 release build log | `796ab5dfacb336d171556d687d6e92be2072c3bbabbf06c3922631fdb0113fe2` |
| Prefill test executable | `69789e533c00d3ee718a73adad606073fe21e431da585b80f1286a6342817c4c` |
| Training test executable | `e55490ba5fa41883bfaa8ca6c0f18f85833bad3be529a2c57ae037a76e60f4ff` |
| `src/shaders/flash_attn_prefill.metal` | `39fc455b5f7b1bcb18752252e281ebec673f2a5010fdc64a6f594403552e04e4` |
| `src/shaders/flash_attn_train_fwd.metal` | `31d1a6738b570516419669664cd914e2e2f89c23b602d1f9498352910ad10ff8` |
| `tests/test_flash_attn_prefill.rs` | `eae25ebe839190d053d4a6e295bd2f3ea319f0d7bd82eec67f24dc533f4afc71` |
| `tests/test_flash_attn_train.rs` | `a034ecd130cb513da8fac757f3bcfd04afade2aa42415c3f7cb49637e6a36b1d` |
| `tests/support/selected_runtime_common.rs` | `6d48a0ea38a2231d718e1099df52f3625a56a371be9016b1f132fc0d2e048995` |
| `tests/support/selected_prefill_runtime.rs` | `eaf85397723f6894bd2f69146a8a0264d49389a6cea9c41e2d328db8a3fd0efd` |
| `tests/support/selected_train_runtime.rs` | `3c0f16b801c14a5c70001f1a4538c530cb921ff80367788ad1e185ad858c456e` |


These results establish the selected compiler fix and five narrow runtime-source/
primitive cases on this toolchain. They do not prove HF seq-major bridge/model
output parity, training/backward generally, all shaders, a general backend fix,
speed, full memory bounds or full Q5. DeepSeek D512 and current Gemma vector paths
are separate readiness questions. Native publication/registry verification and
an exact HF pin are still required before downstream landed adoption; no ignored
local Cargo patch is a release proof.
