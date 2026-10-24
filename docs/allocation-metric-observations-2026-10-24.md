# Bounded allocation observations on Apple M5 Max

This ignored, test-only probe is based on native commit
`7bb30a2fac783e0b340ec2f2086289f9751dda7a` plus the allocation metric module.
It changes no allocator, public API, kernel or production setting. The host
was Apple M5 Max, 128 GiB, macOS build 26B5112a, with 16384-byte host pages.
The probe module SHA256 was
`61c96fbca92bc9bb0276bb41ae6e97ecbc0a6a4a1e78e7d939bcba12ada4a2a0`.

One explicit run passed and emitted 56 observation rows across eight cases,
plus two successful mapped-backing teardown records. Default residency was
active; the other mode was explicitly disabled without changing global
environment state. No model, payload read, encoder or GPU dispatch was used.

## Observed root quantities

The following root-phase values were identical in both residency modes.
All sizes are bytes; the device delta is relative to that case's baseline.
Both queried metric selectors were available on this host.

| Kind | Logical extent | Metal root length | Resource allocatedSize | Device delta |
| --- | ---: | ---: | ---: | ---: |
| Anonymous | 1 | 1 | 1 | 0 |
| Anonymous | 4096 | 4096 | 4096 | 0 |
| Anonymous | 16385 | 16385 | 32768 | 32768 |
| Mapped view at offset 64 | 4096 | 16384 | 16384 | 16384 |

Every case's observed device baseline was 458752 bytes. These observations
do not explain or bound allocator/driver overhead. In particular, the small
resources report nonzero allocatedSize without a device-total increase; the
16385-byte resource reports more allocated size than logical length. A logical
view, a resource observation and a device aggregate are distinct quantities.
Neither a universal rounding rule nor additivity follows from this table.

Clones and a one-byte view preserved the same root observations. Raw Metal
and recorded binding owners were kept alive after the ordinary wrappers ended.
Only live strong owners were used for resource-property reads. After the final
explicit owner, resource fields were null and only the device and backing
escrow were sampled. The large anonymous and mapped cases returned to their
case baselines after final owner cleanup. Mapped escrow counts returned to one,
then Weak expired after escrow release. This does not prove final-release timing
for submitted commands, other shapes or arbitrary external owners.

## Reproduction and limits

```sh
env -u HF2Q_NO_RESIDENCY cargo test --locked --lib \
  device::allocation_metric_tests::bounded_raw_allocation_observations \
  -- --ignored --nocapture --test-threads=1
```

The eight requested roots total at most 12 host pages, below 1 MiB, and each
case has one explicit requested root. Opaque driver retention can overlap
cases. The requested envelope is not a bound on actual allocator/driver memory.
Mapped addresses remain backed by an outer escrow through owner, residency,
autorelease and failure cleanup. Source-bound raw logs and execution receipts
are retained locally; log SHA256 is
`0b592670bfbd3281d18d9dbb1aedb721352a44cb9f96316b88a09f1f1d77a774`.

This is metric availability and ownership evidence, not calibration of a
preallocation upper bound, complete root coverage, process footprint, model
admission or performance. A bounded ownership domain still needs separate
root-count/metadata limits, all allocation/import paths, sticky Unknown on
uncovered interop, safe reuse and lifetime charges. Wrapper drop and command
completion must not manufacture a capacity credit.
