# M5 chunk-o gate warmup calibration

The existing25-dispatch warmup produced both passing and failing runs on the
published0.15.5 source and the opt-in requested-root candidate. The original
full-suite failure at0.829ms remains a failed result, not a retroactive pass.

A prospective fixed32-process experiment compared exact stocke772b31 and
candidate580904 at25/250 warmups, with eight balanced four-treatment blocks.
All runs retained50 timed commit-and-wait dispatches, identical shapes and the
0.720ms ceiling. Raw ordered samples were printed only after timing. Every
pre/post observation was nominal thermal state, low-power disabled, normal
memory pressure and the same AC source on M5Max/macOS27.2 build26B5112a.

Stock25 passed6/8 and candidate25 passed4/8. Both250 settings passed8/8:
stock medians435–470us, candidate446–518us. All251 retained artifact hashes
and32 raw outcomes were independently verified. These results support a
warmup-methodology improvement; they do not identify GPU-frequency causality,
prove candidate nonregression or establish an API/model speed win. The nominal
paired candidate/stock ratio at250 was1.049,95% interval[0.986,1.116], with
serial dependence and eight-block uncertainty explicitly unresolved.

Increase only the fixed warmup count to250. Keep50 timed dispatches, buffers,
initialization, kernel, timer,720us threshold and the M5-only calibration gate.
The old failure remains evidence; fresh all-feature validation is required for
the changed test. This change adds no production behavior or GPU payload buffers.
