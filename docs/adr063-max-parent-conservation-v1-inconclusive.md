# Private maximum-parent conservation V1: inconclusive before advice

Source `8cf2d7c582d5cd44dd530d9bb2f1c40f3e36ad07` was executed once on
Apple Silicon with the published 0.15.14 API substrate and a separately reviewed
parent runner. This isolated experiment is not a public API or model-admission
change. The original finite attempt is closed; its limits and outcome remain
unchanged.

The run stopped at the first NoAdvice PRE plateau because its frozen maximum
pairwise full-category L1 limit of 128 MiB failed. It retained two full
922,746,880-byte GPU readbacks, 15 complete dual-alias sweeps (420 page-query
calls), and nine raw physical-partition frames. No ACK, reclaim advice, churn or
conservation comparison occurred. Source page observations changed from fully
WIRED in the first sweep to unwired but PRESENT in the remaining fourteen.
That is useful partial observation, not a completed reclaim test.

The largest frame-pair L1 difference was 481,656,832 bytes (frames 7 and 8).
Anonymous increased 228.75 MiB, Wired decreased 229.671875 MiB, Free increased
0.8125 MiB and CachedFiles increased 0.109375 MiB. Compressor, Static and
Unaccounted were unchanged. Most observed movement was a classification change
between two fully charged categories, rather than a release to Free. These
numbers do not establish its cause or qualify any physical credit.

The parent terminated the child, reaped it and proved its process group empty;
the terminal receipt completed at 43.193402 seconds. The original case and
protocol remain failed/incomplete with physical credit zero. Root subsequently
removed the three generated fixture files (2,768,240,640 bytes) and directory
only after verifying the emitted directory and file device/inode identities,
regular-file types, lengths and empty process group. This separate cleanup does
not change the failed receipt or refund its historical I/O charge.

Independent audit reconstructed the nine raw partition frames and the complete
retained source prefix. No completed 15-readback, six-cut, 108-frame protocol,
maximum-parent invalidation integrity, driver conservation, production paging,
speed result or whole-Q5 acceptance follows. Any successor must predeclare a
new source and measurement hypothesis; this run must not be rescored or rerun
with a relaxed threshold.
