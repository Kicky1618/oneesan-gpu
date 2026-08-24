# B300 HIGH closure row-depth pruning

This note layers on v0.11 hybrid HIGH closure row packing and the v0.15 exact
frontier-height metadata.

## Structural observation

At DP row `r`, a MAIN source whose exact maximum frontier height exceeds `r` is
structurally zero. HIGH closure only reads that source count and adds it to a
BLOCKED destination; therefore such a closure body can be skipped before the
`mainv[i]` load without changing any reachable result.

The exact source depth is already available from v0.15:

```text
max(HIGH factor peak, LOW factor peak)
```

so v0.20 adds no persistent metadata.

## n=27 model

`factor_highclosure_rowdepth.cpp` enumerates the selected HIGH closure rows and
LOW occupancy classes, then caps both factor segments by row depth. For the
v0.11 threshold-16 hybrid:

```text
per-row dense useful closure states = 1,503,950,445,478
28-row dense useful states          = 42,110,612,473,384
28-row row-depth-active states      = 36,989,860,194,307
active ratio                        = 0.878397582502
heavy-body reduction                = 12.1602417498%
```

The active useful-state count by cap starts as:

```text
cap 1      234,881,024
cap 2   50,784,985,927
cap 3  369,363,350,086
cap 4  857,704,715,417
cap 5 1,231,387,068,804
cap 6 1,417,752,205,726
cap 7 1,483,630,545,931
cap 8 1,500,453,105,892
```

and approaches the dense 1,503,950,445,478 by cap 14.

The same probe also models a future exact compact closure launch. If active HIGH
rows and LOW columns are enumerated directly while preserving the v0.11
threshold-16 policy, the 28-row totals become:

```text
dense hybrid lane slots      = 50,814,816,443,776
exact row-depth lane slots   = 44,720,278,768,384
lane-slot reduction          = 11.9936233207%

dense launch blocks          = 117,915,540,260
exact row-depth blocks       = 104,256,200,361
launch-block reduction       = 11.5840031508%
```

Those compact figures are a future launch model, not the current v0.20 runtime.

## v0.20 implementation

v0.20 keeps v0.11's current row-pack task mapping and launch geometry. Inside
`maskshard_highclosure_rowpack_apply()` it checks the exact factor peaks before
reading `mainv[i]`; unreachable sources return immediately.

At full depth the predicate is bypassed entirely, so saturated rows do not pay
peak metadata reads. The A/B comparison therefore isolates the closure-body
predicate reasonably well:

```bash
python3 scripts/bench/b300_maskshard_highclosure_rowdepth_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

Primary attribution metric: `high_closure_sum_s`. Retain only if `wall_s` also
improves and residues match exactly.

## More important next launch fix

The current shared host still sizes HIGH closure launches from the v0.8 selected
row count even when the v0.11 kernel has converted those rows into fewer hybrid
warp tasks. The existing hybrid probe already pins the task-sized n=27 value:

```text
v0.8 row-sized launch blocks = 8,923,348,057 per DP row
v0.11 task-sized blocks      = 4,211,269,295 per DP row
```

This ~52.8% host-launch reduction is larger than the additional ~11.6% available
from exact row-depth closure compaction, and should be implemented before a more
complex v0.21 row-depth compact closure table.

Fresh nvcc and real B300x8 full-P2P validation remain required before promotion.
