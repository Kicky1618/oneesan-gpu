# B300 HIGH closure row-depth research

This note layers on v0.11 hybrid HIGH closure row packing and the v0.15 exact
frontier-height metadata.

## Structural observation

At DP row `r`, a MAIN source whose exact maximum frontier height exceeds `r` is
structurally zero. HIGH closure only reads that source count and adds it to a
BLOCKED destination, so such a source can be omitted before the `mainv[i]` load.
The exact source depth is already available as

```text
max(HIGH factor peak, LOW factor peak).
```

## v0.20: structural-zero body pruning

v0.20 leaves v0.11 task mapping and launch geometry unchanged. Inside the row-pack
source body it checks the exact factor peaks before reading `mainv[i]`. Full-depth
rows bypass the check.

For n=27 / W=28:

```text
per-row dense useful closure states = 1,503,950,445,478
28-row dense useful states          = 42,110,612,473,384
28-row active useful states         = 36,989,860,194,307
active ratio                        = 0.878397582502
heavy-body reduction                = 12.1602417498%
```

No additional persistent metadata is needed beyond v0.15.

## v0.21: task-sized host launch

The v0.11 kernel already transforms selected closure rows into fewer warp tasks,
but the shared host historically sized the CUDA grid from the older v0.8 row
count. v0.21 gates a host-only fix under `MASKSHARD_HIGH_CLOSURE_TASK_LAUNCH`
and computes exactly the same fixed threshold-16 task count as the kernel.

Pinned n=27 / 256-thread model per DP row:

```text
v0.8 row-sized launch blocks = 8,923,348,057
v0.11 task-sized blocks      = 4,211,269,295
```

This is a launch-count model, not a wall-time claim.

## v0.22: exact row-depth task mapping

Inside one MAIN FBlock the exact active closure source set factorizes as

```text
{selected HIGH closure rows with HIGH peak <= cap}
    x
{LOW ranks with LOW peak <= cap}.
```

v0.22 reuses v0.19's peak-sorted LOW-rank permutation. It adds a peak-sorted
copy of the selected HIGH closure rows and a tiny cumulative HIGH active-count
table. The packing policy itself remains exactly v0.11's dense policy:
`x.stride < 16` is packed; a smaller row-dependent active LOW count never changes
that decision.

Persistent extra GPU metadata is:

```text
selected HIGH rows       9,301,929 * uint32 = 35.484043 MiB/GPU
HIGH active-count table  13*65*15 * uint32  =  0.048351 MiB/GPU
extra v0.22 metadata                           35.532394 MiB/GPU
```

With the existing v0.19-era memory model this gives approximately
`249.215 GiB/GPU`, leaving about `19.375 GiB/GPU` under the 268.59 GiB planning
budget.

Early rows enumerate only the active Cartesian product. At full row depth all
states are active, so the kernel switches back to the original closure-row order
and physical LOW-rank order. Thus saturated rows do not pay compact-rank
indirection or lose the v0.21 locality.

The corrected fixed-policy n=27 model over all 28 rows is:

```text
dense hybrid lane slots = 50,814,816,443,776
exact compact lane slots = 44,779,140,427,808
reduction                = 11.8777876973%
```

v0.22 deliberately keeps the v0.21 dense task-sized host grid.

## v0.23: exact row-depth host launch

v0.23 changes no CUDA task mapping. During setup it builds a host-only table of
exact v0.22 warp-task counts indexed by fixed LOW occupancy mask, HIGH position
and row-depth cap. The shared host uses that count to size the closure grid.

For W=28/LOW14/HIGH13 there are

```text
16,384 masks * 13 HIGH positions * 15 caps = 3,194,880 uint32 entries
host-only table = 12.1875 MiB
extra GPU HBM   = 0
```

The fixed-policy launch model becomes:

```text
v0.21/v0.22 dense task-sized blocks = 117,915,540,260 / residue
v0.23 exact row-depth blocks        = 104,486,127,592 / residue
reduction                           = 11.3890099968%
```

At full cap the host table must equal the v0.21 dense task count for every
`mask, HIGH-position` pair. The runtime asserts this before launching and exits
if the two independent constructions disagree.

The table build is setup-only and is included in `setup_s`, not the timed DP
`wall_s`. Its first implementation intentionally favors a transparent factorized
construction over micro-optimizing setup time; B300 profiling should determine
whether further setup reduction is useful.

## v0.25: threshold-29 HIGH row packing

v0.24 is the separate LOW-closure row-depth experiment. The next HIGH-closure
candidate is therefore numbered v0.25.

The v0.11 threshold of 16 was intentionally conservative. After v0.22 exact
row-depth task compaction and v0.23 exact launch sizing, the tradeoff changes:
we can evaluate the threshold against the actual active LOW prefix on every DP
row while still making the packing decision from the dense physical stride.

`factor_highclosure_rowdepth_threshold.cpp` reproduces the v0.23 threshold-16
aggregate exactly and evaluates the same task map for alternate compile-time
thresholds. For n=27 / W=28 / 256 threads:

```text
                         v0.23 threshold 16     v0.25 threshold 29
active useful items       36,989,860,194,307     36,989,860,194,307
lane slots                44,779,140,427,808     42,734,081,059,456
row/descriptor subgroups   1,948,871,708,005      2,131,117,327,561
warp tasks                   835,870,866,393        771,962,761,132
launch blocks                104,486,127,592         96,497,498,944
capped group-positions                    0                      0
```

Relative to v0.23 this is:

```text
lane slots       -4.566991123%
warp tasks       -7.645691198%
launch blocks    -7.645635676%
descriptor loads +9.351339999%
```

An exhaustive integer threshold scan `1..1002` in the same analytical model
places both the minimum warp-task count and the minimum launch-block count at
threshold 29. Threshold 36 reduces lane slots slightly further, but already
increases task/block count and descriptor work, so threshold 29 is the clean
A/B candidate rather than an assumption that more packing is always better.

v0.25 adds no persistent GPU metadata and changes no source set. It only changes
`MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD` from 16 to 29; both the compact CUDA
task map and the v0.23 host launch table consume that same compile-time policy.

## Semantic validation

`factor_highclosure_rowdepth_taskmap.cpp` exhaustively compares, for every HIGH
position, LOW occupancy mask and row cap, the mathematical exact source set with
the compact task mapping. It also checks duplicate-free mapping and fixes the
packing decision from dense LOW width.

Local validation passed with
`g++ -O3 -std=c++17 -Wall -Wextra -Werror`:

```text
W=10 LOW=5 expected=12,719  compact=12,719  warp_tasks=2,306
W=12 LOW=6 expected=146,624 compact=146,624 warp_tasks=9,311
```

The v0.25 threshold model also passes the same warning-clean local build and
contains pinned threshold-16 and threshold-29 n=27 regressions.

## A/B commands

v0.19 -> v0.20 isolates the source-body predicate:

```bash
python3 scripts/bench/b300_maskshard_highclosure_rowdepth_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.20 -> v0.21 isolates task-sized host launch geometry:

```bash
python3 scripts/bench/b300_maskshard_highclosure_tasklaunch_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.21 -> v0.22 isolates exact closure task mapping:

```bash
python3 scripts/bench/b300_maskshard_highclosure_exacttasks_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.22 -> v0.23 isolates exact host launch sizing:

```bash
python3 scripts/bench/b300_maskshard_highclosure_exactlaunch_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.23 -> v0.25 isolates the threshold-16 -> threshold-29 packing policy:

```bash
python3 scripts/bench/b300_maskshard_highclosure_threshold29_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

`high_closure_sum_s` is the attribution metric; residues must match exactly and
`wall_s` decides retention.

Fresh nvcc and real B300x8 full-P2P validation remain required before promotion.
