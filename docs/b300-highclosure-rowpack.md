# B300 HIGH closure row packing research

Status: experimental follow-up to the v0.8/v0.9 compact-closure backends. This document describes the v0.10 HIGH-row-pack candidate. It is not a B300 timing result.

## Motivation

v0.8 assigns one warp to one selected HIGH closure row. The HIGH row predicate is compact, but the passive LOW side is still a fixed occupancy-mask group. Many such groups contain substantially fewer than 32 LOW columns, so a row-owned warp leaves many lanes unused.

At n=27 / W=28 / LOW=14 / HIGH=13, across all 13 HIGH positions in one DP row:

```text
selected closure state items : 1,503,950,445,478
v0.8 warp lane slots          : 3,021,117,696,896
v0.8 useful lane fraction     : 0.497813
```

The ~50% loss is not caused by invalid CROSS columns. `factor_highclosure_cross.cpp` independently classifies the actual HighDesc closure rows and checks the relevant LOW columns. For n=27 every LOW column used by a CROSS row has a valid boundary match and a valid target factor code.

## Packed mapping

v0.10 keeps the exact same HighDesc closure-row list and persistent metadata as v0.8/v0.9. Only the CUDA work assignment changes.

For each source FBlock and fixed LOW occupancy-mask group, define the compact row-major stream

```text
(selected HIGH closure row, LOW column)
```

and divide that stream into 32-item tasks. One warp owns one task. A warp may therefore span multiple selected HIGH rows when the fixed LOW-mask width is small.

Within a warp, lanes with the same selected row are grouped with `__match_any_sync`. The row-list entry and HighDesc word are loaded once by that row subgroup's leader and broadcast with `__shfl_sync`; they are not independently loaded by every lane.

The implementation is in:

- `src/cuda/b300/maskshard_highclosure_rowpack.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack_batch_guarded.cu`

The backend alias is:

```text
b300-factorized-maskshard-v0.10-highrowpack-fullclosure-batch
```

## n=27 structural model

`factor_highclosure_rowpack.cpp` derives the following W=28/LOW=14/256-thread counts and pins them as regression values:

```text
closure rows / HIGH position      :       715,533
useful closure state items        : 1,503,950,445,478
v0.8 row-owned warp lane slots    : 3,021,117,696,896
packed warp lane slots            : 1,503,993,904,960
packed / row lane ratio           : 0.497826982
packed useful lane fraction       : 0.999971104
```

Thus the packed mapping nearly removes warp-tail waste without changing the number of closure state values that must actually be read and atomically accumulated.

A task-sized host grid would have the following model:

```text
v0.8 row-sized launch blocks / DP row : 8,923,348,057
packed-task launch blocks / DP row    : 4,616,872,663
ratio                                  : 0.517392422

v0.8 blocks / 28-row residue          : 249,853,745,596
packed blocks / 28-row residue        : 129,272,434,564
```

This host-grid reduction is **not yet fully implemented** in the first v0.10 candidate. The current host still sizes the HIGH closure grid from v0.8's selected-row count. The row-pack kernel internally grid-strides over its packed task count, so correctness does not depend on task-sized host launch dimensions, but the potential launch-block reduction above remains future work.

## Memory model

v0.10 introduces no persistent row-pack table. It reuses:

- v0.8 HighDesc closure rows and FBlock offsets;
- v0.9 LOW closure columns;
- existing HighDesc/LowDesc and compact orbit aux tables.

Therefore the n=27 HBM model remains the v0.9 figure:

```text
peak model : ~249.173042 GiB/GPU
headroom   :  ~19.416958 GiB/GPU versus 268.59 GiB planning usable HBM
```

The row-pack kernel adds only a small shared-memory prefix (`65 * sizeof(Code)`) per CTA.

## CROSS analysis

`factor_highclosure_cross.cpp` gives the n=27 closure-row split:

```text
BLOCK rows : 7,315,213
CROSS rows : 1,986,716
```

CROSS state work is 218,576,117,914 items, about 14.5335% of HIGH closure state work. Used CROSS depths are exactly 1..12. The maximum target LOW mask-rank is 1000, so a direct uint16 target table is representable.

A full direct target table for all 12 used depths would cost about 27.510 MiB/GPU. A depth-1..6 hybrid table would cost about 13.755 MiB/GPU while covering about 99.9103% of CROSS state work. This could eliminate the boundary scan and dense LOW packed-rank lookup for most CROSS items, but it is deliberately not part of v0.10: row packing has the much larger structural opportunity and should be measured independently first.

## Validation layers

- `factor_highclosure_rowpack.cpp`: exact n=27 work/warp/task model with pinned counts.
- `factor_highclosure_rowpack_taskmap.cpp`: W=10/W=12 compares the old row-wise source mapping and the new packed source mapping and requires identical source sets with no duplicates.
- `factor_highclosure_cross.cpp`: exact BLOCK/CROSS/depth decomposition and LOW-target validity model.
- existing `maskshard_highclosure_rows_hostplan.cu`: validates the actual HighDesc closure-row list consumed by both v0.8 and v0.10.
- CUDA workflow: compiles v0.10 at W=22 and W=28 once GitHub runner execution is available.

GitHub Actions currently still fails before steps start (`steps=null`), so none of the new CUDA code has fresh nvcc evidence yet. Do not promote v0.10 until nvcc succeeds and B300 full-P2P runs show identical residues.

## Expected experiment

The useful comparison after CI/runtime availability is:

```text
v0.8  compact HIGH rows
v0.9  v0.8 + compact LOW columns
v0.10 v0.9 + packed HIGH row work assignment
```

Rank candidates by end-to-end `wall_s`. `high_closure_sum_s` is useful for attribution but is a sum of worker/GPU activity rather than phase wall-clock time.

A successful v0.10 result should first preserve every residue. Only then ask whether the reduced warp instruction footprint offsets the extra row-subgroup bookkeeping and any loss of memory coalescing when a warp spans non-adjacent selected HIGH rows.
