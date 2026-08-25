# B300 blocked-domain orbit research

Status: experimental follow-up to `docs/b300-mask-shard.md`. v0.4 remains the correctness baseline; v0.5 adds main-coordinate orbit aux; v0.6 moves orbit iteration to blocked states; v0.7 compacts aux into blocked coordinates; v0.8 compacts the HIGH closure pass by HIGH row; v0.9 additionally compacts the LOW closure pass by LOW column while preserving row-local memory access.

## Delete-N bijection

Fix one horizontal position `p`. A main orbit representative is exactly a main state whose upper active symbol at `p` is `N`, i.e. one of `NN`, `NR`, `NL`.

Deleting that `N` produces a valid blocked state. Conversely `blocked_exclude(blocked, p)` inserts `N` at the same position and produces exactly one representative main state. These maps are inverse, so for every `p`:

```
# orbit representatives = # blocked states.
```

At n=27 / W=28:

```
M = main states    = 385,719,506,620
D = blocked states = 135,015,505,407
D / M              = 0.3500354612348228
C = M - 2D         = 115,688,495,806 closure-pair states
```

## v0.6 blocked-domain executor

Existing descriptor tables already contain the inverse map:

- `HighDesc.block_desc`: blocked HIGH coordinate -> representative main coordinate;
- `LowDesc.block_desc`: blocked LOW coordinate -> representative main coordinate.

The orbit update algebra is unchanged. A pure-C++ exhaustive seeded-vector probe compares blocked-domain orbit + closure against the ordinary out-of-place Grid-FP update for every p at W=10 and W=12.

For source scanning per p:

```
v0.5: orbit M + closure M = 2M = 771,439,013,240
v0.6: orbit D + closure M = M+D = 520,735,012,027
ratio v0.6/v0.5            = 0.6750177306
```

## v0.7 compact blocked-coordinate aux

Once iteration is over blocked states, a main-coordinate aux table is wasteful. `block_desc` already supplies the representative main coordinate. v0.7 stores the remaining orbit information in the same blocked-coordinate space.

Exact n=27 memory:

```
v0.5/v0.6 HIGH main aux : 113.573406 MiB/GPU
v0.5/v0.6 LOW  main aux : 186.499115 MiB/GPU
v0.5/v0.6 total         : 300.072521 MiB/GPU

v0.7 HIGH block aux     :  39.044682 MiB/GPU
v0.7 LOW  block aux     :  64.189293 MiB/GPU
v0.7 total              : 103.233974 MiB/GPU
```

HBM model:

```
v0.4 peak                : 248.980810 GiB/GPU
v0.5/v0.6 peak           : 249.273850 GiB/GPU
v0.7 compact-aux peak    : 249.081625 GiB/GPU
planning usable HBM      : 268.590000 GiB/GPU
v0.7 headroom            :  19.508375 GiB/GPU
```

Validation layers include `factor_blockorbit_semantics.cpp`, `factor_blockorbit_compactaux_semantics.cpp`, and the actual CUDA host-plan probe `maskshard_blockorbitaux_hostplan.cu`.

## v0.8 compact HIGH closure rows

For the HIGH window, transition kind depends only on `(HIGH exact row, center, p)`. The passive LOW mask-rank does not affect whether the source is `LL`, `RR`, or `RL`. v0.8 therefore materializes only source HIGH rows whose descriptor is a real closure destination (`BLOCK` or `CROSS`).

At n=27 there are exactly 715,533 such source HIGH rows for every HIGH position. The earlier 715,534 count included the impossible `hs=LOW+1` FBlock; a LOW segment of length 14 cannot descend from height 15 to zero. Across 13 positions:

```
closure row entries     : 9,301,929
closure row table       : 35.484043 MiB/GPU
FBlock range table      : 3.300781 KiB/GPU
v0.8 extra vs v0.7      : 35.487267 MiB/GPU
v0.8 peak               : 249.116280 GiB/GPU
v0.8 headroom           :  19.473720 GiB/GPU
```

The list is stored in source-FBlock order. For each fixed LOW occupancy mask the host precomputes the number of listed rows belonging to active FBlocks. One warp handles one compact HIGH closure row and processes passive LOW columns as `lr = lane, lane+32, ...`.

n=27 work model for all 13 HIGH positions in one DP row:

```
old full-state HIGH closure state threads : 5,014,353,586,060
exact useful closure state updates         : 1,503,950,445,478
valid closure-row warp assignments         :    71,386,429,790
warp rounds incl. LOW widths > 32          :    94,409,928,028
warp lane slots                            : 3,021,117,696,896
```

The warp-lane model is 60.2494% of the old flattened state-thread count, with 49.7813% useful lanes. The launch-grid reduction is much smaller because many large groups hit the 65,535-block cap:

```
old main_n-sized closure blocks / DP row : 9,140,772,719
v0.8 row-sized closure blocks / DP row   : 8,923,348,057
ratio                                     : 0.976213755
```

Thus v0.8 is mainly intended to remove repeated per-state HIGH-row decoding/classification while retaining coalesced LOW-column access, not to claim a 40% wall-time speedup.

## v0.9 compact LOW closure columns

A naive LOW-column executor would assign a warp to one selected LOW column and walk HIGH rows vertically. That gives strided state accesses and is deliberately not used.

Instead v0.9 stores valid LOW closure columns in storage all-rank order for each `(p,FBlock)`. A runtime task is:

```
(FBlock, one HIGH row, one 32-column compact chunk)
```

One warp stays inside one HIGH row and processes up to 32 selected LOW columns. The selected indices are monotonically increasing within each FBlock, retaining useful source locality.

### Exact table size

For n=27, all closure-pair columns (`LL/RR/RL`) are valid under the same representative-HIGH construction used by `LowDesc`: the CPU probe `factor_lowclosure_valid_columns.cpp` checks `include_horizontal(...).valid` for every factor column and every LOW position, and finds zero invalid columns. The exact count is therefore 1,088,282 columns for every one of the 14 LOW positions, not merely an upper bound.

```
LOW closure column table : 58.120529 MiB/GPU
FBlock range table       :  3.554688 KiB/GPU
v0.9 extra vs v0.8       : 58.124001 MiB/GPU
v0.9 peak                : 249.173042 GiB/GPU
v0.9 headroom            :  19.416958 GiB/GPU
```

`factor_v09_memory_delta.cpp` independently derives the same 1,088,282-column count and applies the exact v0.9 table delta to the established v0.8 peak.

### Source locality

`factor_lowclosure_columns.cpp` models the selected source accesses in storage order. Across all 14 LOW positions in one DP row:

- selected source states are 29.9929078% of the old full-state LOW closure scan;
- selected columns touch about 41.70% as many 32-byte source sectors;
- selected columns touch about 47.53% as many 128-byte source lines.

These figures model source reads only; destination atomics, descriptor accesses, cache behavior and scheduling still require B300 measurement.

### Warp and launch model

`factor_lowclosure_launch.cpp` reproduces the real host task formula for every HIGH occupancy mask and applies the 65,535-block launch cap. At 256 threads, across all 14 LOW positions in one DP row:

```
old full-state source threads  : 5,400,073,092,680
selected closure states        : 1,619,638,941,284
selected/state ratio           : 0.2999290775303543

compact warp lane slots        : 1,620,040,986,016
compact lane / old-state ratio : 0.3000035292507477
useful compact lane fraction   : 0.9997518305181965

old closure launch blocks      : 6,328,761,404
v0.9 compact launch blocks     : 3,964,060,594
launch-block ratio             : 0.6263564607593792
launch-block reduction         : 37.3643539%

old capped group-positions     : 81,368
compact capped group-positions : 33,320
```

Across all 28 DP rows in one residue this corresponds to:

```
old LOW closure blocks / residue : 177,205,319,312
v0.9 blocks / residue             : 110,993,696,632
```

This is structurally stronger than the v0.8 HIGH-row compaction: LOW compact chunks are almost perfectly lane-dense, and the launch cap is hit by far fewer group-position pairs. It is still not a timing prediction; the compact-index gather and atomic destinations must be measured on B300.

`factor_lowclosure_taskmap.cpp` independently checks `(row, compact-column chunk)` arithmetic at small widths. `maskshard_lowclosure_cols_hostplan.cu` checks the actual StorageFactorHost + LowDesc-generated table, including descriptor validity and FBlock ranges, once nvcc execution is available.

Both v0.8 and v0.9 require complete warps. Their wrappers reject thread counts outside 32..1024 or not divisible by 32; the default 256-thread launch is unchanged. This prevents a partial final warp from silently dropping compact rows/columns.

With both orbit and closure compacted, the ideal source-state work per p becomes

```
D + C = M - D = 250,704,001,213.
```

The implementation is not assumed to achieve the corresponding wall-time ratio because compact index loads and atomic destinations remain.

## Runtime candidates

- v0.4 baseline: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu`
- v0.5 main-domain aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_aux_batch_guarded.cu`
- v0.6 blocked-domain/full-aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_aux_batch_guarded.cu`
- v0.7 blocked-domain/compact-aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_batch_guarded.cu`
- v0.8 v0.7 + compact HIGH closure rows: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_highclosurerows_batch_guarded.cu`
- v0.9 v0.8 + compact LOW closure columns: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosurerows_batch_guarded.cu`

## Validation status

The following v0.9 CPU probes are now present:

- `factor_lowclosure_columns.cpp`: storage-locality model;
- `factor_lowclosure_valid_columns.cpp`: exact closure-pair validity and 1,088,282-column proof;
- `factor_lowclosure_taskmap.cpp`: independent compact task mapping at W=10/W=12;
- `factor_lowclosure_launch.cpp`: exact n=27/256 task/grid model with pinned regression counts;
- `factor_v09_memory_delta.cpp`: exact HBM delta.

The validity, memory-delta and launch-model calculations have also been compiled locally with `g++ -O3 -std=c++17 -Wall -Wextra -Werror`; n=27 values match the pinned counts above.

GitHub Actions is still failing before job steps start (`steps=null`), including CPU-only probes and the CUDA W=22/W=28 matrix. Therefore there is still no fresh nvcc evidence for v0.2-v0.9. Do not merge any candidate to the production path before fresh nvcc CI and real full-P2P multi-GPU residue checks. The first useful B300 comparison is v0.4 vs v0.7 vs v0.8 vs v0.9 on identical moduli; v0.5/v0.6 remain diagnostic controls.
