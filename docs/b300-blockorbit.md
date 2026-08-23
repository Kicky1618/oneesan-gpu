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
```

## v0.6 blocked-domain executor

Existing descriptor tables already contain the inverse map:

- `HighDesc.block_desc`: blocked HIGH coordinate -> representative main coordinate;
- `LowDesc.block_desc`: blocked LOW coordinate -> representative main coordinate.

The orbit update algebra is unchanged:

- NN: companion gets `+c`; representative becomes `c + old_block`; blocked becomes 0;
- NR/NL, p>1: representative becomes `c + companion + old_block`; blocked becomes `c`;
- NR/NL, p=1: representative becomes `c + companion + old_block`; companion becomes `c + companion`; blocked becomes 0.

A pure-C++ exhaustive seeded-vector probe compares blocked-domain orbit + closure against the ordinary out-of-place Grid-FP update for every p at W=10 and W=12.

For source scanning per p:

```
v0.5: orbit M + closure M = 2M = 771,439,013,240
v0.6: orbit D + closure M = M+D = 520,735,012,027
ratio v0.6/v0.5            = 0.6750177306
```

## v0.7 compact blocked-coordinate aux

Once iteration is over blocked states, a main-coordinate aux table is wasteful. `block_desc` already supplies the representative main coordinate. The only extra information needed is NN-vs-pair and, for NR/NL with p>1, the companion main block/rank. LOW p=1 needs only the PAIR kind because `LowDesc.main_desc` is already the companion target.

v0.7 stores one 32-bit word in the same coordinate system as the descriptor block table:

```
HIGH aux index = [p][HighDesc blocked active coordinate]
LOW  aux index = [p][LowDesc  blocked active coordinate]
```

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

Validation layers:

1. `factor_blockorbit_semantics.cpp`: blocked-domain orbit + closure equals canonical update at W=10/W=12.
2. `factor_blockorbit_compactaux_semantics.cpp`: one compact aux word per blocked state is sufficient, including LOW p=1.
3. `maskshard_blockorbitaux_hostplan.cu`: builds the actual compact CUDA host tables and checks block descriptor -> representative -> compact aux against exact MateID transitions.

## v0.8 compact HIGH closure rows

For the HIGH window, transition kind depends only on `(HIGH exact row, center, p)`. The passive LOW mask-rank does not affect whether the source is `LL`, `RR`, or `RL`. Therefore the old closure kernel repeats the same HIGH-code lookup, row split, pair classification, and HighDesc lookup for every LOW column.

v0.8 extends HighDesc generation with a compact list of source HIGH rows that are both:

- `LL`, `RR`, or `RL`; and
- represented by a `HIGHDESC_BLOCK` or `HIGHDESC_CROSS` descriptor.

At n=27 there are exactly 715,533 such source HIGH rows for every HIGH position. The earlier 715,534 count included the impossible `hs=LOW+1` FBlock; a LOW segment of length 14 cannot descend from height 15 to zero, so that block has zero columns and is now excluded consistently with the storage layout. Across 13 positions:

```
closure row entries     : 9,301,929
closure row table       : 35.484043 MiB/GPU
FBlock range table      : 3.300781 KiB/GPU
v0.8 extra vs v0.7      : 35.487267 MiB/GPU
v0.8 peak               : 249.116280 GiB/GPU
v0.8 headroom           :  19.473720 GiB/GPU
```

The HBM figure is tracked by `factor_maskshard_memory.cpp`; `factor_highclosure_rows.cpp` separately derives the 715,533-row combinatorial count, state work, warp work, and launch-grid model.

The list is stored in source-FBlock order. A tiny `[p][FBlock]` offset table identifies the valid row range for each block. For each fixed LOW occupancy mask, the host precomputes how many listed rows belong to FBlocks whose group-local `stride` is nonzero and stores that count in the HIGH job. The closure grid is launched from the compact row count rather than from `main_n`.

The v0.8 closure kernel assigns one valid HIGH closure row to one warp. The warp processes passive LOW columns coalesced as `lr = lane, lane+32, ...`. Row membership is precomputed once on the host; the kernel no longer performs pair classification for closure.

n=27 work model for the 13 HIGH positions, per DP row:

```
old full-state HIGH closure state threads : 5,014,353,586,060
exact useful closure state updates         : 1,503,950,445,478
valid closure-row warp assignments         :    71,386,429,790
warp rounds incl. LOW widths > 32          :    94,409,928,028
warp lane slots                            : 3,021,117,696,896
```

The warp-lane model is 60.2494% of the old flattened state-thread count, with 49.7813% of those lane slots carrying a useful closure state. This is a structural work model, not a B300 timing prediction.

The host grid-sizing improvement is much smaller because many large LOW-mask groups hit the 65,535-block cap. At 256 threads:

```
old main_n-sized closure blocks / DP row : 9,140,772,719
v0.8 row-sized closure blocks / DP row   : 8,923,348,057
ratio                                     : 0.976213755
```

Thus v0.8 is not justified as a large kernel-launch-count reduction. Its main expected benefit is removing repeated per-state HIGH-row decoding/classification while retaining coalesced LOW-column access.

`maskshard_highclosure_rows_hostplan.cu` validates the actual HighDesc-generated row list, exact membership, source-FBlock ranges, and descriptor kinds once CUDA CI can execute.

## v0.9 compact LOW closure columns

A naive LOW-column executor would assign a warp to one selected LOW column and walk HIGH rows vertically. That gives strided state accesses and is deliberately not used.

Instead v0.9 stores the valid LOW closure columns in storage all-rank order for each `(p,FBlock)`. A runtime task is:

```
(FBlock, one HIGH row, one 32-column compact chunk)
```

One warp therefore stays inside one HIGH row and processes up to 32 selected LOW columns. The selected column indices are monotonically increasing within each FBlock, so source reads retain useful spatial locality even though the columns are compacted.

The n=27 LOW-column model gives 1,088,282 compact source columns per LOW position. Across 14 positions:

```
LOW closure column table : 58.120529 MiB/GPU
FBlock range table       :  3.554688 KiB/GPU
v0.9 extra vs v0.8       : 58.124001 MiB/GPU
v0.9 peak model          : 249.173042 GiB/GPU
v0.9 headroom            :  19.416958 GiB/GPU
```

`factor_v09_memory_delta.cpp` independently derives the 1,088,282-column count and adds the exact v0.9 table delta to the v0.8 peak. This is intentionally a separate probe so the already-established v0.8 HBM model is not rewritten while v0.9 remains experimental.

The locality probe `factor_lowclosure_columns.cpp` reports that the selected source words are about 29.99% of the full LOW closure scan. In a simple source-read locality model, selected columns touch about 41.70% as many 32-byte sectors and about 47.53% as many 128-byte lines as the full storage-order scan. These figures model source reads only; destination atomics, descriptor accesses, cache reuse, and scheduling still require B300 measurement.

`factor_lowclosure_taskmap.cpp` independently checks the arithmetic decomposition into `(row, compact-column chunk)` at small widths. `maskshard_lowclosure_cols_hostplan.cu` checks the actual StorageFactorHost + LowDesc-generated compact table, including descriptor validity and FBlock ranges, when nvcc execution is available.

Both v0.8 and v0.9 require a complete number of warps per block. Their wrappers now reject thread counts that are not multiples of 32 (and values outside 32..1024); the default 256-thread launch is unchanged. This prevents a partial final warp from silently dropping columns.

The fixed-p main-state partition remains:

1. orbit representatives `NN/NR/NL`: D states;
2. companions `LR/RN/LN`: D states;
3. closure sources `LL/RR/RL`: `M - 2D = 115,688,495,806` states.

With both orbit and closure compacted, the ideal source-state work per p is

```
D + (M - 2D) = M - D = 250,704,001,213.
```

The implementation is not assumed to achieve the corresponding wall-time ratio because compact index loads and atomic destinations remain.

## Runtime candidates

- v0.4 baseline: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu`
- v0.5 main-domain aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_aux_batch_guarded.cu`
- v0.6 blocked-domain/full-aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_aux_batch_guarded.cu`
- v0.7 blocked-domain/compact-aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_batch_guarded.cu`
- v0.8 v0.7 + compact HIGH closure rows: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_highclosurerows_batch_guarded.cu`
- v0.9 v0.8 + compact LOW closure columns: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosurerows_batch_guarded.cu`

GitHub Actions is still failing before job steps start (`steps=null`), including ordinary CPU-only probes and the CUDA W=22/W=28 matrix. Therefore there is still no fresh nvcc evidence for v0.2-v0.9. Do not merge any candidate to the production path before fresh nvcc CI and real full-P2P multi-GPU residue checks. The first useful B300 comparison is v0.4 vs v0.7 vs v0.8 vs v0.9 on identical moduli; v0.5/v0.6 remain diagnostic controls.
