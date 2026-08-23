# B300 HIGH closure hybrid row packing

This note records the v0.11 candidate built on top of the v0.9 compact HIGH/LOW closure backend.

## Why not pack every HIGH closure row?

v0.8 assigns one warp to one selected HIGH closure row. At n=27 this gives
`3,021,117,696,896` lane slots for `1,503,950,445,478` useful state items, so
only about 49.7813% of lanes are useful.

v0.10 fully flattens `(selected HIGH row, fixed-LOW-mask column)` within each
FBlock. Its analytical lane utilization is nearly perfect, but a warp can span
multiple HIGH rows. That increases row-list/HighDesc subgroup loads. It also
creates 65,535-block cap hits in large groups once the host grid is sized from
packed tasks.

The CROSS path is not the cause of the old 50% utilization. The exact n=27
CROSS probe finds that all LOW columns in actually used `(hs, depth)` cases have
valid targets.

## v0.11 rule

v0.11 packs only FBlocks whose fixed LOW-mask width satisfies

```text
stride < 16
```

Wider FBlocks keep the v0.8 one-row-per-warp mapping. There is no new persistent
metadata; the v0.8 HIGH closure row table and v0.9 LOW closure table are reused.
The modeled HBM peak therefore remains about 249.173042 GiB/GPU.

Implementation:

- `src/cuda/b300/maskshard_highclosure_rowpack.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_batch_guarded.cu`

Backend alias:

```text
b300-factorized-maskshard-v0.11-highrowpack16-fullclosure-batch
```

## n=27 / 256-thread structural model

The independent probe `factor_highclosure_rowpack_hybrid.cpp` gives the
threshold-16 regression values:

```text
useful states             1,503,950,445,478
v0.8 lane slots           3,021,117,696,896
v0.11 lane slots          1,814,814,872,992
lane ratio                0.6007097555
v0.11 useful-lane fraction 0.8287073618

v0.8 row/desc loads       71,386,429,790
v0.11 row/desc loads      79,173,964,114
desc-load ratio           1.1090898417

v0.8 logical blocks       8,923,348,057
v0.11 task-sized blocks   4,211,269,295
logical block ratio       0.4719382532
capped group-positions    0
```

The task-sized block count is a model of a future host launch grid. The current
shared batch host still sizes the HIGH closure launch from the v0.8 selected-row
count, so v0.11 currently over-launches and relies on the kernel grid-stride loop
to ignore excess warps. This is correct but does not yet realize the full block
count reduction above.

## Threshold tradeoff

Packing more widths reduces lane waste but increases row subgroup loads. Selected
n=27 points are:

| pack when stride < | lane ratio vs v0.8 | descriptor-load ratio | task-sized block ratio | capped groups |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.768351 | 1.016822 | 0.693646 | 0 |
| 8 | 0.716502 | 1.029546 | 0.625076 | 0 |
| 10 | 0.671537 | 1.046777 | 0.565609 | 0 |
| 16 | 0.600710 | 1.109090 | 0.471938 | 0 |
| 21 | 0.583801 | 1.138906 | 0.449578 | 0 |
| 29 | 0.573953 | 1.213178 | 0.436551 | 0 |
| 43 | 0.543380 | 1.280387 | 0.451501 | 0 |
| full | 0.497827 | 1.576692 | 0.517392 | 19,123 |

Threshold 16 is intentionally conservative. Threshold 29 gives a somewhat
smaller lane/block model but pays about 21.3% more descriptor loads. Both should
be treated as candidates until B300 timing resolves the tradeoff.

## Combined structural executor model

Using the same lane-slot accounting as the earlier v0.9 model:

```text
blocked orbit slots       3,645,418,645,989
v0.11 HIGH closure slots  1,814,814,872,992
v0.9 LOW closure slots    1,620,040,986,016
------------------------------------------------
v0.11 total               7,080,274,504,997
```

This is about 0.854427x the v0.9 structural slot count. It is not a wall-time
prediction: HIGH P2P traffic, atomics, cache behavior, descriptor work and the
current over-launched grid remain outside this simple count.

## Validation

`factor_highclosure_rowpack_hybrid_taskmap.cpp` compares the old row-wise source
set with the hybrid mapping at W=10 and W=12, rejecting duplicates or omissions.
The dedicated workflow `.github/workflows/b300-highrowpack-v11.yml` also asks nvcc
to compile both v0.10 and v0.11 at W=22 and W=28.

GitHub Actions is currently failing before job steps start (`steps=null`), so
these new CI jobs are not fresh compile evidence yet.

## First B300 comparison

Use the row-pack A/B entrypoint:

```bash
python3 scripts/bench/b300_maskshard_rowpack_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 \
  --vram-reserve-mib 1024
```

Its default comparison is v0.9 -> v0.10 -> v0.11. The base driver still enforces
exact GPU count, modulus identity, residue equality, build provenance and raw
phase logs. Compare `high_closure_sum_s` first, then total `wall_s`.
