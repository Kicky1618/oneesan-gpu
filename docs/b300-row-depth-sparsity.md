# B300 row-depth structural sparsity and v0.14

This study starts from v0.13, where row-boundary BLOCKED input is already known
zero, its P2P gather is removed, and its local scratch is initialized lazily by
the first HIGH orbit.

The remaining question is whether MAIN/BLOCKED states can also be declared zero
from row geometry without inspecting count values.

## Frontier depth

Interpret a canonical frontier state as the usual Motzkin-like walk read from
high position to low position, starting at height 1:

- `N`: height unchanged;
- `R`: height -1;
- `L`: height +1.

Let `depth(m)` be the maximum height visited by that walk.

`factor_row_depth_support.cpp` propagates boolean reachability through the exact
`include_horizontal()` / `blocked_exclude()` semantics. Through W=14 it verifies
that after complete row `r`, reachable MAIN states are exactly those with
`depth(m) <= r` until the full state space saturates. During the HIGH portion of
row `r`, reachable MAIN/BLOCKED states also never exceed depth `r`.

For W=14 the row-boundary support sequence is pinned as

```text
r=1  8,192
r=2  80,782
r=3  159,094
r=4  190,400
r=5  196,406
r=6  196,924
r=7  196,938  (all states)
```

## Exact max-depth opportunity at n=27

For W=28 MAIN, exact maximum-height capped counts are

```text
cap 1       134,217,728   0.0348%
cap 2    18,457,556,052   4.7852%
cap 3   112,925,875,764  29.2767%
cap 4   240,539,369,472  62.3612%
cap 5   329,056,985,516  85.3099%
cap 6   369,274,024,420  95.7364%
cap 7   382,187,801,740  99.0844%
cap 8   385,169,379,172  99.8574%
cap 14  385,719,506,620  100%
```

Starting from the v0.12/v0.13 dense HIGH-I/O model, an implementation that could
transfer exactly these height-capped regions would use

```text
dense words            25,380,726,522,116
exact depth-cap words   22,074,394,853,240
logical payload         92.334545196 -> 80.306180655 TiB/residue
7/8 peer approximation                  -> 70.267908074 TiB/residue
```

This remains an upper-bound target rather than the first runtime implementation.

## v0.14: coarse FBlock depth filtering

The existing factorized HIGH scratch is organized into FBlocks. All states in a
MAIN FBlock share:

- `he`: frontier height after the HIGH segment;
- `hs`: frontier height after the center symbol / before the LOW segment.

Every BLOCKED FBlock similarly shares its segment boundary height.

Therefore

```text
max(he, hs) > current row cap
```

is a sufficient condition for every MAIN state in that block to be structurally
zero. For BLOCKED, `he > cap` is sufficient. This test is deliberately weaker
than the exact maximum-height predicate because a path may rise above the cap
and return while keeping its segment endpoint heights small.

The advantage is that the kernel already has `FBlock x`; no code reconstruction,
no per-state table, and no additional persistent HBM metadata are required.

`factor_row_depth_fblock.cpp` computes the exact n=27 traffic model for this
coarse predicate. With gather cap `max(1,row-1)` and scatter cap `row`:

```text
v0.13 dense words       25,380,726,522,116
v0.14 FBlock-cap words  23,264,294,823,853
ratio                    0.916612643
reduction                 8.3387357%

logical payload          84.635011531 TiB/residue
7/8 peer approximation   74.055635090 TiB/residue
```

Relative to v0.13's ~80.792727 TiB peer model, v0.14 removes another ~6.737 TiB
per residue. It captures roughly 64% of the remaining peer-traffic reduction
available in the exact max-depth model, without adding metadata.

The gather kernel still walks the whole scratch range. States in excluded
FBlocks receive a local zero instead of a P2P load; scatter simply skips the
remote store. HIGH orbit/closure kernels are unchanged, so v0.13 -> v0.14 is
primarily an I/O experiment rather than a compute-work change.

## Semantic validation

`factor_row_depth_fblock_semantics.cpp` runs the complete numerical Grid-FP DP
with modulus 4294967291 twice:

1. canonical transitions;
2. a simulated v0.14 path that zeros excluded gather states and drops excluded
   scatter states at every HIGH window.

The local validation command

```bash
g++ -O3 -std=c++17 -Wall -Wextra -Werror \
  src/cpp/probes/factor_row_depth_fblock_semantics.cpp \
  -o build/factor_row_depth_fblock_semantics
build/factor_row_depth_fblock_semantics 12
```

passed for W=6,8,10,12. The traffic probe also passed locally and reproduced the
n=27 pinned values above.

## Runtime candidate

Files:

- `src/cuda/b300/maskshard_rowdepth_fblock_io.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch.cu`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_lazyblockinit_rowdepthfblock_batch_guarded.cu`
- `scripts/bench/b300_maskshard_rowdepth_ab.py`
- `.github/workflows/b300-rowdepth-v14.yml`

Backend alias:

```text
b300-factorized-maskshard-v0.14-highrowpack16-lazyblockinit-rowdepthfblock-batch
```

A/B command:

```bash
python3 scripts/bench/b300_maskshard_rowdepth_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 \
  --vram-reserve-mib 1024
```

Residues must match v0.13 exactly. The first metric is `high_io_sum_s`; total
`wall_s` decides whether the added branch/local-zero work is worthwhile.

## Next exact-depth candidate

If v0.14 wins on B300, the next step is an exact per-factor max-depth predicate.
A compact design needs about one byte per LOW/HIGH code: approximately
1,201,917 LOW entries + 787,333 HIGH entries = 1,989,250 bytes, or ~1.90 MiB/GPU.
That would approach the ~70.27 TiB peer upper-bound model while retaining the
same factorized storage layout. It should not be implemented before v0.14 shows
that the cheaper FBlock filter pays for its branch/local-zero overhead.
