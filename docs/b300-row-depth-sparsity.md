# B300 row-depth structural sparsity: v0.14 and v0.15

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

An ideal implementation special-casing the first row's single initialized state
would use

```text
dense words            25,380,726,522,116
ideal depth-cap words   22,074,394,853,240
logical payload         92.334545196 -> 80.306180655 TiB/residue
7/8 peer approximation                  -> 70.267908074 TiB/residue
```

The runtime candidates deliberately use depth cap 1 for the first gather instead
of a one-state special case. The difference is only 134,217,727 words, about
0.000488 TiB logical traffic.

## v0.14: coarse FBlock depth filtering

The existing factorized HIGH scratch is organized into FBlocks. All states in a
MAIN FBlock share `he`, the frontier height after the HIGH segment, and `hs`,
the height after the center symbol / before the LOW segment. BLOCKED FBlocks
similarly share their segment boundary height.

Thus `max(he,hs) > current row cap` is a sufficient condition for every MAIN
state in that FBlock to be zero. It is not exact: a path may rise above the cap
and return while keeping both endpoint heights small.

The benefit is zero extra metadata. `factor_row_depth_fblock.cpp` pins n=27:

```text
v0.13 dense words       25,380,726,522,116
v0.14 FBlock-cap words  23,264,294,823,853
ratio                    0.916612643
reduction                 8.3387357%
logical payload          84.635011531 TiB/residue
7/8 peer approximation   74.055635090 TiB/residue
```

The gather kernel still scans the whole scratch range: excluded states receive a
local zero instead of a P2P load. Scatter skips excluded remote stores. HIGH
orbit/closure computation is unchanged.

Local `g++ -O3 -std=c++17 -Wall -Wextra -Werror` validation passed both the n=27
traffic probe and a complete numerical canonical-vs-filtered DP comparison for
W=6,8,10,12.

## v0.15: exact per-factor maximum-height filtering

The full-state maximum height factorizes exactly. For MAIN:

1. scan the HIGH code from height 1 and record its maximum height and ending
   height `he`;
2. apply the center symbol to obtain `hs`;
3. scan the LOW code starting at `hs` and record its maximum;
4. the full state's `depth(m)` is the maximum of the HIGH/center and LOW peaks.

BLOCKED has the same decomposition without the center symbol.

A LOW code can close to height zero from only one starting height, because its
net `L-R` displacement fixes that height uniquely. Therefore the existing
factor-code arrays need only one byte of peak metadata per stored code:

```text
LOW peak entries   1,201,917 bytes
HIGH peak entries    787,333 bytes
total              1,989,250 bytes = 1.897096634 MiB/GPU
```

`factor_row_depth_factorpeak.cpp` aggregates these peaks by segment ending
height and proves every cap count equals the ordinary height-capped full-state
DP. The runtime's conservative first gather gives

```text
exact-cap words          22,074,529,070,967
logical payload              80.306668937 TiB/residue
7/8 peer approximation       70.268335320 TiB/residue
```

Compared with v0.14 this removes another ~3.787300 TiB/residue of modeled peer
payload. The result is within ~0.000427 TiB peer traffic of the special-cased
one-state ideal.

`factor_row_depth_factorpeak_semantics.cpp` independently enumerates every MAIN
and BLOCKED state for W=6,8,10,12 and requires the split HIGH/LOW factor peak to
be exactly equal to the full frontier maximum height. Local `-Werror` builds and
runs passed:

```text
factor-row-depth-factorpeak W=28 low=14 high=13
low_peak_entries=1201917 high_peak_entries=787333 metadata_mib=1.897096634
main=385719506620 blocked=135015505407 exact_cap_words=22074529070967
logical_tib=80.306668937 balanced_7of8_peer_tib=70.268335320

row-depth-factorpeak-semantics OK W=6
row-depth-factorpeak-semantics OK W=8
row-depth-factorpeak-semantics OK W=10
row-depth-factorpeak-semantics OK W=12
```

The CUDA implementation aligns HIGH peak bytes with `D_F_HIGH_ALL_CODES` and
LOW peak bytes with `D_F_LOW_MASK_CODES`, so `hr/lr` from the existing FBlock
rank split index the tables directly. No state unranking is added.

Peak metadata is built and uploaded during the existing
`report_high_mask_shard_layout()` setup stage, before `setup_s` is finalized, so
one-time table construction does not contaminate `wall_s`. Once the row cap has
reached the maximum possible frontier depth, v0.15 bypasses both peak-table
loads and the depth test and falls through to normal HIGH I/O.

The peak tables raise modeled HBM from ~249.173042 to ~249.174895 GiB/GPU,
leaving ~19.4151 GiB/GPU against the 268.59 GiB planning limit.

## Runtime candidates and A/B

v0.14 files:

- `src/cuda/b300/maskshard_rowdepth_fblock_io.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_lazyblockinit_rowdepthfblock_batch_guarded.cu`
- `scripts/bench/b300_maskshard_rowdepth_ab.py`

v0.15 files:

- `src/cuda/b300/maskshard_rowdepth_exact_io.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_lazyblockinit_rowdepthexact_batch_guarded.cu`
- `src/cpp/probes/factor_row_depth_factorpeak.cpp`
- `src/cpp/probes/factor_row_depth_factorpeak_semantics.cpp`
- `scripts/bench/b300_maskshard_rowdepth_exact_ab.py`
- `.github/workflows/b300-rowdepth-exact-v15.yml`

Backend aliases:

```text
v0.14 b300-factorized-maskshard-v0.14-highrowpack16-lazyblockinit-rowdepthfblock-batch
v0.15 b300-factorized-maskshard-v0.15-highrowpack16-lazyblockinit-rowdepthexact-batch
```

Compare v0.13 -> v0.14:

```bash
python3 scripts/bench/b300_maskshard_rowdepth_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

Compare v0.14 -> v0.15:

```bash
python3 scripts/bench/b300_maskshard_rowdepth_exact_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

Residues must match exactly. `high_io_sum_s` is the primary attribution metric;
`wall_s` decides whether the extra two byte loads and branch in early rows are
worth the additional network reduction.

GitHub Actions remains blocked by the runner/startup problem (`steps=null`), so
fresh nvcc and full-P2P B300 correctness/timing remain mandatory before either
row-depth backend can be promoted.
