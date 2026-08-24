# B300 HIGH orbit launch and row-depth pruning

This note follows the v0.15 exact row-depth HIGH-I/O candidate.

## v0.16: size the launch from the domain the kernel actually scans

Since v0.6, the HIGH ordinary-orbit kernel is BLOCKED-domain: it iterates one
coordinate per local BLOCKED state and reconstructs the paired MAIN states from
descriptors. The shared host nevertheless inherited the older MAIN-domain launch
geometry and launched the kernel with `bm = ceil(main_n / threads)`.

Correctness was unaffected because the kernel grid-strides only until its true
BLOCKED count, but many CTAs had no BLOCKED coordinate to process.

v0.16 adds `MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH`. It changes only the HIGH orbit
launch from `bm` to the already-computed `bd = ceil(block_n / threads)`; v0.15
and earlier experimental wrappers keep the old launch for clean A/B attribution.

`factor_blockorbit_launch_geometry.cpp` computes the exact W=28/LOW=14/256-thread
job geometry over all 16,384 fixed LOW occupancy masks:

```text
sum bm over HIGH jobs = 703,136,363 CTAs
sum bd over HIGH jobs = 401,370,925 CTAs
ratio                  = 0.570829423879
reduction              = 42.9170576121%
```

Across 13 HIGH positions and 28 DP rows per residue:

```text
old launched CTAs = 255,941,636,132
v0.16 CTAs        = 146,099,016,700
```

14,913 of 16,384 HIGH jobs have `bm > bd`. This is a launch/scheduling model,
not a wall-time prediction; capped grids and cheap out-of-range CTAs mean the
runtime gain can be much smaller than 42.9%.

Local validation of the model passed with
`g++ -O3 -std=c++17 -Wall -Wextra -Werror`.

## v0.17: exact row-depth pruning inside BLOCKED orbit tasks

v0.15 already stores exact maximum-frontier-height metadata for each LOW/HIGH
factor code. v0.17 reuses the same ~1.897 MiB/GPU metadata before touching a
BLOCKED orbit task.

For DP row `r`, any state with maximum frontier height greater than `r` is
structurally unreachable. The BLOCKED-domain orbit task associated with a
BLOCKED state `d` contains:

1. `source = blocked_exclude(d,p)`, whose pair is NN/NR/NL;
2. its paired MAIN state obtained by NN->LR, NR->RN, or NL->LN;
3. `d` itself.

`factor_highorbit_rowdepth_semantics.cpp` exhaustively checks through W=12 that
if `depth(d) > cap`, both MAIN members are also above `cap`. Therefore the whole
orbit task may be skipped without dropping an update to a reachable state.

The v0.17 CUDA kernel performs the depth test before any `blockv[di]` read. This
ordering is required because v0.13 lazy initialization intentionally permits
structurally unreachable BLOCKED scratch entries to remain physically
uninitialized.

For n=27, the dense and exact-depth active task-body counts are:

```text
dense HIGH orbit bodies  = 49,145,643,968,148
active row-depth bodies  = 43,627,644,071,375
ratio                    = 0.887721485543
removable heavy bodies   = 11.2278514457%
```

This count multiplies the depth-capped W=27 BLOCKED state count by all 13 HIGH
positions. v0.17 still launches v0.16's full `bd` grid; the saving is in
FBlock/rank-dependent descriptor, aux, state-memory and arithmetic work after the
depth predicate.

Local semantic validation passed for W=6,8,10,12 with `-Werror`.

## A/B commands

v0.15 -> v0.16 isolates launch geometry:

```bash
python3 scripts/bench/b300_maskshard_tightlaunch_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.16 -> v0.17 isolates orbit-body row-depth pruning:

```bash
python3 scripts/bench/b300_maskshard_rowdepth_orbit_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

For both comparisons residues must match exactly. The attribution metric is
`high_orbit_sum_s`; total `wall_s` decides retention.

## Next candidate

The BLOCKED FBlocks are ordered by their shared HIGH/LOW boundary height. For
early rows, blocks whose boundary height already exceeds the row cap form a
suffix. A future candidate can precompute the per-job prefix end and reduce the
actual `bd` launch range row-by-row before applying v0.17's exact per-state peak
filter. This can remove about 6% of dense BLOCKED coordinate traversal in the
n=27 aggregate model, while exact row-depth pruning still removes about 11.2%
of the expensive task bodies.

Fresh nvcc and real B300x8 full-P2P validation remain required before promotion.
