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
descriptor, aux, state-memory and arithmetic work after the depth predicate.

Local semantic validation passed for W=6,8,10,12 with `-Werror`.

## v0.18: cap the BLOCKED launch range by row height

BLOCKED factor blocks are constructed in strictly increasing shared boundary
height `h=0,1,...,HIGH+1`, and their `[off,end)` ranges are contiguous. Since a
state whose boundary height exceeds row cap `r` necessarily has full frontier
depth above `r`, those trailing blocks are unreachable before the kernel runs.

v0.18 adds `MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH`. During HIGH-job construction
the host stores the cumulative BLOCKED end after each boundary height and
asserts the FBlock ordering/contiguity. For zero-based DP row `row`, the orbit
launch uses only the prefix through height `row+1`:

```text
orbit_block_n = block_depth_end[min(row+1, HIGH+1)]
bd_row        = min(65535, ceil(orbit_block_n / threads))
```

The v0.17 exact per-state predicate remains active inside this prefix, so paths
that temporarily rise above the row cap and return to a small boundary height
are still rejected exactly. v0.18 changes launch geometry only; it adds no HBM
metadata beyond v0.15.

`factor_blockorbit_rowcap_launch.cpp` pins the W=28/LOW14/256-thread model:

```text
v0.16/v0.17 CTAs = 146,099,016,700 / residue
v0.18 CTAs        = 139,099,313,353 / residue
ratio             = 0.952089319250
reduction          = 4.7910680750%
```

The summed per-job BLOCKED grid for the first rows is:

```text
row 1  126,073,708   jobs reduced 16,354 / 16,384
row 2  247,591,104   jobs reduced 16,172
row 3  333,369,706   jobs reduced 15,444
row 4  369,538,839   jobs reduced 13,442
row 5  394,328,461   jobs reduced 11,440
row 6  399,112,240   jobs reduced  8,437
row 7  401,183,738   jobs reduced  5,005
row 8  401,330,885   jobs reduced  2,002
row 9  401,370,925   jobs reduced      0
```

After row 9, the 65,535-CTA launch cap hides the remaining prefix-size
reduction, so v0.18 no longer changes the grid even though exact row-depth body
pruning remains useful through later rows.

The model probe passed locally with
`g++ -O3 -std=c++17 -Wall -Wextra -Werror` and reproduces the pinned totals.

## A/B commands

v0.15 -> v0.16 isolates the BLOCKED-domain launch correction:

```bash
python3 scripts/bench/b300_maskshard_tightlaunch_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.16 -> v0.17 isolates exact orbit-body row-depth pruning:

```bash
python3 scripts/bench/b300_maskshard_rowdepth_orbit_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

v0.17 -> v0.18 isolates host-side row-capped launch geometry:

```bash
python3 scripts/bench/b300_maskshard_rowcaplaunch_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

For all comparisons residues must match exactly. The attribution metric is
`high_orbit_sum_s`; total `wall_s` decides retention.

## Next candidate

v0.18 can remove only a contiguous suffix of FBlocks. The remaining gap to the
v0.17 exact active-body count comes from high-depth paths interleaved inside
otherwise eligible boundary-height blocks. A stronger launch compaction would
need row-specific compact BLOCKED task lists or a two-level `(HIGH row, LOW
column)` active-task representation. That is a larger metadata/indexing change
and should be modeled against its extra reads before implementation.

Fresh nvcc and real B300x8 full-P2P validation remain required before promotion.
