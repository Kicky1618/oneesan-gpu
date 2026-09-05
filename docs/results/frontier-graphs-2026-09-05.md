# Frontier DP graph reuse (2026-09-05)

This follows the configuration-transfer and blocked-state-cache changes in
[the preceding measurement](frontier-dp-2026-09-05.md). The baseline for this iteration
is that already optimized solver, not the original solver.

## Implementation

The solver caches CUDA executable graphs by main/blocked state counts, window endpoints,
thread count, cache choices, stream policy and capture scope. Occupancy masks and CRT
moduli are not baked into graph arguments: kernels read the current device configuration.
Graphs are device-local. The I/O policy is fixed when the device context is initialized.

The selected full-group path captures gather, all cell transitions, and scatter. It removes
intermediate CPU waits; a stream synchronization at the group boundary completes the work
before another group changes the configuration. An optional parallel I/O policy schedules
main and blocked gather/scatter on their respective streams.

Configuration copies are explicitly enqueued on the main nonblocking stream before graph
launch. Their host source is the prepared schedule, which remains alive throughout all rows
and moduli. This explicit ordering is necessary: the moved gather kernels must not depend
on implicit ordering with a different/default stream.

Growing the scratch arena destroys all cached executable graphs before releasing the old
allocation. Reusing capacity at a different layout retains graphs, but the shape/cache key
selects only the graph with matching buffer offsets. This adds graph metadata, not another
DP count vector; count and mate-cache admission retain the existing scratch policy.

## Exactness argument

Write `G_M,G_D` for the two gathers, `T_j^M,T_j^D` for the two output kernels at transition
stage j, and `S_M,S_D` for the scatters. Both gathers finish before either first-stage kernel.
Each later stage waits for both preceding output kernels. At the special first column,
identity copy and blocked-buffer clearing finish before the update kernels. Both final
outputs finish before scatter, and both scatters finish before the next group.

The graphs preserve these edges. Parallel I/O only removes ordering between independent
main/blocked gathers and between independent main/blocked scatters. Their output buffers
are disjoint. The first-column modular atomics retain their original dependencies. Each
kernel, argument, predecessor multiplicity and modular operation is unchanged.

A graph key fixes the kernel sizes, positions, pointers within the arena and scratch-cache
presence. Device allocation identity is protected by invalidation on arena growth. Runtime
configuration is ordered before the graph and immutable until its completion. Therefore
replay executes the same recurrence on the current group and current modulus. No new
pruning, approximation, or conjectured counting identity is introduced.

## Controls and reproduction

Defaults: whole-group graphs (`2`), parallel I/O (`1`), and dual transition streams.

- `GRIDFP_TRANSITION_GRAPHS=0`: ordinary launches; `1`: capture transitions only;
  `2`: capture whole groups. Whole-group capture falls back to transition-only capture
  if interval I/O is selected.
- `GRIDFP_PARALLEL_GROUP_IO=0/1`: serial/parallel main-blocked I/O in full-group graphs.
- Existing blocked-cache and single-stream controls continue to work.
- `python scripts/bench/bench_factor_division.py --optimization graphs --n 20 --arch sm_86`
  isolates graph scope against ordinary launches with explicit stream-ordered uploads.
- Use `--optimization graph-io` to isolate parallel I/O.
- `python scripts/test/frontier-runtime.py /path/to/n9-binary` checks all three graph modes,
  cached/uncached targets and single/dual streams, with two exact CRT residues. Repeat with
  `GRIDFP_PARALLEL_GROUP_IO=1` to exercise parallel I/O explicitly.
- `scripts/test/gridfp-reverse.sh --gpu` includes the actual arena/graph lifetime test.

`group_graph_sum_s` records the complete group duration for full captures, including host
setup and graph creation/replay. The historical gather/transition/scatter timers do not
split full captures and are zero for those groups. They must not be interpreted as zero
GPU work. Build, replay, eviction and live-entry counters are also reported and per-modulus
counters reset between residues.

Measurements below use one RTX 3070, n=20 and 512 MiB scratch. The desktop remains active;
use within-trial balanced comparisons. No B300 runtime, physical multi-GPU speedup or
bandwidth saturation is claimed.

## Measurements

Scope isolation: three measured runs per variant after warmup, rotating the run order.

| Variant | Median wall seconds |
|---|---:|
| previous | 8.565030 |
| stream_order | 8.538300 |
| transitions | 8.476900 |
| whole_group | 8.182590 |

Final comparison: six measured runs per variant after warmup, using all six permutations
of the three variants. The baseline is the frozen preceding solver; both candidates use
the same binary, with serial/parallel I/O selected by environment. All 21 runs, including
warmups, matched the known residue.

| Variant | Median wall seconds | Median active group seconds |
|---|---:|---:|
| previous | 8.548960 | 8.277500 |
| serial_io | 8.154515 | 7.887695 |
| parallel_io | 8.001610 | 7.735020 |

Selected default: **1.068x** the previous version, **6.40%** less elapsed time.
The n=20 solve built 22 graphs and replayed them 30,698 times (30,720 group executions).
This is an execution-overhead improvement; these measurements do not isolate device
kernel throughput or establish a memory-bandwidth bottleneck.

Raw data and source fingerprints: [frontier-graphs-2026-09-05.json](frontier-graphs-2026-09-05.json).

## Final validation

- n=9: all 12 graph/cache/stream combinations matched two exact residues. Both serial
  and parallel I/O were exercised. Graph replay and reuse across moduli were asserted;
  full-graph timer resets were checked.
- Final n=9 solver: Compute Sanitizer memcheck, two moduli, zero errors.
- Actual production arena lifecycle test: eviction on allocation growth, retention on
  same-capacity layout changes and replay after restoring a layout; memcheck zero errors.
- Actual reverse main kernel: all 12 cached/uncached rank-policy comparisons passed with
  the asynchronous configuration upload helper.
- n=20, 40 MiB scratch: residues 2308006916 and 3704549185 modulo 4294967291 and
  4294966997. The first modulus exercised an actual arena-growth eviction; the second
  reused all 30,720 group graphs with zero new builds.
- n=21: residue 2124618149 modulo 4294966997; 49,117 replays and one actual eviction.
- n=27 / width 28, LUT split 14/13: B300 sm_103 cubin compilation passed.
- Updated benchmark helper: both ordinary-launch and full-graph n=9 builds/runs passed.

The change is verified by these targeted checks; B300 execution and physical multi-GPU
execution remain untested on the available single-GPU machine.
