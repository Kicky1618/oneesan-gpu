# RAMstream32 CPU LOW scheduling

Backend v5.17 adds an optional sticky scheduler for the persistent sparse LOW executor. The default remains the historical dynamic atomic queue.

## Why sticky scheduling is exact

The LOW window fixes the occupancy mask of the inactive HIGH positions. Each resulting `CpuLowJob` is transition-closed for the complete LOW+center window, so different fixed-HIGH occupancy groups do not write into one another. Scheduling only chooses which persistent worker evaluates each closed group; it does not change the recurrence, descriptor streams, operation ordering inside a group, or authoritative addresses.

The W=10 exhaustive selftest runs both dynamic and sticky persistent pools for two consecutive LOW generations and compares each result with the exact reference recurrence after one and two LOW windows.

## Modes

The backend accepts:

```text
CPU_LOW_SCHEDULE=dynamic
CPU_LOW_SCHEDULE=sticky
```

`dynamic` is the default. Any other value aborts.

The final solver line and plan-only output include:

```text
cpu_low_schedule=dynamic|sticky
```

Sticky runtime output also includes:

```text
cpu_low_schedule_build_s=...
```

so one-time LPT construction can be separated from dispatched LOW generations.

## Exact-work sticky partition

Sticky mode first computes an exact iteration count for every nonempty LOW occupancy group:

```text
sum over LOW positions and source factor blocks:
  HIGH rows in the group
  * (NN ops + NR ops + NL ops + LOCAL closure ops + CROSS closure ops)
```

Groups are sorted by descending exact cell count and assigned with longest-processing-time-first scheduling to the currently least-loaded worker. This partition is constructed once and reused for every grid row.

The scheduler reports:

```text
cpu_low_sticky_schedule jobs=...
  workers=...
  total_cells=...
  min_worker_cells=...
  max_worker_cells=...
  imbalance=...
  build_s=...
```

The purpose is not primarily to reduce scheduling instructions. With `CPU_LOW_CPU_LIST` set, sticky ownership makes the same occupancy group return to the same pinned worker on every row, which can improve cache and NUMA locality compared with the dynamic queue.

## Clean dynamic/sticky A/B

Use the alternating harness:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-63' \
CPU_WORKERS=32 \
CPU_HIGH_WORKERS=32 \
REPEATS=4 \
bash scripts/bench/ramstream32-cpu-low-schedule-ab.sh
```

A cost-model HIGH group file can be supplied with `CPU_HIGH_GROUPS_FILE=/path/to/groups` instead of relying on `CPU_HIGH_MAX_MIB`.

The harness alternates `dynamic-first` and `sticky-first`, verifies identical residues, verifies the backend-reported schedule mode, and records both whole-solver `wall_s` and `cpu_low_wall_s`. NUMA sampling is forced off because `move_pages` queries perturb timing.

The summary reports both whole-solver and LOW-only speedups:

```text
sticky_wall_speedup=...
sticky_low_speedup=...
```

No speedup is assumed in advance. Dynamic scheduling can remain faster when per-worker LPT balance is imperfect or memory placement does not benefit from stable ownership.

## NUMA diagnosis

For placement measurements, use the separate diagnostic runner with the desired LOW scheduler:

```bash
CPU_LOW_SCHEDULE=dynamic bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky  bash scripts/bench/ramstream32-numa-sample.sh
```

Keep CPU affinity, HIGH policy, overlap mode, and sample spacing identical between the two runs. Compare the row1/final node histograms and node-fraction drift before considering `mbind`, interleave, THP changes, or other memory-policy changes.

## Benchmark provenance

The threshold sweep, CPU HIGH policy A/B, stream calibration, and NUMA diagnostic runners all explicitly propagate and record `CPU_LOW_SCHEDULE`. This prevents schedule mode from being inherited accidentally from an interactive shell without appearing in benchmark metadata.
