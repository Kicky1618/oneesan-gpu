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

## Preflight balance and page-cut probe

The exact schedule geometry can be inspected without allocating the authoritative RAM arrays:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
./build/ramstream32_cpu_low_schedule_plan_n27 27 32
```

The first output line reports the production LPT plan:

```text
cpu_low_schedule_plan OK n=27 workers=32 jobs=...
  total_cells=...
  min_worker_cells=...
  max_worker_cells=...
  avg_worker_cells=...
  imbalance=...
  shared_pages_4k=...
  cross_worker_pages_4k=...
  cross_worker_of_shared_4k=...
  cross_worker_auth_page_fraction_4k=...
  shared_pages_2m=...
  cross_worker_pages_2m=...
  cross_worker_of_shared_2m=...
  cross_worker_auth_page_fraction_2m=...
```

`imbalance` is `max_worker_cells / avg_worker_cells`; values near 1 mean the static LPT partition is well balanced before any locality effect is considered.

The factorized storage orders HIGH codes by occupancy mask within each height. When two adjacent nonempty mask ranges meet inside a 4 KiB or 2 MiB page, that page is shared by the two LOW groups. The probe counts the unique shared boundary pages and the subset for which the two groups have different worker owners. These are static exposure metrics, not measured cache-line traffic or NUMA remote-memory traffic.

### Optimal contiguous comparison

The same probe also computes a research-only alternative that is not yet used by production: workers receive contiguous runs in numeric HIGH-occupancy-mask order.

Among all order-preserving contiguous partitions, the probe finds the minimum possible maximum worker load by binary-searching the load cap and using greedy ordered-partition feasibility. If the optimal cap uses fewer segments than available workers, existing segments are split without increasing the cap. Thus `contiguous_imbalance` is the min-max optimum subject to the contiguous-mask constraint, not a target-load heuristic.

The output adds:

```text
contiguous_active_workers=...
contiguous_optimal_cap=...
contiguous_min_worker_cells=...
contiguous_max_worker_cells=...
contiguous_imbalance=...
contiguous_cross_worker_pages_4k=...
contiguous_cross_worker_of_shared_4k=...
contiguous_cross_worker_auth_page_fraction_4k=...
contiguous_cross_worker_pages_2m=...
contiguous_cross_worker_of_shared_2m=...
contiguous_cross_worker_auth_page_fraction_2m=...
```

This directly exposes the tradeoff:

```text
LPT:        lowest unconstrained load imbalance, potentially many mask-boundary cuts
contiguous: optimal load balance under mask-order constraint, potentially far fewer page cuts
```

A large reduction in cross-worker page cuts with only a small increase in `contiguous_imbalance` is evidence for implementing a page-aware production scheduler. A large imbalance penalty is evidence for keeping LPT and seeking a NUMA-node-level hybrid instead.

### NUMA-domain cut model

Cross-worker sharing is deliberately more conservative than cross-socket sharing. Two pinned workers can share a boundary page while remaining on the same NUMA node. The probe therefore accepts an optional static domain model:

```bash
./build/ramstream32_cpu_low_schedule_plan_n27 \
  27 64 --domain-size 32 --workers
```

`--domain-size 32` models worker IDs `0..31` as domain 0, `32..63` as domain 1, and so on. This matches an experiment where `CPU_LOW_CPU_LIST` is ordered in socket-local blocks and the worker count does not wrap around the CPU list.

The output then includes LPT and contiguous domain cuts:

```text
domain_size=32
domains=2
cross_domain_pages_4k=...
cross_domain_auth_page_fraction_4k=...
cross_domain_pages_2m=...
cross_domain_auth_page_fraction_2m=...
contiguous_cross_domain_pages_4k=...
contiguous_cross_domain_auth_page_fraction_4k=...
contiguous_cross_domain_pages_2m=...
contiguous_cross_domain_auth_page_fraction_2m=...
```

This is still a static ownership model. It does not read Linux CPU topology and does not prove that a page physically resides on the modeled node. Use it to estimate which schedule creates cross-domain ownership boundaries, then compare against the measured `move_pages` histogram.

Useful sanity cases are:

```text
--domain-size 1        every worker is its own domain; cross-domain == cross-worker
--domain-size workers  all workers are one domain; cross-domain == 0
```

Do not interpret cross-worker or cross-domain page counts as remote-memory bytes. They identify boundary pages with multiple logical owners; actual traffic depends on first-touch placement, AutoNUMA, THP, cache behavior, and GPU DMA.

Use `--workers` to dump every worker's LPT and contiguous exact-cell load:

```bash
./build/ramstream32_cpu_low_schedule_plan_n27 27 32 --workers
```

The table includes `cells/fraction` for LPT and `contiguous_cells/contiguous_fraction` for the ordered partition, plus the modeled domain when `--domain-size` is supplied.

This probe constructs only topology/descriptor metadata and the LOW job plan. It does not mmap the multi-terabyte authoritative state arrays and does not require an actual solver residue run.

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
