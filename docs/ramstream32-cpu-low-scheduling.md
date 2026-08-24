# RAMstream32 CPU LOW scheduling

Backend v5.18 provides three scheduling modes for the persistent sparse LOW executor. `dynamic` remains the default; `sticky` and `contiguous` are opt-in static-owner experiments.

## Why static scheduling is exact

The LOW window fixes the occupancy mask of the inactive HIGH positions. Each resulting `CpuLowJob` is transition-closed for the complete LOW+center window, so different fixed-HIGH occupancy groups do not write into one another. Scheduling only chooses which persistent worker evaluates each closed group; it does not change the recurrence, descriptor streams, operation ordering inside a group, or authoritative addresses.

The W=10 exhaustive selftest runs dynamic, LPT-sticky, and contiguous persistent pools for two consecutive LOW generations and compares all three with the exact reference recurrence after one and two LOW windows.

## Modes

```text
CPU_LOW_SCHEDULE=dynamic
CPU_LOW_SCHEDULE=sticky
CPU_LOW_SCHEDULE=contiguous
```

`dynamic` uses the historical atomic queue. `sticky` builds a one-time exact-cell LPT partition. `contiguous` builds a one-time min-max optimal partition whose workers own contiguous runs in numeric HIGH-occupancy-mask order. Any other value aborts.

The final solver line and plan-only output include:

```text
cpu_low_schedule=dynamic|sticky|contiguous
```

Runtime output also includes `cpu_low_schedule_build_s`. Contiguous mode additionally reports `cpu_low_contiguous_optimal_cap`.

## Exact work model

Both static modes count the exact structural iterations for every nonempty LOW occupancy group:

```text
sum over LOW positions and source factor blocks:
  HIGH rows in the group
  * (NN ops + NR ops + NL ops + LOCAL closure ops + CROSS closure ops)
```

### LPT sticky

Groups are sorted by descending exact-cell work and assigned to the currently least-loaded worker. The partition is reused for every grid row.

```text
cpu_low_sticky_schedule jobs=...
  workers=...
  total_cells=...
  min_worker_cells=...
  max_worker_cells=...
  imbalance=...
  build_s=...
```

### Contiguous sticky ownership

Groups are sorted by numeric HIGH occupancy mask. Production binary-searches the smallest feasible maximum worker load under the contiguous-order constraint; greedy feasibility decides whether a candidate cap fits in the worker count. If the optimum uses fewer segments than workers, segments are split without increasing the cap.

```text
cpu_low_contiguous_schedule jobs=...
  workers=...
  total_cells=...
  min_worker_cells=...
  max_worker_cells=...
  imbalance=...
  optimal_cap=...
  build_s=...
```

This mode trades some unconstrained load-balancing freedom for stable address-local ownership. With `CPU_LOW_CPU_LIST` ordered in socket-local blocks, neighboring mask ranges can remain on nearby workers or the same NUMA domain across every row.

## Preflight balance and page-cut probe

The schedule geometry can be inspected without allocating the authoritative RAM arrays:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
./build/ramstream32_cpu_low_schedule_plan_n27 27 32
```

The first output line reports the production LPT work balance and static page-boundary exposure:

```text
imbalance=...
shared_pages_4k=...
cross_worker_pages_4k=...
cross_worker_auth_page_fraction_4k=...
shared_pages_2m=...
cross_worker_pages_2m=...
cross_worker_auth_page_fraction_2m=...
```

The factorized storage orders HIGH codes by occupancy mask within each height. If two adjacent nonempty mask ranges meet inside one 4 KiB or 2 MiB page, the page is shared by the two LOW groups. The probe counts unique shared boundary pages and the subset whose logical owners differ.

The same probe constructs the actual production contiguous pool and reports its cached assignment:

```text
contiguous_active_workers=...
contiguous_optimal_cap=...
contiguous_imbalance=...
contiguous_cross_worker_pages_4k=...
contiguous_cross_worker_auth_page_fraction_4k=...
contiguous_cross_worker_pages_2m=...
contiguous_cross_worker_auth_page_fraction_2m=...
```

This exposes the tradeoff before a multi-terabyte residue run:

```text
LPT:        unconstrained static load balance, potentially many mask-boundary cuts
contiguous: min-max optimal under mask-order constraint, potentially far fewer cuts
```

### NUMA-domain cut model

Cross-worker sharing is more conservative than cross-socket sharing. The probe can group consecutive worker IDs into modeled NUMA domains:

```bash
./build/ramstream32_cpu_low_schedule_plan_n27 \
  27 64 --domain-size 32 --workers
```

`--domain-size 32` models workers `0..31` as domain 0, `32..63` as domain 1, and so on. This matches experiments where `CPU_LOW_CPU_LIST` is arranged in socket-local blocks and does not wrap.

It reports both LPT and contiguous domain cuts:

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

### Domain-hybrid research plan

The same `--domain-size` run also evaluates a research-only middle ground that is not yet a production `CPU_LOW_SCHEDULE` mode. `domain-hybrid` constrains only NUMA-domain ownership to contiguous numeric mask ranges, then uses exact-cell LPT among workers inside each domain.

The outer partition binary-searches a normalized per-worker domain capacity. For a domain containing `k` workers, its contiguous mask interval may contain at most `k * cap` exact cells. Once the domain ranges are fixed, jobs inside each range are sorted by exact work and assigned LPT to that domain's workers.

This gives a three-way structural comparison:

```text
sticky/LPT:     best unconstrained worker balance; many possible domain cuts
contiguous:     every worker owns one mask interval; minimum worker-order cuts
 domain-hybrid: each domain owns one mask interval; LPT restored inside domain
```

The probe reports:

```text
hybrid_domain_active_domains=...
hybrid_domain_optimal_per_worker_cap=...
hybrid_domain_min_worker_cells=...
hybrid_domain_max_worker_cells=...
hybrid_domain_imbalance=...
hybrid_domain_cross_worker_pages_4k=...
hybrid_domain_cross_worker_pages_2m=...
hybrid_domain_cross_domain_pages_4k=...
hybrid_domain_cross_domain_pages_2m=...
```

The important comparison is `hybrid_domain_imbalance` versus `contiguous_imbalance`, together with `hybrid_domain_cross_domain_pages_*` versus ordinary LPT's `cross_domain_pages_*`. If domain-hybrid preserves most of LPT's balance while driving domain cuts close to contiguous, it is the stronger candidate for the next production scheduler. It deliberately allows many cross-worker boundaries inside one domain because those do not by themselves imply remote-NUMA traffic.

Sanity cases are `--domain-size 1`, where cross-domain equals cross-worker, and `--domain-size <workers>`, where all workers are one modeled domain and cross-domain cuts are zero.

These are static ownership exposures, not measured remote-memory bytes. First-touch placement, AutoNUMA, THP, caches, and GPU DMA still determine actual traffic.

Use `--workers` to dump every worker's LPT, contiguous, and domain-hybrid exact-cell load. The probe constructs topology/descriptor metadata and LOW jobs only; it does not mmap the multi-terabyte authoritative state arrays.

## Clean timing comparisons

For the original dynamic↔LPT comparison:

```bash
REPEATS=4 bash scripts/bench/ramstream32-cpu-low-schedule-ab.sh
```

For all three production modes, use the cyclic three-way harness:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-63' \
CPU_WORKERS=32 \
CPU_HIGH_WORKERS=32 \
REPEATS=6 \
bash scripts/bench/ramstream32-cpu-low-schedule-compare.sh
```

Its run order rotates through:

```text
dynamic -> sticky -> contiguous
sticky -> contiguous -> dynamic
contiguous -> dynamic -> sticky
```

so each schedule occupies each run position equally over multiples of three repeats. The harness verifies identical residues and backend schedule provenance, forces `RAMSTREAM_NUMA_SAMPLE_MIB=0`, and reports whole-solver and LOW-only ratios between every pair.

A cost-model HIGH policy can be supplied with `CPU_HIGH_GROUPS_FILE=/path/to/groups` instead of relying on `CPU_HIGH_MAX_MIB`.

No schedule is assumed faster. Dynamic can win from finer load balancing; LPT can win from stable ownership with near-perfect balance; contiguous can win when reduced address/page boundary sharing outweighs its additional balance constraint. Domain-hybrid should not be timed as a production mode until its static tradeoff justifies promoting it from the probe.

## NUMA diagnosis

Use separate diagnostic runs because `move_pages` sampling perturbs timing:

```bash
CPU_LOW_SCHEDULE=dynamic    bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky     bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=contiguous bash scripts/bench/ramstream32-numa-sample.sh
```

Keep affinity, HIGH policy, overlap mode, and sample spacing identical. Compare row1/final node histograms and node-fraction drift with clean timing before considering `mbind`, interleave, THP changes, or other memory policy.

## Benchmark provenance

The LOW schedule comparison and NUMA diagnostic runners explicitly propagate and record the selected schedule. CPU HIGH threshold sweeps, policy A/B runs, and stream calibration now also propagate and record `CPU_LOW_SCHEDULE`, including `contiguous`, so mixed CPU/GPU calibration results retain the LOW scheduling condition that produced them.
