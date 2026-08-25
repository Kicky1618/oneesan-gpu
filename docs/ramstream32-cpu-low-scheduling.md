# RAMstream32 CPU LOW scheduling

Backend v5.22 provides four scheduling modes for the persistent sparse LOW executor. `dynamic` remains the default. `sticky`, `contiguous`, and `domain` are opt-in static-owner modes.

## Why static scheduling is exact

The LOW window fixes the occupancy mask of the inactive HIGH positions. Each resulting `CpuLowJob` is transition-closed for the complete LOW+center window, so different fixed-HIGH occupancy groups do not write into one another. Scheduling changes only which persistent worker evaluates each closed group; it does not change the recurrence, descriptor streams, operation ordering inside a group, or authoritative addresses.

The W=10 exhaustive selftest runs dynamic, LPT-sticky, contiguous, refined-domain, page-aware refined-domain, and unrefined-domain pools for two consecutive LOW generations and compares every result with the exact reference recurrence after one and two LOW windows. The boundary refiner also has a deterministic unit case: an initial `[8,8,8] | [1,1,1]` one-worker-per-domain split has pair makespan 24, and the bounded refinement moves one job to reach makespan 16 while preserving the single ordered boundary.

## Production modes

```text
CPU_LOW_SCHEDULE=dynamic
CPU_LOW_SCHEDULE=sticky
CPU_LOW_SCHEDULE=contiguous
CPU_LOW_SCHEDULE=domain
```

`dynamic` uses the historical atomic queue. `sticky` builds one exact-cell LPT partition. `contiguous` builds a min-max optimal partition whose workers own contiguous runs in numeric HIGH-occupancy-mask order.

`domain` is the NUMA-oriented middle ground. It requires:

```text
CPU_LOW_DOMAIN_SIZE=<positive workers per modeled domain>
```

For example, with 64 LOW workers arranged as two socket-local blocks of 32 workers:

```bash
CPU_LOW_CPU_LIST='0-63' \
CPU_LOW_SCHEDULE=domain \
CPU_LOW_DOMAIN_SIZE=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 64
```

Domain ownership is contiguous in numeric HIGH-mask order, but jobs inside each domain are redistributed with exact-cell LPT. `CPU_LOW_DOMAIN_SIZE` must be positive and no larger than `CPU_WORKERS`.

### Domain refinement control

The load-balancing boundary refiner is enabled by default and can be disabled without changing the binary:

```text
CPU_LOW_DOMAIN_REFINE=1   # default: refined domain boundaries
CPU_LOW_DOMAIN_REFINE=0   # initial outer-domain partition only
```

The parser also accepts `true/false`, `yes/no`, and `on/off`. Invalid values abort rather than silently selecting a mode.

Refinement OFF does not change the recurrence. It keeps the initial contiguous domain ranges produced by the outer normalized-cap partition, then performs the same exact-cell LPT assignment inside each domain. Refinement ON starts from that same assignment and only moves ordered domain boundaries when the local two-domain LPT objective improves. The W=10 selftest validates both modes for two consecutive LOW generations.

### v5.22 page-aware tie-break

v5.22 adds a second, independent optimization pass:

```text
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0   # default
CPU_LOW_DOMAIN_PAGE_TIEBREAK=1   # opt-in page-aware second stage
```

It is deliberately valid only with

```text
CPU_LOW_SCHEDULE=domain
CPU_LOW_DOMAIN_REFINE=1
```

The ordinary load refiner runs first and is unchanged. The page-aware pass then considers nearby ordered domain boundaries, but a candidate is eligible only if the affected two domains have exactly the same LPT load tuple as the current boundary:

```text
(max(left LPT makespan, right LPT makespan),
 left LPT makespan + right LPT makespan)
```

Among load-equivalent candidates, v5.22 minimizes the boundary-page penalty lexicographically:

```text
1. fewer 2 MiB boundary pages
2. if tied, fewer 4 KiB boundary pages
```

This makes locality a tie-break rather than a competing objective. A page-aware move cannot increase the affected pair's maximum worker load, and all other domains are unchanged. The implementation rebuilds the final LPT assignments and asserts that global `max_worker_cells` did not increase.

The page penalty is computed directly from the factorized authoritative layout. For each factor block, it finds the byte boundary between the two contiguous HIGH-mask ranges and counts an unaligned boundary as exposure to the corresponding 2 MiB or 4 KiB page. Main and blocked arrays are separate address spaces and are counted separately.

The helper's `page_penalty_*` is a sum of per-domain-boundary penalties used by the local optimizer. It is not necessarily identical to the probe's global unique `hybrid_domain_cross_domain_pages_*` count when there are three or more domains, because two different boundaries can in principle refer to the same VM page. Use the global unique page counts from the schedule-plan probe when evaluating the final assignment.

Production provenance and diagnostics now include:

```text
cpu_low_schedule=dynamic|sticky|contiguous|domain
cpu_low_domain_size=...
cpu_low_domain_refine=0|1
cpu_low_domain_page_tiebreak=0|1
cpu_low_schedule_build_s=...
cpu_low_contiguous_optimal_cap=...
cpu_low_domain_outer_normalized_cap=...
cpu_low_domain_active_domains=...
cpu_low_domain_refined_boundaries=...
cpu_low_domain_refined_job_moves=...
cpu_low_domain_page_boundary_moves=...
cpu_low_domain_page_moved_jobs=...
cpu_low_domain_page_penalty_2m_before=...
cpu_low_domain_page_penalty_2m_after=...
cpu_low_domain_page_penalty_4k_before=...
cpu_low_domain_page_penalty_4k_after=...
cpu_low_domain_page_build_s=...
```

`cpu_low_domain_normalized_cap` is retained as a compatibility alias for `cpu_low_domain_outer_normalized_cap`. The ordinary domain scheduler stderr line records `refine=0|1`; when page-aware mode is enabled a separate `cpu_low_domain_page_tiebreak` line records its before/after load and page-penalty values.

## Exact work model

All static modes use the same exact structural work estimate for every nonempty LOW occupancy group:

```text
sum over LOW positions and source factor blocks:
  HIGH rows in the group
  * (NN ops + NR ops + NL ops + LOCAL closure ops + CROSS closure ops)
```

### LPT sticky

Groups are sorted by descending exact-cell work and assigned to the currently least-loaded worker. The partition is reused for every grid row.

### Contiguous

Groups are sorted by numeric HIGH occupancy mask. Production binary-searches the smallest feasible maximum worker load under the contiguous-worker constraint. Greedy feasibility determines whether a candidate cap fits in the worker count. If the optimum needs fewer segments than workers, existing segments are split without increasing the cap.

### Domain: outer partition

For a domain containing `k` workers, the initial ordered partition gives it a contiguous mask range with total exact work bounded by `k * outer_normalized_cap`. Production binary-searches the smallest outer normalized cap for which all ordered jobs fit across the available domains.

This outer cap is a domain-total/worker-count normalization used to choose the initial boundaries. It is not a mathematical upper bound on each worker after LPT. A large indivisible job or LPT packing can produce a worker load greater than this value; actual `max_worker_cells` and `imbalance` are the relevant balance measurements.

### Boundary refinement

After the initial domain ranges are constructed, the refiner can optimize the actual LPT makespan without giving up domain contiguity.

For every adjacent pair of nonempty domains it searches at most 32 occupancy jobs to each side of the current boundary. Each candidate is evaluated by running the same exact-cell LPT assignment used by production inside the two affected domains. A move is accepted only when the pair objective improves lexicographically:

```text
1. smaller max(left LPT makespan, right LPT makespan)
2. if tied, smaller left makespan + right makespan
```

Two bounded passes are used, with the second pass visiting boundaries in reverse order. Therefore an accepted move cannot increase the current maximum load of the two affected domains; all other domains are unchanged by that move. The search changes only the position of an ordered domain boundary, so each domain still owns one contiguous HIGH-mask interval.

Metrics:

```text
cpu_low_domain_refined_boundaries = number of accepted load-refinement boundary moves
cpu_low_domain_refined_job_moves  = sum of absolute load-refinement boundary displacements
```

The page-aware pass has separate counters so the two optimization stages remain distinguishable.

The radius and pass count are deliberately bounded because the schedule is built on the host before the repeated LOW generations. `cpu_low_schedule_build_s` must be checked together with runtime savings.

Thus the structural tradeoff is:

```text
sticky/LPT: best unconstrained worker balance, potentially many domain cuts
contiguous: every worker owns one ordered mask interval
 domain:    every NUMA domain owns one ordered interval, refined LPT inside domains
 page tie:  same refined load objective, locality only breaks exact load ties
```

## Preflight balance and page-cut probe

The schedule geometry can be inspected without allocating the multi-terabyte authoritative RAM arrays:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32 --workers
```

The probe constructs the actual production sticky, contiguous, and domain pools and analyzes their cached assignments. The same binary can inspect all three domain variants:

```bash
CPU_LOW_DOMAIN_REFINE=0 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32

CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32

CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
```

Important fields include:

```text
imbalance=...
cross_domain_pages_4k=...
cross_domain_pages_2m=...
contiguous_imbalance=...
contiguous_cross_domain_pages_4k=...
contiguous_cross_domain_pages_2m=...
hybrid_domain_refine=...
hybrid_domain_page_tiebreak=...
hybrid_domain_imbalance=...
hybrid_domain_cross_domain_pages_4k=...
hybrid_domain_cross_domain_pages_2m=...
hybrid_domain_outer_normalized_cap=...
hybrid_domain_refined_boundaries=...
hybrid_domain_refined_job_moves=...
hybrid_domain_page_boundary_moves=...
hybrid_domain_page_penalty_2m_before=...
hybrid_domain_page_penalty_2m_after=...
hybrid_domain_page_penalty_4k_before=...
hybrid_domain_page_penalty_4k_after=...
```

The raw probe retains the historical `hybrid_domain_*` prefix for compatibility; those fields describe the current production `CPU_LOW_SCHEDULE=domain` assignment.

`--domain-size 32` models worker IDs `0..31` as domain 0, `32..63` as domain 1, and so on. This only matches the hardware experiment if `CPU_LOW_CPU_LIST` is arranged in corresponding socket-local blocks.

These page counts are static ownership exposures, not measured remote-memory bytes. First-touch placement, AutoNUMA, THP, caches, and GPU DMA still determine actual traffic.

### Refinement preflight A/B

Before spending a full residue on timing, compare refined and unrefined domain plans across the intended worker/socket shapes:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-refine-plan-ab.sh
```

The runner invokes the same production schedule-plan binary twice per topology with `CPU_LOW_DOMAIN_REFINE=0` and `1`. It verifies zero boundary moves in the unrefined variant and records domain imbalance, maximum worker exact-cell load, cross-domain 4 KiB/2 MiB pages, cross-worker pages, outer normalized cap, accepted moves, and build time.

The runner enforces a structural no-regression contract: because every accepted load-refinement move lowers or preserves the two affected domains' LPT maximum and leaves all other domains unchanged, refined `max_worker_cells` must never exceed the unrefined value. A violation exits nonzero as `refinement max-worker regression`.

Its paired summary reports `imbalance_speedup`, `max_worker_cells_saved`, `cross_domain_4k_delta`, `cross_domain_2m_delta`, and `refine_extra_build_s`, together with one classification:

```text
no_change
dominates
balance_page_tradeoff
page_only_improvement
page_regression_without_balance_gain
```

`dominates` is the strongest static signal for refinement. `balance_page_tradeoff` should proceed to clean runtime A/B plus NUMA sampling rather than being accepted or rejected from the static model alone.

### Page-aware preflight

For v5.22, compare the same refined schedule with and without the page tie-break. A direct invocation is:

```bash
CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
```

The primary invariants are:

```text
page-aware max_worker_cells <= ordinary refined max_worker_cells
(page_penalty_2m_after, page_penalty_4k_after)
  <= (page_penalty_2m_before, page_penalty_4k_before)
```

For final topology evaluation, compare `hybrid_domain_cross_domain_pages_2m` and `_4k` between the two complete probe runs. These are global unique page counts and are the appropriate structural locality metrics across multiple domain boundaries.

## Topology sweep and Pareto analysis

Several worker/domain layouts can be compared without allocating authoritative state:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
MAX_IMBALANCE=1.05 \
bash scripts/bench/ramstream32-cpu-low-schedule-plan-sweep.sh
```

Each entry is `workers:domain-size`. The persisted TSV uses production terminology:

```text
lpt_*
contiguous_*
domain_*
domain_refined_boundaries
domain_refined_job_moves
```

The analyzer `scripts/tools/analyze_cpu_low_schedule_plan_sweep.py` computes the Pareto frontier over three objectives:

```text
minimize worker imbalance
minimize cross-domain 4 KiB boundary pages
minimize cross-domain 2 MiB boundary pages
```

It reports `scheme=lpt`, `scheme=contiguous`, or `scheme=domain`. Legacy TSV files with `hybrid_*` columns and `--scheme hybrid` remain accepted as aliases for `domain`.

A candidate is omitted only when another candidate is no worse in all three objectives and strictly better in at least one. The refinement-count columns are diagnostic metadata rather than additional Pareto objectives.

## Clean timing comparison

The original dynamic versus LPT microcomparison remains available:

```bash
REPEATS=4 bash scripts/bench/ramstream32-cpu-low-schedule-ab.sh
```

For all four production modes, use the four-way clean harness. Keep both domain optimization controls fixed for the whole run:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-95' \
CPU_WORKERS=64 \
CPU_HIGH_WORKERS=32 \
CPU_LOW_DOMAIN_SIZE=32 \
CPU_LOW_DOMAIN_REFINE=1 \
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
REPEATS=8 \
bash scripts/bench/ramstream32-cpu-low-schedule-compare.sh
```

The harness uses a cyclic-latin-4 order:

```text
dynamic -> sticky -> contiguous -> domain
sticky -> contiguous -> domain -> dynamic
contiguous -> domain -> dynamic -> sticky
domain -> dynamic -> sticky -> contiguous
```

Over every four repeats each schedule appears once in every run position. Use a fixed domain refinement/page setting for a four-way schedule comparison.

### Isolate load refinement itself

To compare only the ordinary boundary-refinement step, use the dedicated same-binary A/B runner:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-95' \
CPU_WORKERS=64 \
CPU_HIGH_WORKERS=32 \
CPU_LOW_DOMAIN_SIZE=32 \
REPEATS=4 \
bash scripts/bench/ramstream32-cpu-low-domain-refine-ab.sh
```

Odd repeats run `refine=0 -> refine=1`; even repeats reverse that order. Both variants use the same binary, domain size, HIGH policy, affinities, worker counts, modulus, and GPU target. NUMA sampling is forced off. The harness checks final stdout and scheduler stderr provenance, requires zero reported boundary moves when refinement is disabled, and verifies identical residues.

Its final comparison reports:

```text
refine_speedup       = mean wall(refine=0) / mean wall(refine=1)
refine_low_speedup   = mean LOW wall(refine=0) / mean LOW wall(refine=1)
refine_extra_build_s = mean schedule build(refine=1) - mean schedule build(refine=0)
```

A useful refinement must recover more repeated LOW runtime than it adds to one-time schedule construction.

### Isolate page-aware tie-break

The page-aware experiment must keep ordinary refinement enabled in both variants and change only `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0|1`. This isolates locality from the previous load-balancing change. Compare whole wall time, `cpu_low_wall_s`, page-tie build time, page moves, and global cross-domain page counts from the preflight probe. A page-aware schedule is useful only if the locality gain survives the real NUMA/cache/THP behavior of the host.

No schedule or refinement setting is assumed faster. Dynamic can win from fine-grained balancing. Sticky can win from stable ownership with near-perfect load balance. Contiguous can win when page/address locality dominates. Domain is intended to retain most of sticky's balance while reducing cross-NUMA-domain ownership boundaries; the first refinement stage attacks makespan, and v5.22 uses locality only to break exact load ties.

## NUMA diagnosis

`move_pages` sampling is diagnostic and perturbs timing, so run it separately:

```bash
CPU_LOW_SCHEDULE=dynamic    bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky     bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=contiguous bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
  CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
  CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 bash scripts/bench/ramstream32-numa-sample.sh
```

Keep HIGH policy, worker counts, affinity lists, overlap mode, and sample spacing identical. Compare row1/final node histograms and node-fraction drift with clean timing before considering `mbind`, interleave, THP changes, or other memory policy.

## Benchmark provenance

LOW schedule, domain size, load refinement, and page-aware tie-break are all benchmark conditions. Any timing, HIGH cost calibration, stream-weight fit, policy A/B, or NUMA measurement intended for direct comparison must propagate and record all enabled LOW scheduling controls. Changing one invalidates a direct calibration comparison.
