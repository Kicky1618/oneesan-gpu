# RAMstream32 CPU LOW scheduling

The hybrid RAMstream backend provides four persistent sparse LOW scheduling modes. `dynamic` remains the default; `sticky`, `contiguous`, and `domain` are opt-in static-owner schedules.

The core backend banner remains v5.22. The page-aware second-stage scheduler was revised in v5.23 without changing the recurrence or authoritative storage format.

## Exactness

The LOW window fixes the occupancy mask of the inactive HIGH positions. Each `CpuLowJob` is transition-closed for the complete LOW+center window, so distinct fixed-HIGH occupancy groups do not write into one another. Scheduling changes only which persistent worker evaluates each closed group. It does not change descriptors, operation order inside a group, recurrence arithmetic, or authoritative addresses.

The W=10 exhaustive selftest runs dynamic, LPT-sticky, contiguous, refined-domain, page-aware refined-domain, and unrefined-domain pools for two consecutive LOW generations and compares every state with the reference recurrence after one and two LOW windows.

## Production modes

```text
CPU_LOW_SCHEDULE=dynamic
CPU_LOW_SCHEDULE=sticky
CPU_LOW_SCHEDULE=contiguous
CPU_LOW_SCHEDULE=domain
```

`dynamic` uses the atomic work queue.

`sticky` computes exact structural work per occupancy group, sorts by descending work, and performs one LPT assignment. The same worker owns the same groups on every row.

`contiguous` sorts groups by numeric HIGH occupancy mask and computes a min-max contiguous worker partition.

`domain` is the NUMA-oriented middle ground: every modeled NUMA domain owns one contiguous HIGH-mask interval, while jobs inside a domain are redistributed by exact-cell LPT.

Domain mode requires:

```text
CPU_LOW_DOMAIN_SIZE=<workers per modeled domain>
```

For 64 LOW workers modeled as two groups of 32:

```bash
CPU_LOW_CPU_LIST='0-63' \
CPU_LOW_SCHEDULE=domain \
CPU_LOW_DOMAIN_SIZE=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 64
```

`CPU_LOW_DOMAIN_SIZE` must be positive and no larger than `CPU_WORKERS`.

## Stage 1: load refinement

The ordinary domain-boundary refiner is controlled by:

```text
CPU_LOW_DOMAIN_REFINE=1   # default
CPU_LOW_DOMAIN_REFINE=0
```

The initial outer partition bounds each domain's total exact work by `domain_workers * outer_normalized_cap`. It is only a domain-total normalization; it is not a mathematical upper bound on the final per-worker LPT load.

When refinement is enabled, each adjacent domain pair searches at most 32 jobs to either side of the current ordered boundary. Candidates are evaluated with the same exact-cell LPT used by production. The load objective is lexicographic:

```text
1. smaller max(left LPT maximum, right LPT maximum)
2. if tied, smaller left LPT maximum + right LPT maximum
```

Two bounded passes are used, with the second pass visiting boundaries in reverse order. Every accepted move lowers or preserves the affected pair's maximum worker load; all other domains are unchanged. Therefore the global maximum worker load cannot increase.

Diagnostics:

```text
cpu_low_domain_refined_boundaries=...
cpu_low_domain_refined_job_moves=...
```

The W=10 deterministic refinement case starts from `[8,8,8] | [1,1,1]` with one worker per domain. The initial pair maximum is 24 and one boundary move reaches 16.

## Stage 2: page-aware boundary search

The second stage is opt-in:

```text
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0   # default
CPU_LOW_DOMAIN_PAGE_TIEBREAK=1
```

It is valid only with:

```text
CPU_LOW_SCHEDULE=domain
CPU_LOW_DOMAIN_REFINE=1
```

The ordinary load refinement always runs first.

### v5.22 strict objective

The first page-aware implementation admitted a candidate only when both values below were exactly unchanged:

```text
pair_max = max(left LPT maximum, right LPT maximum)
pair_sum = left LPT maximum + right LPT maximum
```

Only then did it compare 2 MiB and 4 KiB boundary-page penalties. This was safe but unnecessarily restrictive: `pair_sum` equality is not required to preserve the solver's static parallel critical-path bound.

### v5.23 relaxed objective

v5.23 keeps the safety condition that matters:

```text
candidate_pair_max <= current_pair_max
```

Every candidate violating that guard is rejected. Among the remaining candidates the search minimizes:

```text
1. local 2 MiB boundary-page exposure
2. local 4 KiB boundary-page exposure
3. pair_sum
4. pair_max
5. distance from the current boundary
```

The current boundary is the baseline candidate. Therefore an accepted move can never worsen the local page tuple. Because the pair-max guard is applied before ranking, an accepted move also cannot increase the affected pair's maximum worker load. Applying this step-by-step preserves the global `max_worker_cells` no-regression guarantee.

This relaxation intentionally permits a case that v5.22 rejected:

```text
page exposure improves
pair_max does not increase
pair_sum increases
```

That is legal because `pair_sum` is not the parallel critical path. The runtime records how often this actually happens rather than assuming it is useful.

The page-aware stderr record contains:

```text
objective=max_guard-page-sum-v5.23
candidate_evaluations=...
max_guard_rejections=...
page_improving_moves=...
page_tie_load_moves=...
page_improve_sum_increase_moves=...
boundary_moves=...
moved_jobs=...
penalty_2m_before=...
penalty_2m_after=...
penalty_4k_before=...
penalty_4k_after=...
max_worker_cells_before=...
max_worker_cells_after=...
build_s=...
```

Interpretation:

- `candidate_evaluations`: nearby ordered boundary candidates whose LPT loads were evaluated.
- `max_guard_rejections`: candidates rejected because their pair maximum would increase.
- `page_improving_moves`: accepted moves with a strictly better `(2 MiB, 4 KiB)` page tuple.
- `page_tie_load_moves`: accepted moves with equal page tuple but improved load secondary criteria.
- `page_improve_sum_increase_moves`: page-improving moves that increase `pair_sum` while respecting the pair-max guard. A nonzero value is direct evidence that the v5.23 relaxation used search space unavailable to the old strict objective.

The implementation checks after rebuilding all worker assignments that:

```text
max_worker_cells_after <= max_worker_cells_before
(penalty_2m_after, penalty_4k_after)
  <= (penalty_2m_before, penalty_4k_before)
```

A violation aborts rather than silently running the solver.

## Page penalty versus global page count

The page optimizer computes a local penalty for each ordered domain boundary directly from the factorized authoritative layout. For every factor block it finds the byte boundary between the left and right HIGH-mask ranges. A boundary not aligned to 2 MiB or 4 KiB contributes one page exposure at that page size. Main and blocked arrays are separate address spaces and are counted separately.

`cpu_low_domain_page_penalty_*` is the sum of per-boundary local penalties. It is the objective used by the bounded local search.

The schedule-plan probe also reports:

```text
hybrid_domain_cross_domain_pages_2m=...
hybrid_domain_cross_domain_pages_4k=...
```

Those are global unique page counts for the final assignment. With three or more domains, two distinct boundaries can theoretically reference the same VM page, so the local penalty sum and global unique count need not be identical. Use the global unique counts for final topology comparison.

This distinction motivates a possible later global-unique optimizer; v5.23 intentionally keeps the production search small and bounded.

## Exact work model

Every static schedule uses the same exact structural work estimate for a LOW occupancy group:

```text
sum over LOW positions and source factor blocks:
  HIGH rows in the group
  * (NN ops + NR ops + NL ops + LOCAL closure ops + CROSS closure ops)
```

The estimate measures recurrence cell iterations, not bytes or elapsed time. Real performance can still be limited by memory bandwidth, remote NUMA access, cache/TLB behavior, or GPU/CPU overlap.

## Preflight probe

Build once:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
```

Inspect the three domain variants:

```bash
CPU_LOW_DOMAIN_REFINE=0 CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32

CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32

CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
```

Important fields include:

```text
imbalance=...
contiguous_imbalance=...
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

The `hybrid_domain_*` prefix is retained in raw probe output for compatibility; it describes the production `domain` schedule.

These page metrics are static ownership exposures, not measured remote-memory bytes. First-touch placement, AutoNUMA, THP, caches, and GPU DMA still determine real traffic.

## Experiment separation

Three experiments answer different questions and must not be mixed.

### Ordinary refinement A/B

```bash
N=27 CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-refine-plan-ab.sh
```

and for full timing:

```bash
bash scripts/bench/ramstream32-cpu-low-domain-refine-ab.sh
```

These runners force `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0`. They measure only the load-refinement stage.

### Page-aware A/B

```bash
N=27 CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-page-plan-ab.sh
```

and for full timing:

```bash
bash scripts/bench/ramstream32-cpu-low-domain-page-ab.sh
```

Both variants force `CPU_LOW_DOMAIN_REFINE=1`; only `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0|1` changes. The runtime harness records the v5.23 search statistics from stderr in addition to wall and LOW timing.

### Standard topology/Pareto sweep

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
MAX_IMBALANCE=1.05 \
bash scripts/bench/ramstream32-cpu-low-schedule-plan-sweep.sh
```

This baseline is deliberately fixed to:

```text
CPU_LOW_DOMAIN_REFINE=1
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0
```

so an inherited shell environment cannot silently contaminate the topology frontier.

The analyzer minimizes three objectives:

```text
worker imbalance
cross-domain 4 KiB boundary pages
cross-domain 2 MiB boundary pages
```

and reports `scheme=lpt`, `scheme=contiguous`, or `scheme=domain`.

## Four-way clean timing

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

The runner rotates `dynamic`, `sticky`, `contiguous`, and `domain` with cyclic-latin-4 ordering. Keep both domain optimization controls fixed for the entire comparison.

## NUMA diagnosis

`move_pages` sampling perturbs timing and is diagnostic only:

```bash
CPU_LOW_SCHEDULE=domain \
CPU_LOW_DOMAIN_SIZE=32 \
CPU_LOW_DOMAIN_REFINE=1 \
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
bash scripts/bench/ramstream32-numa-sample.sh

CPU_LOW_SCHEDULE=domain \
CPU_LOW_DOMAIN_SIZE=32 \
CPU_LOW_DOMAIN_REFINE=1 \
CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 \
bash scripts/bench/ramstream32-numa-sample.sh
```

Compare row1/final node histograms, clean no-sampling wall time, `cpu_low_wall_s`, H2D/D2H time, and the static global unique page counts before considering `mbind`, interleave, THP changes, or other memory policy.

## Benchmark provenance

The following are benchmark conditions:

```text
CPU_LOW_SCHEDULE
CPU_LOW_DOMAIN_SIZE
CPU_LOW_DOMAIN_REFINE
CPU_LOW_DOMAIN_PAGE_TIEBREAK
```

HIGH threshold sweeps, HIGH policy A/B, stream-weight calibration, LOW scheduling comparisons, and NUMA diagnostics propagate and record them. Changing any one invalidates a direct timing or calibration comparison.
