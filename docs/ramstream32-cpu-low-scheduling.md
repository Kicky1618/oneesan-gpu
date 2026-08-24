# RAMstream32 CPU LOW scheduling

Backend v5.20 provides four scheduling modes for the persistent sparse LOW executor. `dynamic` remains the default. `sticky`, `contiguous`, and `domain` are opt-in static-owner modes.

## Why static scheduling is exact

The LOW window fixes the occupancy mask of the inactive HIGH positions. Each resulting `CpuLowJob` is transition-closed for the complete LOW+center window, so different fixed-HIGH occupancy groups do not write into one another. Scheduling changes only which persistent worker evaluates each closed group; it does not change the recurrence, descriptor streams, operation ordering inside a group, or authoritative addresses.

The W=10 exhaustive selftest runs dynamic, LPT-sticky, contiguous, refined-domain, and unrefined-domain pools for two consecutive LOW generations and compares every result with the exact reference recurrence after one and two LOW windows. v5.20 also has a deterministic unit case for the domain-boundary refiner: an initial `[8,8,8] | [1,1,1]` one-worker-per-domain split has pair makespan 24, and the bounded refinement moves one job to reach makespan 16 while preserving the single ordered boundary.

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

The v5.20 boundary refiner is enabled by default. It can be disabled without changing the binary:

```text
CPU_LOW_DOMAIN_REFINE=1   # default: refined domain boundaries
CPU_LOW_DOMAIN_REFINE=0   # initial outer-domain partition only
```

The parser also accepts the usual boolean spellings used by the backend (`true/false`, `yes/no`, `on/off`). Invalid values abort rather than silently selecting a mode.

Refinement OFF does not change the recurrence. It keeps the initial contiguous domain ranges produced by the outer normalized-cap partition, then performs the same exact-cell LPT assignment inside each domain. Refinement ON starts from that same assignment and only moves ordered domain boundaries when the local two-domain LPT objective improves. The W=10 selftest validates both modes for two consecutive LOW generations.

Production provenance and schedule diagnostics include:

```text
cpu_low_schedule=dynamic|sticky|contiguous|domain
cpu_low_domain_size=...
cpu_low_schedule_build_s=...
cpu_low_contiguous_optimal_cap=...
cpu_low_domain_outer_normalized_cap=...
cpu_low_domain_active_domains=...
cpu_low_domain_refined_boundaries=...
cpu_low_domain_refined_job_moves=...
```

`cpu_low_domain_normalized_cap` is retained as a compatibility alias for `cpu_low_domain_outer_normalized_cap`. Until the solver's final stdout line grows a dedicated refinement field, the domain schedule stderr line records `refine=0|1`; benchmark runners that need exact refinement provenance check that line or explicitly record the environment value.

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

### v5.20 boundary refinement

After the initial domain ranges are constructed, v5.20 can optimize the actual LPT makespan without giving up domain contiguity.

For every adjacent pair of nonempty domains it searches at most 32 occupancy jobs to each side of the current boundary. Each candidate is evaluated by running the same exact-cell LPT assignment used by production inside the two affected domains. A move is accepted only when the pair objective improves lexicographically:

```text
1. smaller max(left LPT makespan, right LPT makespan)
2. if tied, smaller left makespan + right makespan
```

Two bounded passes are used, with the second pass visiting boundaries in reverse order. Therefore an accepted move cannot increase the current maximum load of the two affected domains; all other domains are unchanged by that move. The search changes only the position of an ordered domain boundary, so each domain still owns one contiguous HIGH-mask interval.

Metrics:

```text
cpu_low_domain_refined_boundaries = number of accepted boundary moves across passes
cpu_low_domain_refined_job_moves  = sum of absolute boundary displacements in jobs
```

The radius and pass count are deliberately bounded because the schedule is built on the host before the repeated LOW generations. `cpu_low_schedule_build_s` must be checked together with runtime savings.

Thus the structural tradeoff is:

```text
sticky/LPT: best unconstrained worker balance, potentially many domain cuts
contiguous: every worker owns one ordered mask interval
 domain:    every NUMA domain owns one ordered interval, optional refined LPT inside domains
```

## Preflight balance and page-cut probe

The schedule geometry can be inspected without allocating the multi-terabyte authoritative RAM arrays:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32 --workers
```

The probe constructs the actual production sticky, contiguous, and domain pools and analyzes their cached assignments. Because the domain pool's fourth constructor argument defaults from `CPU_LOW_DOMAIN_REFINE`, the same binary can inspect either structural plan:

```bash
CPU_LOW_DOMAIN_REFINE=0 ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
CPU_LOW_DOMAIN_REFINE=1 ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
```

Important fields include:

```text
imbalance=...
cross_domain_pages_4k=...
cross_domain_pages_2m=...
contiguous_imbalance=...
contiguous_cross_domain_pages_4k=...
contiguous_cross_domain_pages_2m=...
hybrid_domain_imbalance=...
hybrid_domain_cross_domain_pages_4k=...
hybrid_domain_cross_domain_pages_2m=...
hybrid_domain_outer_normalized_cap=...
hybrid_domain_refined_boundaries=...
hybrid_domain_refined_job_moves=...
```

The raw probe retains the historical `hybrid_domain_*` prefix for compatibility; those fields describe the current production `CPU_LOW_SCHEDULE=domain` assignment.

`--domain-size 32` models worker IDs `0..31` as domain 0, `32..63` as domain 1, and so on. This only matches the hardware experiment if `CPU_LOW_CPU_LIST` is arranged in corresponding socket-local blocks.

These page counts are static ownership exposures, not measured remote-memory bytes. First-touch placement, AutoNUMA, THP, caches, and GPU DMA still determine actual traffic.

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

For all four production modes, use the four-way clean harness:

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

Over every four repeats each schedule appears once in every run position. The harness verifies identical residues, schedule provenance, and domain-size provenance; it forces `RAMSTREAM_NUMA_SAMPLE_MIB=0` and records `CPU_LOW_DOMAIN_REFINE` in metadata. Use a fixed refinement value for a four-way schedule comparison.

### Isolate refinement itself

To compare only the v5.20 boundary-refinement step, use the dedicated same-binary A/B runner:

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

Odd repeats run `refine=0 -> refine=1`; even repeats reverse that order. Both variants use the same binary, domain size, HIGH policy, affinities, worker counts, modulus, and GPU target. NUMA sampling is forced off. The harness also checks the domain scheduler's stderr `refine=0|1` provenance, requires zero reported boundary moves when refinement is disabled, and verifies identical residues.

Its final comparison reports:

```text
refine_speedup       = mean wall(refine=0) / mean wall(refine=1)
refine_low_speedup   = mean LOW wall(refine=0) / mean LOW wall(refine=1)
refine_extra_build_s = mean schedule build(refine=1) - mean schedule build(refine=0)
```

A useful refinement must recover more repeated LOW runtime than it adds to one-time schedule construction. Whole-solver wall time remains the deciding metric; an improved static LPT makespan can still lose if the workload is memory-bandwidth or NUMA-placement limited rather than tail limited.

No schedule or refinement setting is assumed faster. Dynamic can win from fine-grained balancing. Sticky can win from stable ownership with near-perfect load balance. Contiguous can win when page/address locality dominates. Domain is intended to retain most of sticky's balance while reducing cross-NUMA-domain ownership boundaries; refinement specifically attacks the remaining makespan penalty at those domain boundaries.

## NUMA diagnosis

`move_pages` sampling is diagnostic and perturbs timing, so run it separately:

```bash
CPU_LOW_SCHEDULE=dynamic    bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky     bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=contiguous bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
  bash scripts/bench/ramstream32-numa-sample.sh
```

To determine whether boundary refinement changes placement rather than just balance, repeat the domain diagnostic with `CPU_LOW_DOMAIN_REFINE=0` while holding all other conditions fixed.

Keep HIGH policy, worker counts, affinity lists, overlap mode, and sample spacing identical. Compare row1/final node histograms and node-fraction drift with clean timing before considering `mbind`, interleave, THP changes, or other memory policy.

## Benchmark provenance

The LOW schedule comparison, domain-refinement A/B, NUMA diagnostic, CPU HIGH threshold sweep, policy A/B harness, and stream calibration propagate and record the relevant LOW scheduling controls. When `domain` is used, `CPU_LOW_DOMAIN_SIZE` and `CPU_LOW_DOMAIN_REFINE` are part of the benchmark condition; changing either invalidates a direct timing or calibration comparison.
