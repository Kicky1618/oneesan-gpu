# RAMstream32 CPU LOW scheduling

Backend v5.19 provides four scheduling modes for the persistent sparse LOW executor. `dynamic` remains the default. `sticky`, `contiguous`, and `domain` are opt-in static-owner modes.

## Why static scheduling is exact

The LOW window fixes the occupancy mask of the inactive HIGH positions. Each resulting `CpuLowJob` is transition-closed for the complete LOW+center window, so different fixed-HIGH occupancy groups do not write into one another. Scheduling changes only which persistent worker evaluates each closed group; it does not change the recurrence, descriptor streams, operation ordering inside a group, or authoritative addresses.

The W=10 exhaustive selftest runs dynamic, LPT-sticky, contiguous, and domain pools for two consecutive LOW generations and compares each result with the exact reference recurrence after one and two LOW windows.

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

Domain ownership is contiguous in numeric HIGH-mask order, but jobs inside each domain are redistributed with exact-cell LPT. This is intended to preserve most of LPT's worker balance while sharply reducing cross-domain mask boundaries. `CPU_LOW_DOMAIN_SIZE` must be positive and no larger than `CPU_WORKERS`.

Production provenance includes:

```text
cpu_low_schedule=dynamic|sticky|contiguous|domain
cpu_low_domain_size=...
cpu_low_schedule_build_s=...
cpu_low_contiguous_optimal_cap=...
cpu_low_domain_normalized_cap=...
cpu_low_domain_active_domains=...
```

The domain fields are zero when the selected schedule does not use them.

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

### Domain

For a domain containing `k` workers, the outer ordered partition gives it a contiguous mask range with a normalized capacity derived from `k * cap`. Production binary-searches the minimum feasible normalized per-worker cap across domains. Jobs inside each resulting domain range are then assigned to that domain's workers by exact-cell LPT.

Thus the structural tradeoff is:

```text
sticky/LPT: best unconstrained worker balance, potentially many domain cuts
contiguous: every worker owns one ordered mask interval
 domain:    every NUMA domain owns one ordered interval, LPT inside the domain
```

## Preflight balance and page-cut probe

The schedule geometry can be inspected without allocating the multi-terabyte authoritative RAM arrays:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32 --workers
```

The probe constructs the actual production sticky, contiguous, and domain pools and analyzes their cached assignments. It reports exact-cell imbalance together with 4 KiB and 2 MiB mask-boundary page exposure.

Important fields include:

```text
imbalance=...
cross_worker_pages_4k=...
cross_worker_pages_2m=...
cross_domain_pages_4k=...
cross_domain_pages_2m=...
contiguous_imbalance=...
contiguous_cross_domain_pages_4k=...
contiguous_cross_domain_pages_2m=...
hybrid_domain_imbalance=...
hybrid_domain_cross_domain_pages_4k=...
hybrid_domain_cross_domain_pages_2m=...
```

The probe retains the historical `hybrid_domain_*` field names for compatibility; those fields now describe the production `CPU_LOW_SCHEDULE=domain` assignment.

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

Each entry is `workers:domain-size`. The persisted TSV uses the production terminology:

```text
lpt_*
contiguous_*
domain_*
```

The analyzer:

```text
scripts/tools/analyze_cpu_low_schedule_plan_sweep.py
```

computes the Pareto frontier over three objectives:

```text
minimize worker imbalance
minimize cross-domain 4 KiB boundary pages
minimize cross-domain 2 MiB boundary pages
```

It reports `scheme=lpt`, `scheme=contiguous`, or `scheme=domain`. Legacy TSV files with `hybrid_*` columns and `--scheme hybrid` remain accepted as aliases for `domain`.

A candidate is omitted only when another candidate is no worse in all three objectives and strictly better in at least one. This avoids inventing an arbitrary scalar weight between load balance and NUMA exposure.

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

Over every four repeats each schedule appears once in every run position. The harness verifies identical residues, schedule provenance, and domain-size provenance; it forces `RAMSTREAM_NUMA_SAMPLE_MIB=0` and records whole-solver and LOW-only timings separately.

No schedule is assumed faster. Dynamic can win from fine-grained balancing. Sticky can win from stable ownership with near-perfect load balance. Contiguous can win when page/address locality dominates. Domain is intended to retain most of sticky's load balance while reducing cross-NUMA-domain ownership boundaries.

## NUMA diagnosis

`move_pages` sampling is diagnostic and perturbs timing, so run it separately:

```bash
CPU_LOW_SCHEDULE=dynamic    bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky     bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=contiguous bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 \
  bash scripts/bench/ramstream32-numa-sample.sh
```

Keep HIGH policy, worker counts, affinity lists, overlap mode, and sample spacing identical. Compare row1/final node histograms and node-fraction drift with clean timing before considering `mbind`, interleave, THP changes, or other memory policy.

## Benchmark provenance

The LOW schedule comparison, NUMA diagnostic, CPU HIGH threshold sweep, policy A/B harness, and stream calibration all propagate and record `CPU_LOW_SCHEDULE`. When `domain` is used they also propagate and record `CPU_LOW_DOMAIN_SIZE`, so mixed CPU/GPU measurements retain the topology condition that produced them.
