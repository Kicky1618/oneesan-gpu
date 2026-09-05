# RAMstream32 NUMA placement sampling

Hybrid RAMstream uses very large anonymous authoritative mappings and advises Linux with `MADV_HUGEPAGE`. Worker affinity alone does not determine where those pages live: anonymous memory follows first-touch placement, GPU H2D/D2H activity touches host pages as well, and AutoNUMA may later migrate pages.

Backend v5.16 added a non-mutating placement diagnostic. Set:

```bash
RAMSTREAM_NUMA_SAMPLE_MIB=64
```

and the hybrid-sparse backend sparsely samples the authoritative `main` and `block` mappings after the first complete grid row and again after the final row. Sampling uses Linux `move_pages(pid=0, nodes=nullptr, flags=0)`, which queries page location without requesting migration.

The implementation caps each array at 32,768 sampled pages. If the requested spacing would exceed that cap, the actual spacing is increased and reported. Lazy/unresident pages and syscall failures are logged rather than treated as solver failures.

## One-command diagnostic

Use the dedicated runner instead of a performance benchmark:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-95' \
CPU_LOW_SCHEDULE=domain \
CPU_LOW_DOMAIN_SIZE=32 \
CPU_LOW_DOMAIN_REFINE=1 \
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
CPU_WORKERS=64 \
NUMA_SAMPLE_MIB=64 \
bash scripts/bench/ramstream32-numa-sample.sh
```

The current LOW scheduler exposes four modes: `dynamic`, `sticky`, `contiguous`, and `domain`. Domain scheduling additionally requires `CPU_LOW_DOMAIN_SIZE`.

`CPU_LOW_DOMAIN_REFINE=1` is the default load-balancing boundary refinement. v5.22 adds `CPU_LOW_DOMAIN_PAGE_TIEBREAK=1`, an optional second stage that keeps the refined LPT load objective unchanged and uses 2 MiB then 4 KiB boundary-page exposure only as a tie-break. Page mode requires refined domain scheduling.

To isolate scheduling/locality effects, hold every other condition fixed:

```bash
CPU_LOW_SCHEDULE=dynamic bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=contiguous bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
  CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
  CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 bash scripts/bench/ramstream32-numa-sample.sh
```

The runner propagates and records `CPU_LOW_SCHEDULE`, `CPU_LOW_DOMAIN_SIZE`, `CPU_LOW_DOMAIN_REFINE`, and `CPU_LOW_DOMAIN_PAGE_TIEBREAK`. It checks the final solver provenance. For domain mode it also verifies stderr `refine=...`, and page-aware mode requires the `cpu_low_domain_page_tiebreak` diagnostic line. The output basename includes the page mode so two sampled placements are not accidentally overwritten or confused.

For a cost-model HIGH policy, replace the threshold with:

```bash
CPU_HIGH_MAX_MIB=0 \
CPU_HIGH_GROUPS_FILE=/path/to/cpu-high.groups
```

Sampling perturbs the run through syscalls and cache/TLB effects. Do not use the sampled run's `wall_s` as the clean performance number. Repeat the same configuration with `RAMSTREAM_NUMA_SAMPLE_MIB=0` for timing.

For clean scheduling measurements use:

```text
scripts/bench/ramstream32-cpu-low-schedule-compare.sh
scripts/bench/ramstream32-cpu-low-domain-refine-ab.sh
scripts/bench/ramstream32-cpu-low-domain-page-ab.sh
```

The dedicated page A/B keeps `CPU_LOW_DOMAIN_REFINE=1` in both variants and changes only `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0|1`.

## Static page-cut preflight

Before a multi-terabyte residue run, inspect the production static assignments without allocating authoritative RAM:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
```

For several worker/socket shapes use:

```text
scripts/bench/ramstream32-cpu-low-domain-page-plan-ab.sh
```

The probe constructs the actual production assignment and measures global unique cross-worker and cross-domain 4 KiB/2 MiB boundary pages. These global counts are distinct from the page refiner's internal sum of local per-boundary penalties; use the global probe fields when deciding whether page-aware scheduling structurally improved the final multi-domain layout.

These are static ownership exposures, not measured remote-memory bytes. They must be paired with `move_pages` measurements and clean timing. A lower static boundary count can still lose if first-touch, THP, AutoNUMA, caches, or GPU DMA dominate actual placement.

## Analyzer

Analyze an existing solver stderr log with:

```bash
python3 scripts/tools/analyze_ramstream_numa_samples.py run.stderr.txt
```

For each `(tag,array)` it reports successful sampled-page fraction, node histogram/fractions, dominant node, requested/actual sample spacing, and whole-syscall errno if present.

When both `row1` and `final` samples exist, it also reports:

```text
node_fraction_l1 = sum_node |fraction_row1(node) - fraction_final(node)|
```

for `main` and `block`. A substantial drift indicates that placement changed during the calculation, for example through AutoNUMA migration or a different set of pages becoming resident.

`--max-unplaced-fraction X` can turn excessive query/unresident failures into a nonzero analyzer exit code for controlled experiments.

## How to use the result

On the same host, compare at least refined-domain page=0 and page=1 with identical worker affinity, HIGH policy, overlap setting, THP state, and sample spacing. Read three layers together:

1. preflight `hybrid_domain_cross_domain_pages_2m/4k` for static ownership geometry;
2. sampled node histograms and row1→final drift for actual placement;
3. clean `cpu_low_wall_s` and whole-solver wall time with NUMA sampling disabled.

The purpose is to determine whether page-aware boundary ownership reduces real remote-memory pressure before experimenting with interleave, 4 KiB pages, THP changes, or `mbind`.

Per-group `mbind` remains intentionally unimplemented. Factorized group slices can share boundary pages, especially with 2 MiB huge pages. LOW static page-cut analysis, measured page histograms, and HIGH `analyze_cpu_high_numa.py` exposure should be considered together before imposing a memory policy.
