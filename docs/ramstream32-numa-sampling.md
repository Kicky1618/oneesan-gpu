# RAMstream32 NUMA placement sampling

Hybrid RAMstream uses very large anonymous authoritative mappings and advises Linux with `MADV_HUGEPAGE`. Worker affinity alone does not determine where those pages live: anonymous memory follows first-touch placement, GPU H2D/D2H activity touches host pages as well, and AutoNUMA may later migrate pages.

Backend v5.16 added a non-mutating placement diagnostic. Set:

```bash
RAMSTREAM_NUMA_SAMPLE_MIB=64
```

and the hybrid-sparse backend sparsely samples the authoritative `main` and `block` mappings after the first complete grid row and again after the final row. Sampling uses Linux `move_pages(pid=0, nodes=nullptr, flags=0)`, which queries page location without requesting migration.

The implementation caps each array at 32,768 sampled pages. If the requested spacing would exceed that cap, the actual spacing is increased and reported. Lazy/unresident pages and syscall failures are logged rather than treated as solver failures.

Example raw output:

```text
numa_sample tag=row1 array=main bytes=... requested_spacing_mib=64 actual_spacing_mib=64 samples=30000 success=29980 N0=17000 N1=12980 E-2=20
numa_sample tag=final array=main bytes=... requested_spacing_mib=64 actual_spacing_mib=64 samples=30000 success=30000 N0=15000 N1=15000
```

`E<negative errno>=...` entries are per-page query failures. A whole-syscall failure is reported with `syscall_errno=...` and is non-fatal. Some container/security configurations can deny `move_pages`; in that case run directly on the host or with an appropriate container policy before drawing NUMA conclusions.

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
CPU_WORKERS=64 \
NUMA_SAMPLE_MIB=64 \
bash scripts/bench/ramstream32-numa-sample.sh
```

The current LOW scheduler exposes four modes:

```text
CPU_LOW_SCHEDULE=dynamic
CPU_LOW_SCHEDULE=sticky
CPU_LOW_SCHEDULE=contiguous
CPU_LOW_SCHEDULE=domain
```

`sticky` uses a one-time exact-cell LPT partition. `contiguous` keeps every worker on one ordered HIGH-mask interval. `domain` keeps only NUMA-domain ownership contiguous and restores exact-cell LPT inside each domain. `domain` additionally requires `CPU_LOW_DOMAIN_SIZE`, which must be positive and no larger than `CPU_WORKERS`.

For domain scheduling, `CPU_LOW_DOMAIN_REFINE=1` is the default and enables the bounded domain-boundary LPT refinement. `CPU_LOW_DOMAIN_REFINE=0` keeps the initial outer-domain partition. Both are exact recurrence schedules; the difference is ownership geometry and resulting timing/locality.

To isolate scheduling/locality effects, hold every other condition fixed:

```bash
CPU_LOW_SCHEDULE=dynamic    bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky     bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=contiguous bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
  bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=0 \
  bash scripts/bench/ramstream32-numa-sample.sh
```

The runner propagates and records `CPU_LOW_SCHEDULE`, `CPU_LOW_DOMAIN_SIZE`, and `CPU_LOW_DOMAIN_REFINE`. For domain mode it also checks the scheduler stderr `refine=0|1` provenance before analyzing samples, so placement results cannot silently mix refined and unrefined ownership.

For a cost-model HIGH policy, replace the threshold with:

```bash
CPU_HIGH_MAX_MIB=0 \
CPU_HIGH_GROUPS_FILE=/path/to/cpu-high.groups
```

Sampling perturbs the run through syscalls and cache/TLB effects. Do not use the sampled run's `wall_s` as the clean performance number. Repeat the same configuration with `RAMSTREAM_NUMA_SAMPLE_MIB=0` for timing.

For clean four-way LOW timing use:

```text
scripts/bench/ramstream32-cpu-low-schedule-compare.sh
```

which rotates dynamic/sticky/contiguous/domain with a cyclic-latin-4 order and forces NUMA sampling off. To isolate only the effect of domain boundary refinement, use `scripts/bench/ramstream32-cpu-low-domain-refine-ab.sh` instead.

## Static page-cut preflight

Before a multi-terabyte residue run, inspect the production static assignments without allocating authoritative RAM:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
CPU_LOW_DOMAIN_REFINE=1 \
  ./build/ramstream32_cpu_low_schedule_plan_n27 27 64 --domain-size 32
```

Repeat with `CPU_LOW_DOMAIN_REFINE=0` to see how the boundary refiner changes exact-cell imbalance and page cuts before running the large state arrays.

The probe constructs the production LPT, contiguous, and domain schedules and then measures boundary-page ownership. It reports exact-cell imbalance, cross-worker 4 KiB/2 MiB boundary-page counts, and cross-domain cuts under the supplied worker-to-domain model.

The probe retains historical `hybrid_domain_*` raw field names for the domain schedule. The topology sweep stores those values under `domain_*` columns and the Pareto analyzer reports `scheme=domain`.

These are static ownership exposures, not measured remote-memory bytes. They are useful for deciding whether domain ownership is structurally promising, but must be paired with `move_pages` measurements and clean timing.

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

On the same host, compare at least:

1. dynamic LOW scheduling with default affinity;
2. explicit HIGH/LOW affinity with dynamic scheduling;
3. sticky with the same lists;
4. contiguous with the same lists;
5. refined domain with a hardware-matching `CPU_LOW_DOMAIN_SIZE`;
6. unrefined domain with the same domain size;
7. same-socket and split-socket HIGH/LOW layouts;
8. overlap 0 and overlap 1.

Compare node histograms together with `cpu_high_wall_s`, `cpu_low_wall_s`, H2D/D2H time, static page-cut exposure, and clean no-sampling wall time. Stable scheduling is useful only if locality gains outweigh static-load imbalance.

The purpose is to determine whether the workload is actually remote-memory limited before experimenting with interleave, 4 KiB pages, THP changes, or `mbind`.

Per-group `mbind` is intentionally not implemented yet. Factorized group slices can share boundary pages, especially with 2 MiB huge pages. LOW static page-cut analysis, measured page histograms, and `analyze_cpu_high_numa.py` HIGH page exposure should be considered together before imposing a memory policy.
