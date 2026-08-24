# RAMstream32 NUMA placement sampling

Hybrid RAMstream uses very large anonymous authoritative mappings and advises Linux with `MADV_HUGEPAGE`. Worker affinity alone does not determine where those pages live: anonymous memory follows first-touch placement, GPU H2D/D2H activity touches host pages as well, and AutoNUMA may later migrate pages.

Backend v5.16 adds a non-mutating diagnostic. Set:

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

`E<negative errno>=...` entries are per-page query failures. A whole-syscall failure is reported with `syscall_errno=...` and is non-fatal. Some container/security configurations can deny `move_pages`; in that case run the solver directly on the host or with an appropriate container policy before drawing NUMA conclusions.

## One-command diagnostic

Use the dedicated runner instead of a performance benchmark:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-63' \
CPU_LOW_SCHEDULE=dynamic \
NUMA_SAMPLE_MIB=64 \
bash scripts/bench/ramstream32-numa-sample.sh
```

Backend v5.17 also supports `CPU_LOW_SCHEDULE=sticky`, which keeps each fixed-HIGH occupancy group on the same persistent LOW worker after a one-time exact-cell LPT partition. To isolate scheduling/locality effects, run the same diagnostic twice with only the schedule changed:

```bash
CPU_LOW_SCHEDULE=dynamic bash scripts/bench/ramstream32-numa-sample.sh
CPU_LOW_SCHEDULE=sticky  bash scripts/bench/ramstream32-numa-sample.sh
```

For a cost-model group policy, replace the threshold with:

```bash
CPU_HIGH_MAX_MIB=0 \
CPU_HIGH_GROUPS_FILE=/path/to/cpu-high.groups
```

The runner saves stdout, stderr, environment metadata, binary hash, optional groups-file hash, LOW schedule mode, and an analyzed placement summary.

Sampling perturbs the run through syscalls and cache/TLB effects. Do not use the sampled run's `wall_s` as the clean performance number; repeat the same configuration with `RAMSTREAM_NUMA_SAMPLE_MIB=0` for timing. For clean dynamic/sticky timing, use `scripts/bench/ramstream32-cpu-low-schedule-ab.sh`, which forces NUMA sampling off and alternates run order.

## Analyzer

Analyze an existing solver stderr log with:

```bash
python3 scripts/tools/analyze_ramstream_numa_samples.py run.stderr.txt
```

For each `(tag,array)` it reports:

- successful sampled-page fraction;
- node histogram and fractions;
- dominant node and fraction;
- requested and actual sample spacing;
- whole-syscall errno if present.

When both `row1` and `final` samples exist, it also reports

```text
node_fraction_l1 = sum_node |fraction_row1(node) - fraction_final(node)|
```

for `main` and `block`. A substantial drift indicates that placement changed during the calculation, for example through AutoNUMA migration or a different set of pages becoming resident.

`--max-unplaced-fraction X` can turn excessive query/unresident failures into a nonzero analyzer exit code for controlled experiments.

## How to use the result

Run at least these configurations on the same host before adding a memory policy:

1. dynamic LOW scheduling with default affinity;
2. dynamic scheduling with explicit `CPU_HIGH_CPU_LIST` and `CPU_LOW_CPU_LIST`;
3. sticky scheduling with the same explicit lists;
4. same-socket and split-socket HIGH/LOW CPU-list layouts;
5. overlap 0 and overlap 1.

Compare node histograms together with `cpu_high_wall_s`, `cpu_low_wall_s`, H2D/D2H time, and clean no-sampling wall time. Sticky scheduling is useful only if stable group ownership outweighs any loss of dynamic load balancing, so placement and timing must be considered together.

The purpose is to determine whether the workload is actually remote-memory limited before experimenting with interleave, 4 KiB pages, THP changes, or `mbind`.

Per-group `mbind` is intentionally not implemented yet. Factorized group slices can share boundary pages, especially with 2 MiB huge pages, and the existing `analyze_cpu_high_numa.py` page-exposure analysis should be considered together with these measured node distributions.
