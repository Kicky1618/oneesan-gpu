# RAMstream32 CPU LOW page-aware experiment protocol

The core hybrid backend banner remains v5.22, while the page-aware second-stage scheduler uses the v5.23 relaxed objective.

Production first builds the ordinary refined-domain assignment. Page-aware mode then searches nearby ordered domain boundaries subject to the hard guard:

```text
candidate_pair_max <= current_pair_max
```

Among eligible candidates it minimizes local `(2 MiB pages, 4 KiB pages)`, then `pair_sum`, then `pair_max`. This preserves the global maximum-worker no-regression guarantee while exposing more locality candidates than the old strict `(pair_max,pair_sum)` equality rule.

## Keep the experiments separated

The standard topology/Pareto sweep is intentionally pinned to:

```text
CPU_LOW_DOMAIN_REFINE=1
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0
```

Run it with:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-schedule-plan-sweep.sh
```

That sweep remains the stable refined-domain baseline and cannot inherit page-aware mode from the parent shell.

The load-refinement experiment is also explicitly page-free. Both:

```text
scripts/bench/ramstream32-cpu-low-domain-refine-ab.sh
scripts/bench/ramstream32-cpu-low-domain-refine-plan-ab.sh
```

force `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0`. Therefore refine=0 versus refine=1 measures only load-balancing boundary refinement.

## Page-aware preflight

Use:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-page-plan-ab.sh
```

Both variants keep `CPU_LOW_DOMAIN_REFINE=1`; only `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0|1` changes.

The structural contracts are:

```text
page1 max_worker_cells <= page0 max_worker_cells
local (2 MiB, 4 KiB) page penalty does not regress
```

The plan probe also compares global unique cross-domain 2 MiB/4 KiB page counts. Use those global unique counts for final structural evaluation across several domain boundaries; the optimizer's local boundary penalties can share VM pages and therefore need not equal the global unique count.

## v5.23 search diagnostics

When page-aware mode runs, stderr contains:

```text
objective=max_guard-page-sum-v5.23
candidate_evaluations=...
max_guard_rejections=...
page_improving_moves=...
page_tie_load_moves=...
page_improve_sum_increase_moves=...
max_worker_cells_before=...
max_worker_cells_after=...
```

`page_improve_sum_increase_moves` is particularly useful. A nonzero value means v5.23 accepted a locality-improving boundary whose `pair_sum` increased while `pair_max` did not. The previous strict implementation rejected exactly this class of candidate.

A high `max_guard_rejections / candidate_evaluations` ratio means the remaining search space is constrained mainly by load balance. A low ratio with few page moves means page alignment, rather than the load guard, is probably the limiting factor.

## Page-aware runtime A/B

After a promising preflight result:

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
bash scripts/bench/ramstream32-cpu-low-domain-page-ab.sh
```

The harness fixes:

```text
CPU_LOW_SCHEDULE=domain
CPU_LOW_DOMAIN_REFINE=1
RAMSTREAM_NUMA_SAMPLE_MIB=0
```

and alternates only page=0/page=1 order. It verifies identical residues and stdout/stderr provenance.

The TSV now records both performance and search behavior:

```text
wall_s
cpu_low_wall_s
page_build_s
page_candidate_evaluations
page_max_guard_rejections
page_improving_moves
page_tie_load_moves
page_improve_sum_increase_moves
page_boundary_moves
page_moved_jobs
page_max_worker_cells_before/after
page_penalty_2m_before/after
page_penalty_4k_before/after
```

A useful result must improve clean `cpu_low_wall_s` or whole wall time enough to amortize one-time schedule construction. Static page improvement alone is not sufficient.

## How to interpret v5.23 versus v5.22

The most informative cases are:

```text
page_improve_sum_increase_moves > 0
```

The relaxed criterion definitely used candidates unavailable to v5.22.

```text
page_improving_moves > 0
page_improve_sum_increase_moves = 0
```

v5.23 improved locality, but the observed accepted moves might also have satisfied the former strict sum condition. A strict-emulation A/B would be required to attribute the gain specifically to the relaxation.

```text
page_improving_moves = 0
```

The page stage made no locality move for that topology. Inspect candidate and rejection counts before spending time on a full residue.

## NUMA diagnosis

If runtime changes materially, compare actual page placement with:

```bash
CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 bash scripts/bench/ramstream32-numa-sample.sh

CPU_LOW_SCHEDULE=domain CPU_LOW_DOMAIN_SIZE=32 CPU_LOW_DOMAIN_REFINE=1 \
CPU_LOW_DOMAIN_PAGE_TIEBREAK=1 bash scripts/bench/ramstream32-numa-sample.sh
```

Keep HIGH policy, affinity, overlap, THP state, and sample spacing fixed. Interpret together:

- global unique static cross-domain 2 MiB/4 KiB page counts;
- v5.23 candidate/rejection/move statistics;
- `move_pages` row1/final node histograms and drift;
- clean no-sampling LOW and whole-solver wall time.

This separation prevents LOW scheduler changes from being absorbed accidentally into CPU HIGH throughput or stream-weight calibration.
