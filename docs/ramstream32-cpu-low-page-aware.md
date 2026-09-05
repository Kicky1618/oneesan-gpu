# RAMstream32 CPU LOW page-aware experiment protocol

The core hybrid backend banner remains v5.22, while the production page-aware second-stage scheduler uses the v5.23 relaxed objective.

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

## v5.24 global-unique research optimizer

For three or more domains, the production v5.23 objective is still local: it minimizes the sum of per-boundary page penalties. Two different domain boundaries can map to the same VM page, so minimizing that sum is not always identical to minimizing the final global unique cross-domain page count.

A research-only v5.24 planner now evaluates the exact union of page IDs across every domain boundary. Main and blocked authoritative arrays are tagged as distinct address spaces before the union is counted.

It is deliberately not wired into production. The experiment runs:

```text
ordinary refined domain
  -> local v5.23 page pass
  -> global-unique v5.24 page pass
```

The v5.24 stage keeps the same pair-max safety guard and ranks candidates by:

```text
1. global unique 2 MiB cross-domain pages
2. global unique 4 KiB cross-domain pages
3. pair_sum
4. pair_max
5. boundary displacement
```

Build and sweep it with:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-global-page-plan.sh

N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-global-page-plan-sweep.sh
```

The probe compares three assignments:

```text
refined: ordinary load-refined domain schedule
local:   refined + production v5.23 local page pass
global:  local + research v5.24 global-unique pass
```

Hard checks are:

```text
local_max_worker_cells <= refined_max_worker_cells
global_max_worker_cells <= local_max_worker_cells
(global_pages_2m, global_pages_4k)
  <= (local_global_pages_2m, local_global_pages_4k)
```

The sweep reports `classification=global_beats_local` only when the exact global unique tuple improves beyond v5.23. `global_ties_local` means the extra optimization stage did not buy structural locality for that topology.

This is the current promotion gate: v5.24 should not be added to production unless n=27 topology sweeps show repeated `global_beats_local` cases with modest schedule-build cost. Runtime timing would still be required after that, because fewer shared VM pages do not automatically imply fewer remote-memory transactions.

## Page-aware runtime A/B

After a promising v5.23 preflight result:

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

The TSV records both performance and search behavior:

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

## How to interpret v5.23 versus the old strict objective

The most informative case is:

```text
page_improve_sum_increase_moves > 0
```

The relaxed criterion definitely used candidates unavailable to the old strict objective.

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
