# RAMstream32 CPU LOW page-aware experiment protocol

Backend v5.22 adds `CPU_LOW_DOMAIN_PAGE_TIEBREAK=1` as a second-stage locality optimization for refined domain scheduling. It never replaces the ordinary load refinement: production first builds the refined domain assignment, then page-aware mode is allowed to move only boundaries whose affected two-domain LPT load tuple is exactly unchanged.

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

That sweep remains the stable refined-domain baseline. It must not inherit page-aware mode from the parent shell.

The load-refinement experiment is also explicitly page-free. Both:

```text
scripts/bench/ramstream32-cpu-low-domain-refine-ab.sh
scripts/bench/ramstream32-cpu-low-domain-refine-plan-ab.sh
```

force `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0`. Therefore refine=0 versus refine=1 measures only the v5.21 load-balancing boundary refinement.

## Page-aware preflight

Use the dedicated structural A/B:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-page-plan-ab.sh
```

Both variants keep `CPU_LOW_DOMAIN_REFINE=1`; only `CPU_LOW_DOMAIN_PAGE_TIEBREAK=0|1` changes.

The runner checks two different locality notions:

1. the page refiner's local lexicographic boundary penalty `(2 MiB, 4 KiB)` must not regress;
2. the final schedule-plan probe's global unique cross-domain 2 MiB/4 KiB page counts are compared separately.

The second metric is the one to use for final structural evaluation across several domain boundaries. Local boundary penalties can share VM pages and therefore need not equal the global unique count.

The page-aware assignment must also satisfy:

```text
page1 max_worker_cells <= page0 max_worker_cells
```

The current implementation should normally preserve the refined load objective exactly; the nonincrease check is retained as a hard safety contract.

## Page-aware runtime A/B

After a promising preflight result, run:

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

and alternates only page=0/page=1 order. It verifies identical residues and stdout/stderr provenance, and reports whole-solver wall time, LOW wall time, page schedule build time, accepted page moves, and the page penalty before/after.

A useful page-aware result should improve clean `cpu_low_wall_s` or whole wall time enough to amortize its one-time schedule-build cost. Lower static page counts alone are not sufficient evidence.

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
- `move_pages` row1/final node histograms and drift;
- clean no-sampling LOW and whole-solver wall time.

This separation prevents LOW scheduler changes from being absorbed accidentally into CPU HIGH throughput or stream-weight calibration.
