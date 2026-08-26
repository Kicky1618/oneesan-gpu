# RAMstream32 CPU LOW shared exact workspace

v5.31 is a research-only build-time optimization for the v5.28 multistart worker-locality search.

The v5.28 branches are:

```text
direct: v5.25 -> exact search
hybrid: v5.25 -> v5.26 -> exact search
```

The exact search needs topology-only data that is identical in both branches:

- ordered non-empty HIGH-mask jobs and exact-cell weights;
- job-index to ordered-position mapping;
- HIGH-mask next-nonempty page index;
- v5.30 dense 2 MiB/4 KiB boundary signatures;
- weighted StorageBlock transition cost for every ordered boundary.

v5.31 builds those objects once in `CpuLowWorkerExactWorkspace` and reuses the
same immutable workspace for direct and hybrid exact searches.

The exact objective and accepted schedules are unchanged.

## Workspace provenance

The shared exact coalescer logs:

```text
implementation=shared-dense-page-ref-v5.31
workspace_reuse=1
workspace_mib=...
workspace_build_s=...
branch_search_s=...
```

A workspace records the exact `jobs` and `CpuLowSparseHost` object addresses
that produced it. A branch using a different topology aborts rather than silently
reusing stale metadata.

## W=10 exactness/equivalence

`ramstream32_cpu_low_worker_shared_multistart_selftest.cu` constructs legacy and
shared versions of both direct and hybrid branches. It requires:

```text
legacy direct schedule == shared direct schedule
legacy hybrid schedule == shared hybrid schedule
legacy selector == shared selector
candidate and accepted-move traces agree
```

The shared selector winner then executes two real LOW generations, with every
main and blocked state checked against the independent reference recurrence.

## n=27 plan comparison

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-shared-multistart-plan.sh
```

Sweep:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-worker-shared-multistart-plan-sweep.sh
```

The plan compares:

```text
legacy_total_build_s
  = legacy direct exact build
  + v5.26 build
  + legacy hybrid exact build

shared_total_build_s
  = one workspace build
  + shared direct search
  + v5.26 build
  + shared hybrid search
```

and reports:

```text
workspace_build_s
workspace_mib
shared_direct_search_s
shared_v526_build_s
shared_hybrid_search_s
shared_total_build_s
shared_vs_legacy_speedup
```

Every topology also hard-checks:

```text
identical_direct_schedule=1
identical_hybrid_schedule=1
identical_selector=1
```

## Promotion gate

v5.31 remains research-only until n=27 measurements show that sharing the dense
workspace reduces total multistart construction time without excessive metadata.
The target is not merely a faster individual exact branch: the relevant metric
is `shared_total_build_s` for the complete direct+hybrid selection pipeline.

If the workspace build dominates despite reuse, keep v5.29 or v5.30 branch-local
construction. If shared search wins, the next step is to integrate one immutable
workspace into the same-binary runtime A/B before changing the production LOW
scheduler.
