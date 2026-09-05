# RAMstream32 CPU LOW worker multistart exact selection

v5.28 is a research-only selector built on the v5.25-v5.27 intra-domain locality work. The exact v5.27 objective now uses the v5.29 flat page-delta implementation during candidate evaluation.

The motivation is a greedy-search basin problem. v5.27 optimizes the exact global unique cross-worker page tuple, but greedily accepting only immediate exact improvements can still get stuck. v5.26 follows a different local page-cardinality objective and can take a different route through the ownership space.

v5.28 therefore evaluates two branches from the same deterministic v5.25 schedule:

```text
direct:
  v5.25 -> v5.27

hybrid:
  v5.25 -> v5.26 -> v5.27
```

The selector chooses the smaller exact tuple:

```text
(global unique 2 MiB cross-worker pages,
 global unique 4 KiB cross-worker pages,
 weighted StorageBlock owner transitions)
```

If the exact tuples tie, it chooses the branch with the smaller maximum worker load. If that also ties, `direct` wins as the stable deterministic fallback.

The provenance string is:

```text
objective=multistart-global-unique-worker-v5.28-plan
```

## Structural dominance argument

Let `P` be the v5.25 parent exact tuple, `L` the raw v5.26 tuple, `D` the direct v5.27 result, and `H` the hybrid result after v5.26 followed by v5.27.

v5.27 never accepts an exact-objective regression. Therefore:

```text
D <= P
H <= L
```

v5.28 selects:

```text
S = min(D, H)
```

so:

```text
S <= P
S <= D
S <= H
S <= L
```

The selected exact tuple therefore structurally dominates the v5.25 parent, raw v5.26, and direct v5.27 result. This does not claim runtime dominance; only the static exact page/transition objective is covered.

Both branches inherit the same hard load rule from v5.26/v5.27: jobs stay inside their original NUMA domain and a destination worker cannot exceed the branch's pre-pass domain maximum. Thus neither branch can increase the v5.25 maximum worker load. The selector's exact-tuple tie-break may additionally choose a branch with a smaller maximum worker load.

## Why the hybrid branch can matter

A `v5.26_better_basin` result in the v5.27 sweep means the local v5.26 greedy path reached a state whose exact tuple is better than the direct v5.27 final state.

Running v5.27 after that v5.26 state is safe because exact cleanup can only keep or improve its exact tuple. Therefore the hybrid branch turns a diagnostic `v5.26_better_basin` case into a branch that can be selected automatically without sacrificing the exact objective.

## Plan probe

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-domain-worker-multistart-plan.sh
```

Run one topology:

```bash
./build/ramstream32_cpu_low_domain_worker_multistart_plan_n27 27 64 32
```

Important output fields are:

```text
parent_pages_2m/4k
parent_transitions

direct_pages_2m/4k
direct_transitions
direct_max_worker_cells

raw_v526_pages_2m/4k
raw_v526_transitions
hybrid_raw_v526_max_worker_cells

hybrid_pages_2m/4k
hybrid_transitions
hybrid_max_worker_cells

selected_source=direct|hybrid
selected_pages_2m/4k
selected_transitions
selected_max_worker_cells
```

The probe hard-checks all structural dominance relations and verifies that the copied selected schedule is exactly the winning branch's worker-job assignment and worker-cell vector.

The exact branch stderr also carries:

```text
objective=global-unique-neighbor-coalesce-v5.27-plan
implementation=flat-page-delta-v5.29
```

so the plan result records both objective semantics and candidate-evaluation implementation.

## n=27 topology sweep

Use:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-worker-multistart-plan-sweep.sh
```

The sweep classifies the selected result relative to the v5.25 parent as:

```text
multistart_improvement
multistart_no_change
```

and reports which exact branch basin wins:

```text
hybrid_wins
direct_wins
exact_tie
```

`hybrid_wins` is direct evidence that the exact-objective greedy search has multiple useful basins and that the v5.26 detour adds structural value even though v5.26 itself is not the final objective.

The sweep also reports the hybrid v5.27 cleanup delta relative to raw v5.26 and separate schedule-build times for the direct and hybrid paths.

## Exactness validation

`ramstream32_cpu_low_domain_worker_multistart_selftest.cu` reuses the independent W=10 reference-recurrence helpers from the existing worker-locality selftest.

It constructs three identical v5.25 schedules, then builds:

```text
direct   = v5.25 -> v5.27
hybrid   = v5.25 -> v5.26 -> v5.27
selected = exact copy of the v5.28 winner
```

The selected schedule executes two consecutive real LOW generations. Every main and blocked state is compared against the independent reference after each generation.

The test also checks that the selected exact tuple dominates parent, direct, raw v5.26, and hybrid, and that the selected maximum worker load does not exceed the v5.25 parent maximum.

`ramstream32_cpu_low_worker_flat_delta_selftest.cu` separately validates the v5.29 page-delta algebra, including opposite-sign cancellation and normalized `+2/-2` duplicate-page deltas.

## Promotion gate

v5.28 is not wired into the production solver.

The structural gate for n=27 is:

```text
selected_pages tuple <= parent, raw v5.26, direct v5.27, hybrid v5.27
selected_max_worker_cells <= parent_max_worker_cells
```

A stronger practical signal is repeated `hybrid_wins` or a useful selected page reduction on the intended two-socket host topologies.

v5.29 has already removed the per-candidate temporary `unordered_map` construction from the exact branches. The remaining cost question is measured schedule-build time. If exact planning is still material at n=27, the next targets are the persistent page-reference lookup table and the tiny flat-delta normalization itself.

After structural preflight, a same-binary runtime A/B is still mandatory. Clean `cpu_low_wall_s` and whole-solver wall time, not static page counts, decide whether the scheduling stage belongs in production.
