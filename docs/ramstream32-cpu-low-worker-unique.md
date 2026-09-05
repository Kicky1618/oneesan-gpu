# RAMstream32 CPU LOW exact-unique worker coalescing

v5.27 is a research-only continuation of the v5.25/v5.26 intra-domain worker-locality work.

The earlier stages are:

```text
v5.25  whole-domain contiguous conversion under the existing LPT cap
v5.26  one-job neighbour coalescing using sums of per-boundary page counts
v5.27  one-job neighbour coalescing using the exact global unique page set
```

## Why v5.27 exists

v5.26 can reduce the sum of page cardinalities across worker boundaries, but two different boundaries can expose the same VM page. Therefore a smaller local sum does not prove that the final unique cross-worker page set shrank.

v5.27 keeps a reference count for every page ID exposed by every current worker transition. Main and blocked authoritative arrays are tagged as distinct address spaces. Fixed transitions at NUMA-domain boundaries are included in the reference counts even though v5.27 never moves those boundaries.

A one-job move can affect only the transition on its left and the transition on its right. v5.27 computes the exact reference-count delta for those at most two boundary signatures. Pages still referenced by any other worker boundary remain in the unique set.

The objective is lexicographic:

```text
1. exact global unique 2 MiB cross-worker pages
2. exact global unique 4 KiB cross-worker pages
3. worker owner-transition count
```

A move is accepted only if this tuple decreases.

## Load and ownership safety

The legal move set is the same as v5.26. One ordered HIGH-mask job may move only to the worker owning its immediate left or right neighbour.

For each domain, the pre-pass maximum worker load is frozen as the cap:

```text
M = max worker exact-cell load in that domain before v5.27
```

A candidate is legal only when:

```text
destination is in the same domain
destination_load + job_cells <= M
```

Therefore:

```text
NUMA-domain ownership never changes
global max worker load cannot increase
```

## Incremental-accounting validation

After the greedy search, v5.27 rebuilds the complete 2 MiB and 4 KiB page-reference maps from the final owner vector. The rebuilt maps and transition count must exactly equal the incrementally maintained state.

The plan probe also computes cross-worker page metrics with its older independent storage scan and requires those values to equal v5.27's internal before/after unique counts.

The provenance string is:

```text
objective=global-unique-neighbor-coalesce-v5.27-plan
```

## Independent v5.26 and v5.27 branches

The structural experiment does not run v5.27 after v5.26. Instead both begin from the same v5.25 assignment:

```text
baseline = refined domain + v5.23
locality = baseline + v5.25

local branch:
  locality -> v5.26

exact branch:
  locality -> v5.27
```

This matters because both searches are greedy. A locally attractive v5.26 move can enter a different basin. It is possible in principle for the final v5.26 state to have a better exact unique tuple than the direct v5.27 greedy state even though v5.27 never accepts an exact-objective regression.

The sweep reports this case as:

```text
v5.26_better_basin
```

Such a result would motivate multi-start or controlled two-stage exact selection.

## Structural probe

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-domain-worker-locality-plan.sh
```

Run:

```bash
./build/ramstream32_cpu_low_domain_worker_locality_plan_n27 27 64 32
```

The probe compares four variants:

```text
baseline   = refined domain + v5.23
locality   = baseline + v5.25
coalesced  = locality + v5.26
unique     = locality + v5.27
```

Important v5.27 fields include:

```text
unique_noncontiguous_domains_before
unique_improved_domains
unique_accepted_moves
unique_cap_rejections
unique_page_improving_moves
unique_transition_only_moves
unique_internal_pages_2m_before/after
unique_internal_pages_4k_before/after
unique_max_worker_cells
unique_cross_worker_pages_2m/4k
unique_worker_owner_transitions
worker_unique_coalesce_build_s
```

Hard checks include:

```text
unique_max_worker_cells <= locality_max_worker_cells

(unique_cross_worker_pages_2m,
 unique_cross_worker_pages_4k,
 unique_worker_owner_transitions)
<=
(locality_cross_worker_pages_2m,
 locality_cross_worker_pages_4k,
 locality_worker_owner_transitions)
```

All variants must preserve identical cross-domain page and owner-transition metrics.

## n=27 topology sweep

Use:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-worker-locality-plan-sweep.sh
```

v5.27 relative to v5.25 is classified as:

```text
v5.27_unique_page_improvement
v5.27_transition_only_improvement
v5.27_no_change
```

There is no v5.27 unique-page tradeoff class because the exact objective forbids that regression.

The final v5.26 and v5.27 states are also compared as:

```text
v5.27_better_than_v5.26
v5.26_better_basin
v5.26_v5.27_tie
```

## Exactness validation

The W=10 selftest creates two independent schedule branches from the same initial domain assignment:

```text
A: refined -> v5.23 -> v5.25 -> v5.26
B: refined -> v5.23 -> v5.25 -> v5.27
```

Each branch executes two consecutive LOW generations. Every main and blocked state is compared with the independent reference recurrence after both generations.

Synthetic tests also cover non-contiguous owner detection and exact page-reference delta arithmetic.

## Promotion gate

v5.27 is not wired into the production solver yet.

A useful promotion result requires:

```text
unique_accepted_moves > 0
unique_max_worker_cells <= locality_max_worker_cells
(unique_cross_worker_pages_2m, unique_cross_worker_pages_4k)
  < (locality_cross_worker_pages_2m, locality_cross_worker_pages_4k)
```

for relevant n=27 host topologies, with acceptable schedule-build cost.

If repeated `v5.26_better_basin` cases appear, the next algorithm should be a multi-start exact selector rather than immediately promoting the direct v5.27 greedy path.

After structural preflight, a same-binary runtime A/B is still required. Clean `cpu_low_wall_s` and whole-solver wall time remain the deciding measurements.
