# RAMstream32 CPU LOW intra-domain worker locality

This research thread now has two intra-domain stages:

```text
v5.25  whole-domain contiguous conversion under the existing LPT cap
v5.26  cap-safe neighbour coalescing for non-contiguous ownership left behind
```

The production `domain` schedule gives each modeled NUMA domain one contiguous HIGH-occupancy-mask interval, then uses exact-cell LPT inside that domain. LPT is good for load balance, but adjacent occupancy masks can be scattered across many workers. That can create many cross-worker ownership boundaries inside one socket even when cross-domain ownership is already well controlled.

## v5.25: whole-domain contiguous conversion

For one domain, let

```text
M = maximum exact-cell load of any worker in the current LPT assignment
```

The domain's ordered mask jobs are greedily partitioned into contiguous segments with per-segment work at most `M`. If at most `nworkers` segments are required, the domain is converted to contiguous worker ownership. Extra splits can be added without violating the cap.

If more than `nworkers` segments are required at cap `M`, the domain falls back to its original LPT assignment exactly.

Consequently:

```text
each converted domain max <= previous domain LPT max
each fallback domain is unchanged
global max worker load cannot increase
```

The stage does not move NUMA-domain boundaries.

## v5.26: neighbour page coalescing under the same cap

A v5.25 fallback domain can still contain an owner sequence such as:

```text
worker 0, worker 3, worker 0, worker 7, ...
```

Full contiguous conversion may be impossible under the LPT cap, but some individual boundary jobs can still move safely.

v5.26 processes only non-contiguous domain ownership. For one ordered HIGH-mask job it considers at most two destinations: the worker owning the immediate left neighbour and the worker owning the immediate right neighbour.

A candidate is legal only when:

```text
destination worker is in the same domain
destination_load + job_cells <= pre-pass domain max
```

Thus a move cannot change NUMA-domain ownership and cannot exceed the domain's previous maximum worker load.

Only the two owner boundaries adjacent to the moved job can change. The current v5.26 research objective accepts a move only when the affected tuple decreases lexicographically:

```text
1. sum of 2 MiB boundary-page penalties
2. sum of 4 KiB boundary-page penalties
3. owner-transition count
```

The pass alternates scan direction for up to 12 passes and stops when a full pass makes no move. Every accepted move strictly decreases that objective, so the search cannot cycle.

The provenance string is:

```text
objective=neighbor-page-coalesce-under-domain-cap-v5.26-plan
```

## Research pipeline

The structural comparison is now:

```text
ordinary refined-domain schedule
  -> v5.23 domain-boundary page pass
  -> v5.25 whole-domain worker-locality pass
  -> v5.26 neighbour worker coalescing pass
```

v5.23 controls NUMA-domain boundaries. v5.25 and v5.26 leave those boundaries fixed and only change worker ownership inside each domain.

The plan probe hard-checks that cross-domain 2 MiB/4 KiB page counts and cross-domain owner-transition counts are identical across all three variants.

## Why this matters on a two-domain host

The v5.24 global-unique domain-boundary optimizer has no additional degree of freedom with exactly two domains: there is only one domain boundary, so its global and local objectives coincide.

v5.25 and v5.26 work inside each domain. A `64 workers / domain_size=32` configuration still has many possible worker ownership boundaries inside each socket, so this optimization remains relevant on a two-socket host.

## Structural probe

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-domain-worker-locality-plan.sh
```

Run one topology:

```bash
./build/ramstream32_cpu_low_domain_worker_locality_plan_n27 27 64 32
```

The probe compares:

```text
baseline   = refined domain + v5.23
locality   = baseline + v5.25
coalesced  = locality + v5.26
```

Important v5.25 fields include:

```text
converted_domains
fallback_domains
baseline_max_worker_cells
locality_max_worker_cells
baseline_cross_worker_pages_2m/4k
locality_cross_worker_pages_2m/4k
baseline_worker_owner_transitions
locality_worker_owner_transitions
```

Important v5.26 fields include:

```text
coalesce_noncontiguous_domains_before
coalesce_improved_domains
coalesce_candidate_evaluations
coalesce_cap_rejections
coalesce_accepted_moves
coalesce_page_improving_moves
coalesce_transition_only_moves
coalesce_penalty_2m_before/after
coalesce_penalty_4k_before/after
coalesce_owner_transitions_before/after
coalesced_max_worker_cells
coalesced_cross_worker_pages_2m/4k
coalesced_worker_owner_transitions
worker_coalesce_build_s
```

The hard load contract is:

```text
coalesced_max_worker_cells
  <= locality_max_worker_cells
  <= baseline_max_worker_cells
```

The internal v5.26 objective tuple must also be non-increasing.

Cross-worker unique page counts are independently measured by the probe rather than inferred from the local objective. Different owner boundaries can refer to the same VM page, so a reduction in the sum of local penalties is not by itself a proof that the global unique page set shrank.

## n=27 topology sweep

Use:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-worker-locality-plan-sweep.sh
```

The sweep reports a parent classification for v5.25 and a separate v5.26 classification:

```text
v5.26_unique_page_improvement
v5.26_transition_only_improvement
v5.26_no_change
v5.26_unique_page_tradeoff
```

`v5.26_unique_page_improvement` is the strongest result. It means the exact measured cross-worker `(2 MiB, 4 KiB)` unique page tuple improved beyond the v5.25 assignment while the load cap remained intact.

`v5.26_transition_only_improvement` means the unique page tuple is unchanged but fewer worker ownership transitions remain.

`v5.26_unique_page_tradeoff` is not a production promotion result. The current local-sum objective can in principle encounter page-ID overlap effects, so this classification must remain visible until an exact global-unique worker-boundary objective is used.

## Exactness validation

`ramstream32_cpu_low_domain_worker_locality_selftest.cu` first checks synthetic partition and owner-pattern cases, including:

```text
contiguous-under-cap succeeds
contiguous-under-cap requires too many segments and falls back
0,1,0 is detected as non-contiguous worker ownership
0,0,1,1 is detected as contiguous ownership
```

It then constructs the real W=10 factorized LOW recurrence and applies:

```text
refined domain
-> v5.23 page stage
-> v5.25 worker-locality stage
-> v5.26 worker-coalescing stage
```

Two consecutive LOW generations are executed. Every main and blocked state is compared with the independent reference recurrence after both generations.

Therefore v5.25/v5.26 alter scheduling ownership only; they do not alter the recurrence being counted.

## Promotion gate

Neither v5.25 nor v5.26 is wired into the production solver yet.

For v5.25, promotion requires useful n=27 structural improvement with no load regression.

For v5.26, the stricter gate is:

```text
coalesce_accepted_moves > 0
coalesced_max_worker_cells <= locality_max_worker_cells
(coalesced_cross_worker_pages_2m, coalesced_cross_worker_pages_4k)
  <= (locality_cross_worker_pages_2m, locality_cross_worker_pages_4k)
```

with strict improvement in at least one page metric or a substantial transition reduction. A `v5.26_unique_page_tradeoff` topology should not be promoted.

After structural preflight, a same-binary runtime A/B is still required. The deciding measurements are clean `cpu_low_wall_s` and whole-solver wall time; static page counts alone do not prove a speedup.
