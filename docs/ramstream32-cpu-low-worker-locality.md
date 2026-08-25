# RAMstream32 CPU LOW intra-domain worker locality

v5.25 is a research-only scheduling stage for the persistent sparse CPU LOW executor. It targets a locality problem that remains after NUMA-domain ownership has already been made contiguous.

The production `domain` schedule gives each modeled NUMA domain one contiguous HIGH-occupancy-mask interval, then uses exact-cell LPT inside that domain. LPT is good for load balance, but adjacent occupancy masks can be scattered across many workers. That can create many cross-worker ownership boundaries inside one socket even when cross-domain ownership is already well controlled.

v5.25 asks a narrower question: can each domain keep its existing load bound while replacing its internal LPT assignment with contiguous worker intervals?

## Safety rule

For one domain, let

```text
M = maximum exact-cell load of any worker in the current LPT assignment
```

The domain's ordered mask jobs are greedily partitioned into contiguous segments with per-segment work at most `M`. If at most `nworkers` segments are required, the domain is convertible. Extra splits can be added until the desired number of active worker segments is reached; splitting an existing segment cannot increase its load above `M`.

Therefore every converted worker has load `<= M`.

If more than `nworkers` contiguous segments are required at cap `M`, the domain is not converted and its original LPT assignment is preserved exactly.

Consequently:

```text
each converted domain max <= previous domain LPT max
each fallback domain is unchanged
global max worker load cannot increase
```

The stage does not move NUMA-domain boundaries at all.

## Placement in the research pipeline

The current structural comparison starts from the same schedule used by the page-aware work:

```text
ordinary refined-domain schedule
  -> v5.23 domain-boundary page pass
  -> v5.25 intra-domain worker-locality pass
```

This separation is intentional. v5.23 controls where one NUMA domain ends and the next begins. v5.25 leaves those boundaries fixed and changes only which worker inside each domain owns each mask.

The plan probe asserts that cross-domain page counts and cross-domain owner-transition counts are identical before and after v5.25.

## Why this matters more than v5.24 on a two-domain host

The v5.24 global-unique page research optimizer operates on NUMA-domain boundaries. With exactly two domains there is only one such boundary, so the local v5.23 page penalty and the global unique page objective are mathematically identical. v5.24 cannot improve that case.

v5.25 instead works inside each domain and therefore remains relevant on an ordinary two-socket host. A 64-worker `domain_size=32` configuration still has 31 possible worker boundaries inside each socket even though there is only one cross-socket boundary.

## Structural probe

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-domain-worker-locality-plan.sh
```

Run one topology:

```bash
./build/ramstream32_cpu_low_domain_worker_locality_plan_n27 27 64 32
```

Important output fields are:

```text
objective=contiguous-under-lpt-cap-v5.25-plan
converted_domains=...
fallback_domains=...
converted_jobs=...
baseline_max_worker_cells=...
locality_max_worker_cells=...
baseline_cross_worker_pages_2m=...
locality_cross_worker_pages_2m=...
baseline_cross_worker_pages_4k=...
locality_cross_worker_pages_4k=...
baseline_worker_owner_transitions=...
locality_worker_owner_transitions=...
cross_domain_pages_2m=...
cross_domain_pages_4k=...
worker_locality_build_s=...
```

The baseline is `refined domain + v5.23 page-aware boundary pass`; the locality variant starts from the same deterministic baseline and then applies v5.25.

The hard structural check is:

```text
locality_max_worker_cells <= baseline_max_worker_cells
```

Cross-domain page counts must be exactly unchanged because v5.25 never changes a domain boundary.

Cross-worker page counts are measured, not assumed. Contiguous worker intervals normally reduce owner transitions, but page alignment can make the exact unique 2 MiB/4 KiB result differ from the raw transition count.

## n=27 topology sweep

Use:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-domain-worker-locality-plan-sweep.sh
```

These are deliberately two-domain-oriented configurations because that is the case where v5.24 has no extra domain-boundary objective but v5.25 can still reduce intra-domain sharing.

The sweep classifies each topology as one of:

```text
page_improvement
transition_only_improvement
no_change
page_tradeoff
```

`page_improvement` is the strongest promotion signal: max worker load is already guaranteed non-increasing, and the exact static cross-worker page tuple also improves.

`transition_only_improvement` means fewer ownership changes but no reduction in the unique page tuple.

`page_tradeoff` means the contiguous conversion reduced some structural disorder but worsened the `(2 MiB, 4 KiB)` page tuple. Such a topology should not be promoted without a strong measured runtime reason.

## Exactness validation

`ramstream32_cpu_low_domain_worker_locality_selftest.cu` contains two kinds of checks.

First, synthetic job weights exercise both branches of the partitioner:

```text
contiguous-under-cap succeeds
contiguous-under-cap requires too many segments and falls back
```

Second, W=10 constructs the real factorized LOW recurrence, applies:

```text
refined domain
-> v5.23 page stage
-> v5.25 worker-locality stage
```

and executes two consecutive LOW generations. Every main and blocked state is compared with the independent reference recurrence after both generations.

Thus v5.25 changes only ownership and does not change the counted recurrence.

## Promotion gate

v5.25 is not wired into the production solver yet. Promotion should require n=27 preflight evidence that:

```text
converted_domains > 0
locality_max_worker_cells <= baseline_max_worker_cells
(locality_cross_worker_pages_2m, locality_cross_worker_pages_4k)
  <= (baseline_cross_worker_pages_2m, baseline_cross_worker_pages_4k)
```

with a useful reduction in at least one page metric or a large reduction in owner transitions.

After that, a same-binary runtime A/B should be added before making the stage a production option. The deciding measurements are clean `cpu_low_wall_s` and whole-solver wall time; static page counts alone do not prove a speedup.
