# RAMstream32 CPU LOW bounded-run exact coalescing

v5.32 is a research-only search extension after the v5.31 shared multistart selector.

The exact single-job search can reach a local minimum where moving one job is
worse, but moving two or more adjacent jobs as one atomic operation is better.
v5.32 searches bounded chunks without materializing the intermediate one-job
states.

## Legal run move

Within one NUMA domain, take a maximal interval owned by one worker. v5.32 may
move a prefix to the immediate left owner or a suffix to the immediate right
owner. The run length is bounded by `max_run`.

A run is legal only if:

```text
all moved jobs have the same source owner
destination is the adjacent worker owner
destination remains inside the same domain
destination_load + run_cells <= pre-pass domain max
```

Therefore domain ownership never changes and maximum worker load cannot increase.

Only the two run-end boundaries can change. The v5.31 dense exact workspace is
used to evaluate the atomic final delta in:

```text
(global unique 2 MiB pages,
 global unique 4 KiB pages,
 weighted StorageBlock owner transitions)
```

A run is committed only for a strict exact-objective improvement.

The provenance string is:

```text
objective=bounded-run-global-unique-v5.32-plan
```

## Search order

Each iteration enumerates all currently maximal owner runs and prefix/suffix
chunks up to `max_run`, then commits the best exact-improving candidate. Ties use
shorter run length, then lower ordered position, then lower destination worker.
The process repeats until no improvement remains or the safety move limit is hit.

The plan treats `move_limit_hit=1` as an incomplete search result.

## n=27 plan sweep

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-run-plan.sh
```

Sweep:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
RUNS='1 2 4 8' \
bash scripts/bench/ramstream32-cpu-low-worker-run-plan-sweep.sh
```

Every run length starts from the same v5.31 selected multistart schedule.
Important fields include:

```text
before_pages_2m/4k
after_pages_2m/4k
before_transitions/after_transitions
accepted_runs
moved_jobs
max_run_used
candidate_evaluations
cap_rejections
move_limit_hit
run_build_s
```

The sweep compares larger runs against `max_run=1` and emits:

```text
atomic_run_improvement
run_improvement
no_change
```

`atomic_run_improvement` for `max_run>1` is the key result: the final exact tuple
is better than the result reachable with one-job moves under the same best-move
search order. That is direct evidence that atomic multi-job steps escape a real
single-move local minimum.

## Exactness validation

`ramstream32_cpu_low_worker_run_selftest.cu` constructs the same v5.31 selected
parent twice and independently runs `max_run=1` and `max_run=4`. Both exact tuples
and worker maxima are required to be non-regressing. The `max_run=4` schedule
then executes two real W=10 LOW generations and every main/blocked state is
compared with the independent reference recurrence.

## Promotion gate

v5.32 is not production code. Promotion requires:

```text
move_limit_hit=0
max_worker_cells_after <= max_worker_cells_before
exact tuple after <= exact tuple before
```

and, for algorithmic value, at least one intended n=27 topology should show:

```text
max_run > 1
max_run_used > 1
classification=atomic_run_improvement
```

A static improvement is still not a runtime result. After a useful topology is
found, same-binary timing must include schedule construction plus repeated LOW
row execution before production integration.
