# RAMstream32 CPU LOW alternating exact fixed point

v5.34 is a research-only local-search composition after the v5.31 selected
multistart schedule.

Two exact-improving move families are available:

```text
v5.32 bounded one-way run moves
v5.33 bounded adjacent-run swaps
```

Either family can create opportunities for the other. v5.34 therefore alternates
them until one complete round accepts no move.

## Two search orders

Two branches start from the same v5.31 selected schedule:

```text
run-swap:
  run fixed point -> swap fixed point -> repeat

swap-run:
  swap fixed point -> run fixed point -> repeat
```

Both component searches commit only strict improvements in the same tuple:

```text
(global unique 2 MiB pages,
 global unique 4 KiB pages,
 weighted StorageBlock owner transitions)
```

Therefore the alternating search cannot cycle. The only incomplete cases are an
explicit component move limit or the outer round safety limit.

Provenance:

```text
objective=alternating-run-swap-v5.34-plan
```

The plan evaluates both orders and chooses the smaller final exact tuple. Exact
ties use smaller maximum worker load, then `run-swap` as deterministic fallback.

## n=27 topology sweep

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-fixedpoint-plan.sh
```

Sweep:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
MAX_RUN=4 MAX_SWAP=4 \
bash scripts/bench/ramstream32-cpu-low-worker-fixedpoint-plan-sweep.sh
```

The sweep reports:

```text
run_swap_wins
swap_run_wins
exact_tie
```

as the basin classification, plus `fixedpoint_improvement` or `no_change`
relative to the v5.31 selected parent.

Important diagnostics are:

```text
rs_rounds / sr_rounds
rs_run_accepted / sr_run_accepted
rs_swap_accepted / sr_swap_accepted
rs_build_s / sr_build_s
selected_order
selected exact tuple
```

A `run_swap_wins` or `swap_run_wins` result proves that exact-greedy ordering
changes the reachable fixed point for that topology. Multiple rounds with moves
in a later pass prove cross-family opportunity creation directly.

## Exactness validation

`ramstream32_cpu_low_worker_fixedpoint_selftest.cu` constructs one W=10 v5.31
selected parent and copies it into both order branches. It requires:

```text
same initial exact tuple
no component/round limit hit
both final exact tuples <= parent
both final worker maxima <= parent maximum
```

The better fixed point is copied into a third pool and executes two real LOW
generations. Every main and blocked state is compared with the independent
reference recurrence.

## Promotion gate

v5.34 remains research-only. Structural promotion requires:

```text
limits_clear=1
selected tuple <= parent and both order branches
selected_max_worker_cells <= parent_max_worker_cells
```

Algorithmic value is strongest when at least one intended n=27 topology has a
non-trivial second round or an order-dependent winner. Runtime promotion still
requires same-binary measurement including one-time schedule construction and
repeated LOW row execution.
