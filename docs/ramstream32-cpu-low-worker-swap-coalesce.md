# RAMstream32 CPU LOW bounded-swap exact coalescing

v5.33 is a research-only search extension after the v5.32 bounded-run stage.

A one-way run move can be blocked when the destination worker is already at its
allowed domain load cap.  Two neighboring workers may still be able to exchange
short chunks atomically while both remain under the same cap.

## Legal swap

At an adjacent worker-owner boundary inside one NUMA domain, choose:

```text
left suffix length  1..max_swap
right prefix length 1..max_swap
```

The left chunk changes to the right owner and the right chunk changes to the
left owner in one atomic scheduling step.

For left/right worker loads `A/B` and chunk costs `x/y`, legality requires:

```text
A' = A - x + y <= domain cap
B' = B - y + x <= domain cap
```

The chunks are already owned by their respective workers, so `x<=A` and `y<=B`.
Domain ownership is unchanged.

The exact dense workspace evaluates the changed outer boundaries in:

```text
(global unique 2 MiB pages,
 global unique 4 KiB pages,
 weighted StorageBlock owner transitions)
```

The central A/B boundary becomes B/A and remains active, so it does not change
page-set membership. Generic edge accounting still checks it safely.

Only strict exact-objective improvements are committed.

Provenance:

```text
objective=bounded-swap-global-unique-v5.33-plan
```

## n=27 plan sweep

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-swap-plan.sh
```

Sweep after a fixed v5.32 `RUN_MAX=4` parent:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
RUN_MAX=4 \
SWAPS='1 2 4 8' \
bash scripts/bench/ramstream32-cpu-low-worker-swap-plan-sweep.sh
```

The sweep emits:

```text
atomic_swap_improvement
swap_improvement
no_change
```

`atomic_swap_improvement` for `max_swap>1` means a larger atomic exchange reaches
an exact tuple better than the one reachable with one-job-per-side swaps under
the same search order.

Important metrics are:

```text
accepted_swaps
moved_jobs
max_left_used
max_right_used
swap_candidate_evaluations
swap_cap_rejections
move_limit_hit
swap_build_s
```

## Exactness validation

`ramstream32_cpu_low_worker_swap_selftest.cu` builds the same v5.31 selected
parent, applies v5.32 `max_run=4`, then independently evaluates `max_swap=1` and
`max_swap=4` from that identical run-local-minimum schedule.

Both swap outputs must preserve the exact-objective and maximum-worker
non-regression contracts.  The `max_swap=4` result then executes two real W=10
LOW generations and every main/blocked state is compared with the independent
reference recurrence.

## Promotion gate

v5.33 remains research-only.  A useful n=27 result requires:

```text
move_limit_hit=0
max_worker_cells_after <= max_worker_cells_before
exact tuple after <= exact tuple before
```

Algorithmic evidence is strongest when:

```text
max_swap > 1
max_left_used > 1 or max_right_used > 1
classification=atomic_swap_improvement
```

If swaps add value after v5.32, the next search should alternate run and swap
passes to a common fixed point. Since each accepted run or swap strictly
improves the same exact tuple, such alternation cannot cycle.
