# RAMstream32 CPU LOW augmented fixed point (v5.36)

v5.36 is a research-only outer fixed point for the LOW worker-locality search.
It is not wired into the production solver.

The search uses a lexicographic combined potential:

```text
1. exact worker-boundary objective
   (global unique 2 MiB pages,
    global unique 4 KiB pages,
    weighted StorageBlock owner transitions)
2. descending worker-load profile
```

The first component is always primary.  A neutral move is allowed only when the
exact tuple is unchanged and the descending worker-load profile becomes strictly
smaller.

## Outer loop

Each augmented round performs:

```text
best exact fixed point
  = best of run->swap and swap->run v5.34 fixed points
then
exact-neutral load descent (v5.35)
```

The loop repeats until neither stage changes the schedule.

This cannot cycle:

- an exact improvement strictly decreases the primary tuple;
- an exact-neutral move leaves the primary tuple unchanged and strictly
  decreases the secondary load profile.

The implementation rejects any exact, max-worker-load, component-limit, or
round-limit regression.

## Preflight plan

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-augmented-plan.sh
```

Sweep the usual two-domain topologies:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
MAX_RUN=4 MAX_SWAP=4 \
bash scripts/bench/ramstream32-cpu-low-worker-augmented-plan-sweep.sh
```

The baseline is one `best exact fixed point` from the same v5.31 multistart
parent.  The augmented branch starts from the same parent and runs the outer
fixed point above.

Important fields include:

```text
baseline_pages_2m / augmented_pages_2m
baseline_pages_4k / augmented_pages_4k
baseline_transitions / augmented_transitions
baseline_max_worker_cells / augmented_max_worker_cells
augmented_rounds
exact_schedule_changes
exact_primary_improvements
exact_profile_improvements
neutral_moves
neutral_candidates
augmented_build_s
workspace_build_s
```

The sweep classifies each topology as one of:

```text
augmented_plateau_escape
augmented_exact_improvement
augmented_load_improvement
neutral_fixedpoint_change
no_change
```

Do not claim a performance benefit until n=27 preflight and a real CPU LOW
runtime A/B both show one.  The plan search may improve page locality while its
own construction cost is too large to amortize.

## Correctness gate

`ramstream32_cpu_low_worker_augmented_selftest.cu` uses W=10 and:

1. constructs the same v5.31 multistart parent;
2. compares the ordinary best exact fixed point with v5.36;
3. requires v5.36 to be no worse in exact tuple and max worker load;
4. if the exact tuple ties, requires the sorted load profile to be no worse;
5. runs the selected augmented schedule through the actual LOW recurrence twice;
6. compares every main and blocked state after each generation with an
   independent reference recurrence.

Production promotion requires this gate, the plan no-regression checks, and a
measured n=27 runtime win.  Until then v5.36 remains research-only.
