# RAMstream32 CPU LOW exact-neutral plateau descent

v5.35 is a research-only bridge beyond the v5.34 exact fixed point.

The v5.34 move families require a strict improvement of the exact page tuple. A
schedule can therefore be locally optimal even when an exact-neutral ownership
change would redistribute worker load and expose a later exact improvement.

v5.35 allows only neighbor moves with exactly unchanged:

```text
(global unique 2 MiB pages,
 global unique 4 KiB pages,
 weighted StorageBlock owner transitions)
```

Among such moves it requires a strict lexicographic improvement of the worker
load vector sorted from largest to smallest.

Thus the combined potential

```text
(exact page tuple, descending load profile)
```

strictly decreases and the neutral pass cannot cycle.

Provenance:

```text
objective=exact-neutral-load-descent-v5.35-plan
```

## Plateau bridge experiment

The plan performs:

```text
v5.34 best exact fixed point
  -> v5.35 exact-neutral load descent
  -> v5.34 exact fixed point again
```

The neutral step must preserve the exact tuple byte-for-byte while not increasing
maximum worker load. If the second exact search reaches a lower page tuple, the
neutral move has escaped a genuine strict-exact local plateau.

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-plateau-plan.sh
```

Sweep:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
MAX_RUN=4 MAX_SWAP=4 \
bash scripts/bench/ramstream32-cpu-low-worker-plateau-plan-sweep.sh
```

Classifications are:

```text
plateau_escape_improvement
load_profile_only
neutral_rearrangement
no_neutral_move
```

`plateau_escape_improvement` is the strongest algorithmic result: at least one
exact-neutral move occurred and the subsequent exact fixed point is strictly
better than the original v5.34 fixed point.

`load_profile_only` keeps the exact tuple but lowers maximum worker load.
`neutral_rearrangement` changes the sorted load profile without changing the
reported maximum or exact tuple. Such a result may still matter for later search
but is not sufficient for promotion by itself.

## Exactness validation

`ramstream32_cpu_low_worker_plateau_selftest.cu` checks W=10:

```text
v5.34 baseline exact fixed point
neutral exact tuple before == after
neutral max worker <= baseline max
refixed exact tuple <= baseline exact tuple
selected max worker <= baseline max
```

The final schedule executes two real LOW generations and every main/blocked
state is compared with the independent reference recurrence.

## Promotion gate

v5.35 remains research-only. A useful n=27 promotion signal requires no safety
limit hit and either a repeated `plateau_escape_improvement` or a meaningful
load-profile improvement that reduces measured LOW wall time. Static neutral
rearrangement alone is not enough for production integration.
