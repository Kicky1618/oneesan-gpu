# RAMstream32 CPU LOW worker research preflight

This is the convergence entry point for the research-only LOW worker-locality
series v5.29 through v5.36.  It runs plan probes only; it does not allocate the
multi-terabyte authoritative n=27 state arrays and does not run the exact count.

## One command

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
RUNS='1 2 4 8' \
SWAPS='1 2 4 8' \
MAX_RUN=4 \
MAX_SWAP=4 \
bash scripts/bench/ramstream32-cpu-low-worker-research-preflight.sh
```

The suite executes, in order:

```text
v5.29/v5.30  flat-page-delta vs dense-page representation
v5.31        shared immutable exact workspace
v5.32        bounded atomic run moves
v5.33        bounded adjacent run swaps
v5.34        alternating run/swap exact fixed point
v5.35        exact-neutral load-profile plateau bridge
v5.36        augmented exact/neutral outer fixed point
```

Every stage uses its existing standalone sweep rather than reimplementing its
result parser.  `set -euo pipefail` means any build, provenance, accounting, or
no-regression failure aborts the suite immediately.

## Output

A timestamped directory is created under:

```text
work/bench_ramstream32_cpu_low_worker_research_preflight/
```

It contains:

- one log and one result subdirectory per stage;
- `manifest.txt` with commit, host, topology, and search limits;
- `summary.txt` containing the stage `comparison`, `summary`, `results=`, and
  `metadata=` lines.

The manifest records:

```text
mode=plan-only-no-authoritative-ram
```

so plan results cannot be confused with a real n=27 runtime measurement.

## Decision order

Do not promote later stages merely because they are newer.

1. **v5.30 representation** — require identical v5.29/v5.30 schedules and a
   construction-time or metadata advantage.
2. **v5.31 workspace** — require `workspace_audit_ok=1` for every topology and
   evaluate `shared_vs_legacy_speedup`, including `workspace_audit_s`.
3. **v5.32/v5.33 move families** — keep larger run/swap bounds only when they
   improve the exact tuple enough to justify their extra candidate count.
4. **v5.34 ordering** — check whether run/swap order produces distinct greedy
   basins. If all topologies tie, keep the simpler deterministic order.
5. **v5.35 plateau bridge** — `plateau_escape_improvement` is evidence that
   exact-neutral load descent exposes a new strict-exact move. A load-only
   improvement is useful only if runtime locality also improves.
6. **v5.36 augmented fixed point** — require no exact/max-load regression and a
   repeatable plan improvement before doing a real runtime A/B.

The final promotion gate is still a same-binary runtime A/B on the target host.
A lower page-count plan is not itself a measured speedup.

## CI policy

The v5.30-v5.36 W10 correctness gates are consolidated into one workflow:

```text
.github/workflows/ramstream32-worker-research-ci.yml
```

The older per-stage workflows were removed so a shared-header change does not
start many redundant CUDA Actions jobs.  The consolidated workflow keeps the
specialized exact selftests and compiles all plan wrappers in a single CUDA
container.
