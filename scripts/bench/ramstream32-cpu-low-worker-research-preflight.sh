#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
CONFIGS="${CONFIGS:-32:16 64:32 96:48 128:64}"
RUNS="${RUNS:-1 2 4 8}"
SWAPS="${SWAPS:-1 2 4 8}"
MAX_RUN="${MAX_RUN:-4}"
MAX_SWAP="${MAX_SWAP:-4}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_research_preflight}"

if (( N < 2 || N > 27 )); then
  echo "N must be in 2..27" >&2
  exit 2
fi
for name in MAX_RUN MAX_SWAP; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || (( value > 64 )); then
    echo "$name must be in 1..64" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$OUT_DIR/n${N}-${ts}"
mkdir -p "$run_dir"
manifest="$run_dir/manifest.txt"
summary="$run_dir/summary.txt"

cat >"$manifest" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
runs=$RUNS
swaps=$SWAPS
max_run=$MAX_RUN
max_swap=$MAX_SWAP
mode=plan-only-no-authoritative-ram
EOF
: >"$summary"

run_stage() {
  local name="$1"
  shift
  local log="$run_dir/${name}.log"
  echo "=== $name ===" | tee -a "$summary"
  "$@" 2>&1 | tee "$log"
  grep -E '^(comparison|summary|results=|metadata=)' "$log" | tee -a "$summary" || true
  echo >>"$summary"
}

# v5.29/v5.30 representation equivalence and construction cost.
run_stage dense-compare \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" \
      OUT_DIR="$run_dir/dense-compare" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-dense-compare-plan-sweep.sh

# v5.31 shared immutable exact workspace versus branch-local construction.
run_stage shared-workspace \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" \
      OUT_DIR="$run_dir/shared-workspace" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-shared-multistart-plan-sweep.sh

# v5.32 bounded atomic run moves.
run_stage run-coalesce \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" RUNS="$RUNS" \
      OUT_DIR="$run_dir/run-coalesce" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-run-plan-sweep.sh

# v5.33 bounded adjacent run swaps, using the same v5.32 max-run parent.
run_stage swap-coalesce \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" SWAPS="$SWAPS" MAX_RUN="$MAX_RUN" \
      OUT_DIR="$run_dir/swap-coalesce" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-swap-plan-sweep.sh

# v5.34 alternating run/swap fixed points and order selection.
run_stage exact-fixedpoint \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" MAX_RUN="$MAX_RUN" MAX_SWAP="$MAX_SWAP" \
      OUT_DIR="$run_dir/exact-fixedpoint" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-fixedpoint-plan-sweep.sh

# v5.35 exact-neutral load-profile bridge followed by exact refix.
run_stage neutral-plateau \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" MAX_RUN="$MAX_RUN" MAX_SWAP="$MAX_SWAP" \
      OUT_DIR="$run_dir/neutral-plateau" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-plateau-plan-sweep.sh

# v5.36 outer augmented fixed point.
run_stage augmented-fixedpoint \
  env N="$N" ARCH="$ARCH" CONFIGS="$CONFIGS" MAX_RUN="$MAX_RUN" MAX_SWAP="$MAX_SWAP" \
      OUT_DIR="$run_dir/augmented-fixedpoint" BUILD=1 \
  bash scripts/bench/ramstream32-cpu-low-worker-augmented-plan-sweep.sh

{
  echo "run_dir=$run_dir"
  echo "manifest=$manifest"
  echo "summary=$summary"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$manifest"

echo "research_preflight_complete=1"
echo "run_dir=$run_dir"
echo "summary=$summary"
