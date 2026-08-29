#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

joint="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"
weight="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-weight-race.sh"
adaptive="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive-race.sh"
adaptive_wrap="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive.sh"
external="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled.sh"
canonical="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh"
build="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh"

for f in "$joint" "$weight" "$adaptive" "$adaptive_wrap" "$external" "$canonical" "$build"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'N27_PRODUCER_WEIGHT_RACE=' \
  'PWW_REPEATS=' \
  'WEIGHT_REPEATS="$PWW_REPEATS"' \
  'b300x8-exact-auto-hbm-profiled-producer-weight-race.sh' \
  'WEIGHT_RACE_ONLY=1' \
  'N27_PRODUCER_ADAPTIVE_RACE=' \
  'PAC_REPEATS=' \
  'ADAPTIVE_REPEATS="$PAC_REPEATS"' \
  'b300x8-exact-auto-hbm-profiled-producer-adaptive-race.sh' \
  'ADAPTIVE_RACE_ONLY=1' \
  'ORBIT_N27_PRODUCER_ADAPTIVE_COLS' \
  'export PRODUCER_ADAPTIVE_COLS' \
  'export REBUILD_BUCKETS=1'; do
  grep -Fq "$s" "$joint" || { echo "joint calibration marker missing: $s" >&2; exit 3; }
done

for s in \
  'PRODUCER_WEIGHTS=' \
  'WEIGHT_REPEATS=' \
  'statistics.median' \
  'ORBIT_N27_PRODUCER_WEIGHT_REPEATS=' \
  'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT' \
  'CANDIDATES=orbit_tuned' \
  'FATAL producer-weight residue mismatch'; do
  grep -Fq "$s" "$weight" || { echo "weight-race marker missing: $s" >&2; exit 3; }
done

for s in \
  'PRODUCER_ADAPTIVE_THRESHOLDS=' \
  'ADAPTIVE_REPEATS=' \
  'statistics.median' \
  'ORBIT_N27_PRODUCER_ADAPTIVE_REPEATS=' \
  'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS' \
  'CANDIDATES=orbit_tuned' \
  'FATAL producer-adaptive residue mismatch' \
  'PRODUCER_ADAPTIVE_COLS="$best_threshold"'; do
  grep -Fq "$s" "$adaptive" || { echo "adaptive-race marker missing: $s" >&2; exit 3; }
done

for s in \
  'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS' \
  'export PRODUCER_ADAPTIVE_COLS="$THRESHOLD"'; do
  grep -Fq "$s" "$adaptive_wrap" || { echo "adaptive wrapper marker missing: $s" >&2; exit 3; }
done

# Shell environment is inherited across both selector layers. Guard against a
# future cleanup accidentally deleting the adaptive build knob before nvcc.
if grep -Eq '(^|[[:space:]])(unset|env[[:space:]]+-u)[[:space:]]+PRODUCER_ADAPTIVE_COLS([[:space:]]|$)' "$external" "$canonical"; then
  echo 'PRODUCER_ADAPTIVE_COLS is stripped before producer build' >&2
  exit 4
fi
grep -Fq 'b300x8-exact-auto-hbm-profiled.sh' "$external" || { echo 'external race no longer delegates to canonical selector' >&2; exit 4; }
grep -Fq 'PRODUCER_ADAPTIVE_COLS=' "$build" || { echo 'producer build wrapper lost adaptive knob' >&2; exit 4; }
grep -Fq 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS' "$build" || { echo 'producer build wrapper lost adaptive nvcc macro' >&2; exit 4; }

echo 'b300_joint_producer_calibration_preflight=OK weight=OK adaptive=OK median=OK rebuild_guard=OK env_propagation=OK gpu_work=0 actions_triggered=0'
