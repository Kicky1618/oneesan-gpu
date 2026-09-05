#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

joint="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"
weight="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-weight-race.sh"
adaptive="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive-race.sh"
coordinate="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-coordinate-race.sh"
adaptive_wrap="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive.sh"
single="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
buckets="$ONEESAN_ROOT/scripts/build/b300-profiled-buckets-only.sh"
build="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh"

for f in "$joint" "$weight" "$adaptive" "$coordinate" "$adaptive_wrap" "$single" "$buckets" "$build"; do
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
  'final_race=single_pass' \
  'b300x8-race-external-forced-profiled-once.sh'; do
  grep -Fq "$s" "$joint" || { echo "joint calibration marker missing: $s" >&2; exit 3; }
done

for s in \
  'PRODUCER_WEIGHTS=' \
  'WEIGHT_ADAPTIVE_COLS=' \
  'WEIGHT_REPEATS=' \
  'statistics.median' \
  'ORBIT_N27_PRODUCER_WEIGHT_REPEATS=' \
  'ORBIT_N27_PRODUCER_WEIGHT_FIXED_ADAPTIVE_COLS=' \
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
  'COORDINATE_ROUNDS=' \
  'COORDINATE_REPEATS=' \
  'WEIGHT_ADAPTIVE_COLS="$old_t"' \
  'b300x8-exact-auto-hbm-profiled-producer-weight-race.sh' \
  'b300x8-exact-auto-hbm-profiled-producer-adaptive-race.sh' \
  'PRODUCER COORDINATE converged' \
  'ORBIT_N27_PRODUCER_COORDINATE_WEIGHT=' \
  'ORBIT_N27_PRODUCER_COORDINATE_ADAPTIVE_COLS='; do
  grep -Fq "$s" "$coordinate" || { echo "coordinate-refine marker missing: $s" >&2; exit 3; }
done

for s in \
  'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS' \
  'export PRODUCER_ADAPTIVE_COLS="$THRESHOLD"'; do
  grep -Fq "$s" "$adaptive_wrap" || { echo "adaptive wrapper marker missing: $s" >&2; exit 3; }
done

for s in \
  'b300-profiled-buckets-only.sh' \
  'one complete prime per candidate' \
  'FATAL single-pass residue mismatch' \
  'profile_sha256'; do
  grep -Fq "$s" "$single" || { echo "single-pass race marker missing: $s" >&2; exit 4; }
done

for s in \
  'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT' \
  'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS' \
  '_pww${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT}_pac${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS}_ppw' \
  'PRODUCER_WORKER_WEIGHT="$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT"' \
  'PRODUCER_ADAPTIVE_COLS="$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS"' \
  'PROFILE_SHA256='; do
  grep -Fq "$s" "$buckets" || { echo "bucket build-only marker missing: $s" >&2; exit 4; }
done

grep -Fq 'PRODUCER_ADAPTIVE_COLS=' "$build" || { echo 'producer build wrapper lost adaptive knob' >&2; exit 4; }
grep -Fq 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS' "$build" || { echo 'producer build wrapper lost adaptive nvcc macro' >&2; exit 4; }

echo 'b300_joint_producer_calibration_preflight=OK weight=OK adaptive=OK coordinate=OK median=OK pww_pac_fingerprint=OK single_pass=OK gpu_work=0 actions_triggered=0'
