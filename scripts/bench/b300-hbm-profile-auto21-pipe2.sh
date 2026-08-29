#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PREPIPE_PROFILE="${PREPIPE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined_prepipe21.env}"
OUT_PROFILE="${OUT_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PIPE_PREFIX="${PIPE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_pipe21}"
RUN_ORBIT_DYNAMIC_PIPE2="${RUN_ORBIT_DYNAMIC_PIPE2:-1}"
[[ "$RUN_ORBIT_DYNAMIC_PIPE2" == 0 || "$RUN_ORBIT_DYNAMIC_PIPE2" == 1 ]] || { echo 'RUN_ORBIT_DYNAMIC_PIPE2 must be 0/1' >&2; exit 2; }
[[ "$PREPIPE_PROFILE" != "$OUT_PROFILE" ]] || { echo 'PREPIPE_PROFILE and OUT_PROFILE must differ' >&2; exit 2; }

# Preserve the established tuner exactly; only redirect its final output so the
# pipe2 experiment cannot overwrite the input profile it is comparing against.
echo "=== HBM auto21 established pipeline (pre-pipe2) ===" >&2
FINAL_PROFILE="$PREPIPE_PROFILE" bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-auto21.sh"
[[ -f "$PREPIPE_PROFILE" ]] || { echo "missing pre-pipe profile=$PREPIPE_PROFILE" >&2; exit 3; }

ORBITCTA_FLAT_DYNAMIC=0 ORBITCTA_FLAT_DYNAMIC_PIPE2=0
# shellcheck disable=SC1090
source "$PREPIPE_PROFILE"
DYNAMIC_SELECTED="${ORBITCTA_FLAT_DYNAMIC:-0}"
[[ "$DYNAMIC_SELECTED" == 0 || "$DYNAMIC_SELECTED" == 1 ]] || { echo 'bad ORBITCTA_FLAT_DYNAMIC in pre-pipe profile' >&2; exit 3; }

if [[ "$RUN_ORBIT_DYNAMIC_PIPE2" == 1 && "$DYNAMIC_SELECTED" == 1 ]]; then
  echo "=== HBM tune21 dynamic two-context prepare/columns pipeline ===" >&2
  PROFILE_IN="$PREPIPE_PROFILE" PROFILE_OUT="$OUT_PROFILE" PREFIX="$PIPE_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-pipe21.sh"
else
  [[ "$DYNAMIC_SELECTED" == 1 ]] || echo "=== skip dynamic pipe2: dynamic queue not selected ===" >&2
  cp "$PREPIPE_PROFILE" "$OUT_PROFILE"
fi

echo "=== final HBM profile (pipe2-aware) ===" >&2
cat "$OUT_PROFILE"
echo "b300 HBM profile auto21 pipe2 OK prepipe_profile=$PREPIPE_PROFILE final_profile=$OUT_PROFILE dynamic_selected=$DYNAMIC_SELECTED" >&2
