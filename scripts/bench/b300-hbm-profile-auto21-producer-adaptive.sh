#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_FINAL="${BASE_FINAL:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21_base.env}"
FINAL_PROFILE="${FINAL_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ADAPTIVE_PREFIX="${ADAPTIVE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_producer_adaptive21}"
RUN_ADAPTIVE="${RUN_ADAPTIVE:-1}"
[[ "$RUN_ADAPTIVE" == 0 || "$RUN_ADAPTIVE" == 1 ]] || { echo 'RUN_ADAPTIVE must be 0/1' >&2; exit 2; }

# Let the authoritative tuner finish first. Write its final profile to a private
# intermediate path so this wrapper can promote or reject the adaptive branch
# without changing any earlier tuning stage.
FINAL_PROFILE="$BASE_FINAL" bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-auto21.sh"
[[ -f "$BASE_FINAL" ]] || { echo "missing base final profile=$BASE_FINAL" >&2; exit 3; }

if [[ "$RUN_ADAPTIVE" == 0 ]]; then
  cp "$BASE_FINAL" "$FINAL_PROFILE"
  cat "$FINAL_PROFILE"
  exit 0
fi

ORBITCTA_FLAT_DYNAMIC=0
ORBITCTA_FLAT_DYNAMIC_PIPE2=0
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP=0
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT=1
# shellcheck disable=SC1090
source "$BASE_FINAL"
if [[ "${ORBITCTA_FLAT_DYNAMIC:-0}" == 1 && "${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}" == 1 && "${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP:-0}" == 1 && "${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT:-1}" != 1 ]]; then
  PROFILE_IN="$BASE_FINAL" PROFILE_OUT="$FINAL_PROFILE" PREFIX="$ADAPTIVE_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-producer-adaptive21.sh"
else
  echo "=== skip adaptive producer threshold: selected path is not eligible ===" >&2
  cp "$BASE_FINAL" "$FINAL_PROFILE"
fi

cat "$FINAL_PROFILE"
echo "b300 HBM auto21 producer-adaptive wrapper OK base=$BASE_FINAL final=$FINAL_PROFILE" >&2
