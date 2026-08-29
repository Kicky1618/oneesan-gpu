#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_PROFILE="${BASE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
PRECTX_PROFILE="${PRECTX_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_prectx21.env}"
FINAL_PROFILE="${FINAL_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
BASE_PREFIX="${BASE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21}"
PRECTX_PREFIX="${PRECTX_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_refine_compact_prectx21}"
SCHED_PREFIX="${SCHED_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_scheduler21}"
RUN_PRECTX="${RUN_PRECTX:-1}"
RUN_ORBIT_SCHEDULER="${RUN_ORBIT_SCHEDULER:-1}"
for x in RUN_PRECTX RUN_ORBIT_SCHEDULER; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done

echo "=== HBM tune21 base search ===" >&2
RUN_PRECTX="$RUN_PRECTX" PROFILE_OUT="$BASE_PROFILE" PREFIX="$BASE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-tune21.sh"

if [[ "$RUN_PRECTX" == 1 ]]; then
  echo "=== HBM tune21 pointer/compact prectx refinement ===" >&2
  PROFILE_IN="$BASE_PROFILE" PROFILE_OUT="$PRECTX_PROFILE" PREFIX="$PRECTX_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-compact-prectx21.sh"
else
  cp "$BASE_PROFILE" "$PRECTX_PROFILE"
fi

if [[ "$RUN_ORBIT_SCHEDULER" == 1 ]]; then
  echo "=== HBM tune21 orbit scheduler refinement ===" >&2
  PROFILE_IN="$PRECTX_PROFILE" PROFILE_OUT="$FINAL_PROFILE" PREFIX="$SCHED_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-scheduler21.sh"
else
  cp "$PRECTX_PROFILE" "$FINAL_PROFILE"
fi

echo "=== final HBM profile ===" >&2
cat "$FINAL_PROFILE"
echo "b300 HBM profile auto21 OK base_profile=$BASE_PROFILE prectx_profile=$PRECTX_PROFILE final_profile=$FINAL_PROFILE" >&2
