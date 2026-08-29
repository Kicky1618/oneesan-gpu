#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_PROFILE="${BASE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
PRECTX_PROFILE="${PRECTX_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_prectx21.env}"
SCHED_PROFILE="${SCHED_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_scheduler21.env}"
ADV_PROFILE="${ADV_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_advanced21.env}"
FINAL_PROFILE="${FINAL_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
BASE_PREFIX="${BASE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21}"
PRECTX_PREFIX="${PRECTX_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_refine_compact_prectx21}"
SCHED_PREFIX="${SCHED_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_scheduler21}"
ADV_PREFIX="${ADV_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_advanced21}"
QUAD_DESC_PREFIX="${QUAD_DESC_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad_desc21}"
RUN_PRECTX="${RUN_PRECTX:-1}"
RUN_ORBIT_SCHEDULER="${RUN_ORBIT_SCHEDULER:-1}"
# Keep RUN_ORBIT_QUAD as a compatibility alias for older launch commands.
RUN_ORBIT_ADVANCED="${RUN_ORBIT_ADVANCED:-${RUN_ORBIT_QUAD:-1}}"
RUN_ORBIT_QUAD_DESC="${RUN_ORBIT_QUAD_DESC:-1}"
for x in RUN_PRECTX RUN_ORBIT_SCHEDULER RUN_ORBIT_ADVANCED RUN_ORBIT_QUAD_DESC; do
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
  PROFILE_IN="$PRECTX_PROFILE" PROFILE_OUT="$SCHED_PROFILE" PREFIX="$SCHED_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-scheduler21.sh"
else
  cp "$PRECTX_PROFILE" "$SCHED_PROFILE"
fi

if [[ "$RUN_ORBIT_ADVANCED" == 1 ]]; then
  echo "=== HBM tune21 advanced orbit branch selection ===" >&2
  PROFILE_IN="$SCHED_PROFILE" PROFILE_OUT="$ADV_PROFILE" PREFIX="$ADV_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-advanced21.sh"
else
  cp "$SCHED_PROFILE" "$ADV_PROFILE"
fi

# Descriptor MLP only applies when advanced branch selection actually chose the
# sparse64 chunked-quad overlap-local path. Flat-bid is mutually exclusive and
# bypasses this refinement by construction.
if [[ "$RUN_ORBIT_QUAD_DESC" == 1 ]]; then
  ORBIT_QUAD_MLP=0 ORBIT_QUAD_OVERLAP_LOCAL=0 ORBIT_SPARSE64=0
  # shellcheck disable=SC1090
  source "$ADV_PROFILE"
  if [[ "${ORBIT_QUAD_MLP:-0}" == 1 && "${ORBIT_QUAD_OVERLAP_LOCAL:-0}" == 1 && "${ORBIT_SPARSE64:-0}" == 1 ]]; then
    echo "=== HBM tune21 quad sparse descriptor MLP refinement ===" >&2
    PROFILE_IN="$ADV_PROFILE" PROFILE_OUT="$FINAL_PROFILE" PREFIX="$QUAD_DESC_PREFIX" \
      bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-quad-desc21.sh"
  else
    echo "=== skip quad sparse descriptor MLP: advanced winner is not sparse QOL ===" >&2
    cp "$ADV_PROFILE" "$FINAL_PROFILE"
  fi
else
  cp "$ADV_PROFILE" "$FINAL_PROFILE"
fi

echo "=== final HBM profile ===" >&2
cat "$FINAL_PROFILE"
echo "b300 HBM profile auto21 OK base_profile=$BASE_PROFILE prectx_profile=$PRECTX_PROFILE sched_profile=$SCHED_PROFILE advanced_profile=$ADV_PROFILE final_profile=$FINAL_PROFILE" >&2
