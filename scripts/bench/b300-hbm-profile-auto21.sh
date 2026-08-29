#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_PROFILE="${BASE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
PRECTX_PROFILE="${PRECTX_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_prectx21.env}"
SCHED_PROFILE="${SCHED_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_scheduler21.env}"
ADV_PROFILE="${ADV_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_advanced21.env}"
DYNAMIC_PROFILE="${DYNAMIC_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic21.env}"
DYNAMIC_BID_PROFILE="${DYNAMIC_BID_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_bid21.env}"
DYNAMIC_FUSE_PROFILE="${DYNAMIC_FUSE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_fuse21.env}"
DYNAMIC_ADAPTIVE_PROFILE="${DYNAMIC_ADAPTIVE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_adaptive21.env}"
WARPCOOP_PROFILE="${WARPCOOP_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_warpcoop21.env}"
WARPCOOP_AUTO_PROFILE="${WARPCOOP_AUTO_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_warpcoop_auto21.env}"
DESC_PROFILE="${DESC_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_desc21.env}"
LOCAL0_PROFILE="${LOCAL0_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_local021.env}"
GROUP_PROFILE="${GROUP_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_group21.env}"
FINAL_PROFILE="${FINAL_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
BASE_PREFIX="${BASE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21}"
PRECTX_PREFIX="${PRECTX_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_refine_compact_prectx21}"
SCHED_PREFIX="${SCHED_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_scheduler21}"
ADV_PREFIX="${ADV_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_advanced21}"
DYNAMIC_PREFIX="${DYNAMIC_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic21}"
DYNAMIC_BID_PREFIX="${DYNAMIC_BID_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_bid21}"
DYNAMIC_FUSE_PREFIX="${DYNAMIC_FUSE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_fuse21}"
DYNAMIC_ADAPTIVE_PREFIX="${DYNAMIC_ADAPTIVE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_adaptive21}"
WARPCOOP_PREFIX="${WARPCOOP_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_warpcoop21}"
WARPCOOP_AUTO_PREFIX="${WARPCOOP_AUTO_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_warpcoop_auto21}"
QUAD_DESC_PREFIX="${QUAD_DESC_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad_desc21}"
QUAD_LOCAL0_PREFIX="${QUAD_LOCAL0_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad_local021}"
QUAD_GROUP_PREFIX="${QUAD_GROUP_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad_group21}"
QUAD_PREFETCH_PREFIX="${QUAD_PREFETCH_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad_prefetch21}"
RUN_PRECTX="${RUN_PRECTX:-1}"
RUN_ORBIT_SCHEDULER="${RUN_ORBIT_SCHEDULER:-1}"
# Keep RUN_ORBIT_QUAD as a compatibility alias for older launch commands.
RUN_ORBIT_ADVANCED="${RUN_ORBIT_ADVANCED:-${RUN_ORBIT_QUAD:-1}}"
RUN_ORBIT_DYNAMIC="${RUN_ORBIT_DYNAMIC:-1}"
RUN_ORBIT_DYNAMIC_BID="${RUN_ORBIT_DYNAMIC_BID:-1}"
RUN_ORBIT_DYNAMIC_FUSE="${RUN_ORBIT_DYNAMIC_FUSE:-1}"
RUN_ORBIT_DYNAMIC_ADAPTIVE="${RUN_ORBIT_DYNAMIC_ADAPTIVE:-1}"
RUN_ORBIT_WARPCOOP="${RUN_ORBIT_WARPCOOP:-1}"
RUN_ORBIT_WARPCOOP_AUTO="${RUN_ORBIT_WARPCOOP_AUTO:-1}"
RUN_ORBIT_QUAD_DESC="${RUN_ORBIT_QUAD_DESC:-1}"
RUN_ORBIT_QUAD_LOCAL0="${RUN_ORBIT_QUAD_LOCAL0:-1}"
RUN_ORBIT_QUAD_GROUP="${RUN_ORBIT_QUAD_GROUP:-1}"
RUN_ORBIT_QUAD_PREFETCH="${RUN_ORBIT_QUAD_PREFETCH:-1}"
for x in RUN_PRECTX RUN_ORBIT_SCHEDULER RUN_ORBIT_ADVANCED RUN_ORBIT_DYNAMIC RUN_ORBIT_DYNAMIC_BID RUN_ORBIT_DYNAMIC_FUSE RUN_ORBIT_DYNAMIC_ADAPTIVE RUN_ORBIT_WARPCOOP RUN_ORBIT_WARPCOOP_AUTO RUN_ORBIT_QUAD_DESC RUN_ORBIT_QUAD_LOCAL0 RUN_ORBIT_QUAD_GROUP RUN_ORBIT_QUAD_PREFETCH; do
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

# Dynamic flat queue is mutually exclusive with the chunked-quad path. Race the
# best lease batch against the completed advanced selection, using the same
# pre-advanced scheduler root. Promote only a genuinely dynamic winner.
if [[ "$RUN_ORBIT_DYNAMIC" == 1 && "$RUN_ORBIT_ADVANCED" == 1 ]]; then
  echo "=== HBM tune21 dynamic flat queue refinement ===" >&2
  PROFILE_BASE="$SCHED_PROFILE" PROFILE_ADV="$ADV_PROFILE" ADV_SUMMARY="${ADV_PREFIX}_summary.tsv" \
    PROFILE_OUT="$DYNAMIC_PROFILE" PREFIX="$DYNAMIC_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic21.sh"
else
  cp "$ADV_PROFILE" "$DYNAMIC_PROFILE"
fi

ORBITCTA_FLAT_DYNAMIC=0 ORBIT_PRECTX_FORWARD=0 ORBIT_PRECTX_REVERSE=0 ORBIT_PRECTX_COMPACT=0
# shellcheck disable=SC1090
source "$DYNAMIC_PROFILE"
DYNAMIC_SELECTED="${ORBITCTA_FLAT_DYNAMIC:-0}"
[[ "$DYNAMIC_SELECTED" == 0 || "$DYNAMIC_SELECTED" == 1 ]] || { echo 'bad ORBITCTA_FLAT_DYNAMIC after dynamic refinement' >&2; exit 2; }

# When the dynamic winner already uses compact prectx in both directions, the
# flat-bid metadata can remove its per-orbit block-id binary search without
# changing any other scheduling or closure choice. Race plain/bid/fused with
# the selected dynamic lease batch fixed.
if [[ "$RUN_ORBIT_DYNAMIC_BID" == 1 && "$DYNAMIC_SELECTED" == 1 && "${ORBIT_PRECTX_FORWARD:-0}" == 1 && "${ORBIT_PRECTX_REVERSE:-0}" == 1 && "${ORBIT_PRECTX_COMPACT:-0}" == 1 ]]; then
  echo "=== HBM tune21 dynamic compact flat-bid refinement ===" >&2
  PROFILE_IN="$DYNAMIC_PROFILE" PROFILE_OUT="$DYNAMIC_BID_PROFILE" PREFIX="$DYNAMIC_BID_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-bid21.sh"
else
  if [[ "$DYNAMIC_SELECTED" == 1 && "$RUN_ORBIT_DYNAMIC_BID" == 1 ]]; then
    echo "=== skip dynamic flat-bid: dynamic winner lacks compact forward+reverse prectx ===" >&2
  fi
  cp "$DYNAMIC_PROFILE" "$DYNAMIC_BID_PROFILE"
fi

ORBITCTA_FLAT_DYNAMIC=0
# shellcheck disable=SC1090
source "$DYNAMIC_BID_PROFILE"
DYNAMIC_SELECTED="${ORBITCTA_FLAT_DYNAMIC:-0}"

# The dynamic queue has one CTA barrier between acquiring a lease and preparing
# its first orbit. The fused variant lets lane 0 acquire and prepare before the
# same publish barrier; compare only this synchronization change with every
# selected dynamic knob fixed.
if [[ "$RUN_ORBIT_DYNAMIC_FUSE" == 1 && "$DYNAMIC_SELECTED" == 1 ]]; then
  echo "=== HBM tune21 dynamic lease/first-prepare fusion refinement ===" >&2
  PROFILE_IN="$DYNAMIC_BID_PROFILE" PROFILE_OUT="$DYNAMIC_FUSE_PROFILE" PREFIX="$DYNAMIC_FUSE_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-fuse21.sh"
else
  cp "$DYNAMIC_BID_PROFILE" "$DYNAMIC_FUSE_PROFILE"
fi

ORBITCTA_FLAT_DYNAMIC=0 ORBITCTA_FLAT_DYNAMIC_BATCH=1
# shellcheck disable=SC1090
source "$DYNAMIC_FUSE_PROFILE"
DYNAMIC_SELECTED="${ORBITCTA_FLAT_DYNAMIC:-0}"
DYNAMIC_BATCH_SELECTED="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"

# A fixed large lease batch can underfill the persistent pool on small HIGH
# positions. When batch>1, tune how many lease waves must remain available;
# waves=0 is the exact fixed-batch baseline.
if [[ "$RUN_ORBIT_DYNAMIC_ADAPTIVE" == 1 && "$DYNAMIC_SELECTED" == 1 && "$DYNAMIC_BATCH_SELECTED" != 1 ]]; then
  echo "=== HBM tune21 adaptive dynamic lease refinement ===" >&2
  PROFILE_IN="$DYNAMIC_FUSE_PROFILE" PROFILE_OUT="$DYNAMIC_ADAPTIVE_PROFILE" PREFIX="$DYNAMIC_ADAPTIVE_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-adaptive21.sh"
else
  [[ "$DYNAMIC_SELECTED" == 0 || "$DYNAMIC_BATCH_SELECTED" != 1 || "$RUN_ORBIT_DYNAMIC_ADAPTIVE" == 0 ]] || echo "=== skip adaptive dynamic lease: selected batch=1 ===" >&2
  cp "$DYNAMIC_FUSE_PROFILE" "$DYNAMIC_ADAPTIVE_PROFILE"
fi

ORBITCTA_FLAT_DYNAMIC=0
# shellcheck disable=SC1090
source "$DYNAMIC_ADAPTIVE_PROFILE"
DYNAMIC_SELECTED="${ORBITCTA_FLAT_DYNAMIC:-0}"

# Warp-cooperative compact prectx is a chunked-QOL branch. It cannot compose
# with the dynamic queue, so skip this entire family when dynamic won.
if [[ "$RUN_ORBIT_WARPCOOP" == 1 && "$RUN_ORBIT_ADVANCED" == 1 && "$DYNAMIC_SELECTED" == 0 ]]; then
  echo "=== HBM tune21 warp-cooperative compact prectx refinement ===" >&2
  PROFILE_IN="$DYNAMIC_ADAPTIVE_PROFILE" PROFILE_OUT="$WARPCOOP_PROFILE" PREFIX="$WARPCOOP_PREFIX" \
    QUAD_WINNER_ENV="${ADV_PREFIX}_quad_winner.env" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-warpcoop21.sh"
else
  [[ "$DYNAMIC_SELECTED" == 0 ]] || echo "=== skip warpcoop refine: dynamic queue selected ===" >&2
  cp "$DYNAMIC_ADAPTIVE_PROFILE" "$WARPCOOP_PROFILE"
fi

# A changed register footprint can change the occupancy-derived persistent pool.
if [[ "$RUN_ORBIT_WARPCOOP_AUTO" == 1 && "$RUN_ORBIT_ADVANCED" == 1 && "$DYNAMIC_SELECTED" == 0 ]]; then
  echo "=== HBM tune21 warpcoop occupancy-pool refinement ===" >&2
  PROFILE_IN="$WARPCOOP_PROFILE" PROFILE_OUT="$WARPCOOP_AUTO_PROFILE" PREFIX="$WARPCOOP_AUTO_PREFIX" \
    QUAD_WINNER_ENV="${ADV_PREFIX}_quad_winner.env" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-warpcoop-auto21.sh"
else
  cp "$WARPCOOP_PROFILE" "$WARPCOOP_AUTO_PROFILE"
fi

# Descriptor MLP only applies to sparse64 chunked-quad overlap-local.
if [[ "$RUN_ORBIT_QUAD_DESC" == 1 ]]; then
  ORBIT_QUAD_MLP=0 ORBIT_QUAD_OVERLAP_LOCAL=0 ORBIT_SPARSE64=0
  # shellcheck disable=SC1090
  source "$WARPCOOP_AUTO_PROFILE"
  if [[ "${ORBIT_QUAD_MLP:-0}" == 1 && "${ORBIT_QUAD_OVERLAP_LOCAL:-0}" == 1 && "${ORBIT_SPARSE64:-0}" == 1 ]]; then
    echo "=== HBM tune21 quad sparse descriptor MLP refinement ===" >&2
    PROFILE_IN="$WARPCOOP_AUTO_PROFILE" PROFILE_OUT="$DESC_PROFILE" PREFIX="$QUAD_DESC_PREFIX" \
      bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-quad-desc21.sh"
  else
    echo "=== skip quad sparse descriptor MLP: selected orbit path is not sparse QOL ===" >&2
    cp "$WARPCOOP_AUTO_PROFILE" "$DESC_PROFILE"
  fi
else
  cp "$WARPCOOP_AUTO_PROFILE" "$DESC_PROFILE"
fi

# QOL has nothing useful to overlap when local_n==0.
if [[ "$RUN_ORBIT_QUAD_LOCAL0" == 1 ]]; then
  ORBIT_QUAD_MLP=0 ORBIT_QUAD_OVERLAP_LOCAL=0
  # shellcheck disable=SC1090
  source "$DESC_PROFILE"
  if [[ "${ORBIT_QUAD_MLP:-0}" == 1 && "${ORBIT_QUAD_OVERLAP_LOCAL:-0}" == 1 ]]; then
    echo "=== HBM tune21 QOL local_n=0 bypass refinement ===" >&2
    PROFILE_IN="$DESC_PROFILE" PROFILE_OUT="$LOCAL0_PROFILE" PREFIX="$QUAD_LOCAL0_PREFIX" \
      bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-quad-local0-21.sh"
  else
    echo "=== skip QOL local_n=0 bypass: selected orbit path is not QOL ===" >&2
    cp "$DESC_PROFILE" "$LOCAL0_PROFILE"
  fi
else
  cp "$DESC_PROFILE" "$LOCAL0_PROFILE"
fi

# Tune cp.async commit_group granularity 1/2/4 columns per group.
if [[ "$RUN_ORBIT_QUAD_GROUP" == 1 ]]; then
  ORBIT_QUAD_MLP=0 ORBIT_QUAD_OVERLAP_LOCAL=0
  # shellcheck disable=SC1090
  source "$LOCAL0_PROFILE"
  if [[ "${ORBIT_QUAD_MLP:-0}" == 1 && "${ORBIT_QUAD_OVERLAP_LOCAL:-0}" == 1 ]]; then
    echo "=== HBM tune21 QOL cp.async commit-group refinement ===" >&2
    PROFILE_IN="$LOCAL0_PROFILE" PROFILE_OUT="$GROUP_PROFILE" PREFIX="$QUAD_GROUP_PREFIX" \
      bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-quad-group21.sh"
  else
    echo "=== skip QOL cp.async group refinement: selected orbit path is not QOL ===" >&2
    cp "$LOCAL0_PROFILE" "$GROUP_PROFILE"
  fi
else
  cp "$LOCAL0_PROFILE" "$GROUP_PROFILE"
fi

# Finally tune the L2 prefetch-size hint while preserving all selected QOL knobs.
if [[ "$RUN_ORBIT_QUAD_PREFETCH" == 1 ]]; then
  ORBIT_QUAD_MLP=0 ORBIT_QUAD_OVERLAP_LOCAL=0
  # shellcheck disable=SC1090
  source "$GROUP_PROFILE"
  if [[ "${ORBIT_QUAD_MLP:-0}" == 1 && "${ORBIT_QUAD_OVERLAP_LOCAL:-0}" == 1 ]]; then
    echo "=== HBM tune21 QOL cp.async L2 prefetch refinement ===" >&2
    PROFILE_IN="$GROUP_PROFILE" PROFILE_OUT="$FINAL_PROFILE" PREFIX="$QUAD_PREFETCH_PREFIX" \
      bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-quad-prefetch21.sh"
  else
    echo "=== skip QOL cp.async L2 prefetch: selected orbit path is not QOL ===" >&2
    cp "$GROUP_PROFILE" "$FINAL_PROFILE"
  fi
else
  cp "$GROUP_PROFILE" "$FINAL_PROFILE"
fi

echo "=== final HBM profile ===" >&2
cat "$FINAL_PROFILE"
echo "b300 HBM profile auto21 OK base_profile=$BASE_PROFILE prectx_profile=$PRECTX_PROFILE sched_profile=$SCHED_PROFILE advanced_profile=$ADV_PROFILE dynamic_profile=$DYNAMIC_PROFILE dynamic_bid_profile=$DYNAMIC_BID_PROFILE dynamic_fuse_profile=$DYNAMIC_FUSE_PROFILE dynamic_adaptive_profile=$DYNAMIC_ADAPTIVE_PROFILE dynamic_selected=$DYNAMIC_SELECTED warpcoop_profile=$WARPCOOP_PROFILE warpcoop_auto_profile=$WARPCOOP_AUTO_PROFILE desc_profile=$DESC_PROFILE local0_profile=$LOCAL0_PROFILE group_profile=$GROUP_PROFILE final_profile=$FINAL_PROFILE" >&2
