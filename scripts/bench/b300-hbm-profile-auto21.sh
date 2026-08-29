#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_PROFILE="${BASE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
FINAL_PROFILE="${FINAL_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
BASE_PREFIX="${BASE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21}"
REFINE_PREFIX="${REFINE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_refine_compact_prectx21}"
RUN_PRECTX="${RUN_PRECTX:-1}"
[[ "$RUN_PRECTX" == 0 || "$RUN_PRECTX" == 1 ]] || { echo 'RUN_PRECTX must be 0 or 1' >&2; exit 2; }

if [[ "$RUN_PRECTX" == 1 ]]; then
  echo "=== compact HIGH prectx CUDA selftest ===" >&2
  ARCH="${ARCH:-native}" PM_ACCUM="${PM_ACCUM:-1}" \
    bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh"
fi

echo "=== HBM tune21 base search ===" >&2
RUN_PRECTX="$RUN_PRECTX" PROFILE_OUT="$BASE_PROFILE" PREFIX="$BASE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-tune21.sh"

if [[ "$RUN_PRECTX" == 1 ]]; then
  echo "=== HBM tune21 pointer/compact prectx refinement ===" >&2
  PROFILE_IN="$BASE_PROFILE" PROFILE_OUT="$FINAL_PROFILE" PREFIX="$REFINE_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-compact-prectx21.sh"
else
  cp "$BASE_PROFILE" "$FINAL_PROFILE"
fi

echo "=== final HBM profile ===" >&2
cat "$FINAL_PROFILE"
echo "b300 HBM profile auto21 OK final_profile=$FINAL_PROFILE" >&2
