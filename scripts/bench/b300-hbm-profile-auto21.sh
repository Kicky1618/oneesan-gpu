#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_PROFILE="${BASE_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
FINAL_PROFILE="${FINAL_PROFILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
BASE_PREFIX="${BASE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21}"
REFINE_PREFIX="${REFINE_PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_refine_prectx21}"

echo "=== HBM tune21 base search ===" >&2
PROFILE_OUT="$BASE_PROFILE" PREFIX="$BASE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-tune21.sh"

echo "=== HBM tune21 prectx refinement ===" >&2
PROFILE_IN="$BASE_PROFILE" PROFILE_OUT="$FINAL_PROFILE" PREFIX="$REFINE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-prectx21.sh"

echo "=== final HBM profile ===" >&2
cat "$FINAL_PROFILE"
echo "b300 HBM profile auto21 OK final_profile=$FINAL_PROFILE" >&2
