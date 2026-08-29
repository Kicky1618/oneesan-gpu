#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SELECTED_ENV="${SELECTED_ENV:-$ONEESAN_ROOT/work/b300_grand_stageg_firstpass_n27.selected.env}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing selected env=$SELECTED_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$SELECTED_ENV"

if [[ "${B300_GRAND_STAGEG_SELECTED_VALIDATED:-0}" == 1 ]]; then
  BIN="$B300_GRAND_STAGEG_SELECTED_BINARY"; BIN_SHA="$B300_GRAND_STAGEG_SELECTED_BINARY_SHA256"; RUNTIME="$B300_GRAND_STAGEG_SELECTED_RUNTIME_KIND"
  RUN_THREADS="$B300_GRAND_STAGEG_SELECTED_THREADS"; TARGET="$B300_GRAND_STAGEG_SELECTED_TARGET_MIB"; WORK="$B300_GRAND_STAGEG_SELECTED_WORK_DIR"; CHECKPOINT="$B300_GRAND_STAGEG_SELECTED_CHECKPOINT"
  BASE_ENV="$B300_GRAND_STAGEG_BASE_SELECTED_ENV"
  [[ -s "$BASE_ENV" ]] || { echo 'Stage-G base selected env missing' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$BASE_ENV"
elif [[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]]; then
  BIN="$B300_GRAND_SELECTED_BINARY"; BIN_SHA="$B300_GRAND_SELECTED_BINARY_SHA256"; RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"
  RUN_THREADS="$B300_GRAND_SELECTED_THREADS"; TARGET="$B300_GRAND_SELECTED_TARGET_MIB"; WORK="$B300_GRAND_SELECTED_WORK_DIR"; CHECKPOINT="$B300_GRAND_SELECTED_CHECKPOINT"
else
  echo 'selected env has no validated grand selection' >&2; exit 3
fi

PROFILE_FILE="${PROFILE_FILE:-${B300_GRAND_SELECTED_PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}}"
MAX_WINDOW="${MAX_WINDOW:-${B300_GRAND_SELECTED_MAX_WINDOW:-14}}"
[[ -x "$BIN" && -f "$PROFILE_FILE" && -d "$WORK" && -s "$CHECKPOINT" ]] || { echo 'selected artifact missing' >&2; exit 3; }
NOW_SHA="$(sha256sum "$BIN" | awk '{print $1}')"; [[ "$NOW_SHA" == "$BIN_SHA" ]] || { echo 'selected binary SHA changed' >&2; exit 3; }
[[ "$MAX_WINDOW" =~ ^[1-9][0-9]*$ && "$TARGET" =~ ^[1-9][0-9]*$ ]] || exit 3

# Recreate only the runtime launch environment; the schema-3 checkpoint already
# pins the binary and profile hashes and solve_b300_exact_batch.py verifies it.
case "$RUNTIME" in
  forced)
    [[ "$RUN_THREADS" =~ ^[0-9]+$ ]] && ((RUN_THREADS>=32&&RUN_THREADS<=1024&&RUN_THREADS%32==0)) || exit 3
    export GRIDFP_THREADS="$RUN_THREADS"
    ;;
  warp|orbit)
    # shellcheck disable=SC1090
    source "$PROFILE_FILE"
    THREADS="${BUCKET_THREADS:-256}"; WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
    if [[ "$RUNTIME" == warp ]]; then
      export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"
    else
      export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"
      unset BUCKET_ORBITCTA_FLAT_BLOCKS
      if [[ "${ORBITCTA_FLAT:-0}" == 1 && "${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}" != 0 ]]; then export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM"; else unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM; fi
    fi
    ;;
  *) echo "bad selected runtime=$RUNTIME" >&2; exit 3;;
esac

echo "=== continue exact n=27 runtime=$RUNTIME binary=$(basename "$BIN") work=$WORK cached_checkpoint=$CHECKPOINT ===" >&2
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BIN" --target-mib "$TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$WORK" "$@"
