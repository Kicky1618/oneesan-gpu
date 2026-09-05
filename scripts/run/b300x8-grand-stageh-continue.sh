#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SELECTED_ENV="${SELECTED_ENV:-$ONEESAN_ROOT/work/b300_grand_stageh_firstpass_n27.selected.env}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing selected env=$SELECTED_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 && "${B300_GRAND_SELECTED_N:-0}" == 27 ]] || { echo 'selected env has no normalized validated n=27 selection' >&2; exit 3; }
for k in B300_GRAND_SELECTED_PROFILE_FILE B300_GRAND_SELECTED_PROFILE_SHA256 B300_GRAND_SELECTED_BINARY B300_GRAND_SELECTED_BINARY_SHA256 \
  B300_GRAND_SELECTED_RUNTIME_KIND B300_GRAND_SELECTED_THREADS B300_GRAND_SELECTED_TARGET_MIB B300_GRAND_SELECTED_MAX_WINDOW \
  B300_GRAND_SELECTED_WORK_DIR B300_GRAND_SELECTED_CHECKPOINT B300_GRAND_SELECTED_SMOKE_PRIME B300_GRAND_SELECTED_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "selected env missing $k" >&2; exit 3; }
done
BIN="$B300_GRAND_SELECTED_BINARY"; BIN_SHA="$B300_GRAND_SELECTED_BINARY_SHA256"; RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; RUN_THREADS="$B300_GRAND_SELECTED_THREADS"; TARGET="$B300_GRAND_SELECTED_TARGET_MIB"; WORK="$B300_GRAND_SELECTED_WORK_DIR"; CHECKPOINT="$B300_GRAND_SELECTED_CHECKPOINT"
PROFILE_FILE="${PROFILE_FILE:-$B300_GRAND_SELECTED_PROFILE_FILE}"; MAX_WINDOW="${MAX_WINDOW:-$B300_GRAND_SELECTED_MAX_WINDOW}"
[[ -x "$BIN" && -f "$PROFILE_FILE" && -d "$WORK" && -s "$CHECKPOINT" ]] || { echo 'selected artifact missing' >&2; exit 3; }
NOW_SHA="$(sha256sum "$BIN" | awk '{print $1}')"; [[ "$NOW_SHA" == "$BIN_SHA" ]] || { echo 'selected binary SHA changed' >&2; exit 3; }
PROFILE_SHA="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"; [[ "$PROFILE_SHA" == "$B300_GRAND_SELECTED_PROFILE_SHA256" ]] || { echo 'selected profile SHA changed' >&2; exit 3; }
case "$RUNTIME" in
 forced) [[ "$RUN_THREADS" =~ ^[0-9]+$ ]] && ((RUN_THREADS>=32&&RUN_THREADS<=1024&&RUN_THREADS%32==0)) || exit 3; export GRIDFP_THREADS="$RUN_THREADS" ;;
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
python3 - "$CHECKPOINT" "$BIN_SHA" "$PROFILE_SHA" "$B300_GRAND_SELECTED_SMOKE_PRIME" "$B300_GRAND_SELECTED_RESIDUE" <<'PY'
import json,sys
cp,bsha,psha,prime,residue=sys.argv[1:]
d=json.load(open(cp)); fp=d.get('solver_fingerprint',{})
if int(d.get('n',-1))!=27 or fp != {'schema':3,'binary_sha256':bsha,'profile_sha256':psha}: raise SystemExit('selected schema-3 checkpoint fingerprint mismatch')
r=d.get('residues',{}).get(str(int(prime)))
if not r or int(r.get('residue',-1)) != int(residue): raise SystemExit('selected smoke residue mismatch')
PY
echo "=== continue Stage-H exact n=27 runtime=$RUNTIME binary=$(basename "$BIN") work=$WORK ===" >&2
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BIN" --target-mib "$TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$WORK" "$@"
