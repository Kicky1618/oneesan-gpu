#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_evict_staged}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"; mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
[[ -s "$INPUT_ENV" ]] || exit 2
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1])<1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
# shellcheck disable=SC1090
source "$INPUT_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || exit 4
W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"; D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"
stage_validate_rows=()
for rows in $VALIDATE_ROWS "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do [[ "$rows" == "$SEARCH_ROWS" ]] && continue; [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || exit 2; seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || stage_validate_rows+=("$rows"); done
check_known_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then echo 'FATAL Stage-I/core residue mismatch' >&2; exit 4; fi; if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then echo 'FATAL Stage-I/Stage-E-final residue mismatch' >&2; exit 4; fi; if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE" ]]; then echo 'FATAL Stage-I/Stage-F-final residue mismatch' >&2; exit 4; fi; }
run_stage(){ local rows="$1" threads="$2" repeats="$3" tag="$4" evicts="$5"; local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"; INPUT_ENV="$INPUT_ENV" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$threads" EVICT_LIST="$evicts" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-evict-sweep.sh" | tee "$log" >&2; grep -Fq 'b300_evict_exact_match=1' "$log" || exit 4; [[ -s "$env" ]] || exit 4; printf '%s\n' "$env"; }
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1])>=float(sys.argv[2]) else 0)
PY
}
SEARCH_ENV="$(run_stage "$SEARCH_ROWS" '128 256 512' "$SEARCH_REPEATS" search 'default normal last')"
# shellcheck disable=SC1090
source "$SEARCH_ENV"; check_known_residue "$SEARCH_ROWS" "$B300_EVICT_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"; SELECTED_HINT="$B300_EVICT_HINT"
case "$SELECTED_HINT" in normal|last) if [[ "$B300_EVICT_BEST_ENABLED" == 1 && "$B300_EVICT_DEFAULT_SPILL_FREE" == 1 && "$B300_EVICT_SPILL_FREE" == 1 && "$(passes "$B300_EVICT_SPEEDUP")" == 1 ]]; then VALIDATED=1; fi;; *) VALIDATED=0;; esac
if [[ "$VALIDATED" == 1 ]]; then
  default_threads="$B300_EVICT_DEFAULT_THREADS"; hint_threads="$B300_EVICT_THREADS"; validation_threads="$default_threads"; [[ "$hint_threads" == "$default_threads" ]] || validation_threads+=" $hint_threads"; stage=0
  for rows in "${stage_validate_rows[@]}"; do ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}" "default $SELECTED_HINT")"; source "$CURRENT_ENV"; check_known_residue "$rows" "$B300_EVICT_RESIDUE"; if [[ "$B300_EVICT_HINT" != "$SELECTED_HINT" || "$B300_EVICT_BEST_ENABLED" != 1 || "$B300_EVICT_DEFAULT_SPILL_FREE" != 1 || "$B300_EVICT_SPILL_FREE" != 1 || "$(passes "$B300_EVICT_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi; default_threads="$B300_EVICT_DEFAULT_THREADS"; hint_threads="$B300_EVICT_THREADS"; validation_threads="$default_threads"; [[ "$hint_threads" == "$default_threads" ]] || validation_threads+=" $hint_threads"; done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then FINAL_HINT="$SELECTED_HINT"; FINAL_BIN="$B300_EVICT_BIN"; FINAL_THREADS="$B300_EVICT_THREADS"; FINAL_WALL="$B300_EVICT_WALL_S"; FINAL_HIGH="$B300_EVICT_HIGH_S"; FINAL_SPEED="$B300_EVICT_SPEEDUP"; else FINAL_HINT=default; FINAL_BIN="$B300_EVICT_DEFAULT_BIN"; FINAL_THREADS="$B300_EVICT_DEFAULT_THREADS"; FINAL_WALL="$B300_EVICT_DEFAULT_WALL_S"; FINAL_HIGH="$B300_EVICT_DEFAULT_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
 printf 'B300_STAGEI_STAGED_VALIDATED=%q\n' "$VALIDATED"
 printf 'B300_STAGEI_FINAL_ENABLED=%q\n' "$VALIDATED"
 printf 'B300_STAGEI_FINAL_WIDTH=%q\n' "$W"
 printf 'B300_STAGEI_FINAL_DISTANCE=%q\n' "$D"
 printf 'B300_STAGEI_FINAL_HINT=%q\n' "$FINAL_HINT"
 printf 'B300_STAGEI_FINAL_BIN=%q\n' "$FINAL_BIN"
 printf 'B300_STAGEI_FINAL_THREADS=%q\n' "$FINAL_THREADS"
 printf 'B300_STAGEI_FINAL_WALL_S=%q\n' "$FINAL_WALL"
 printf 'B300_STAGEI_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"
 printf 'B300_STAGEI_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"
 printf 'B300_STAGEI_FINAL_SPILL_FREE=1\n'
 printf 'B300_STAGEI_CONTROL_BIN=%q\n' "$B300_EVICT_DEFAULT_BIN"
 printf 'B300_STAGEI_CONTROL_THREADS=%q\n' "$B300_EVICT_DEFAULT_THREADS"
 printf 'B300_STAGEI_FINAL_STAGE_ROWS=%q\n' "$B300_EVICT_ROWS"
 printf 'B300_STAGEI_FINAL_STAGE_RESIDUE=%q\n' "$B300_EVICT_RESIDUE"
 printf 'B300_STAGEI_INPUT_STAGE_F_ENV=%q\n' "$INPUT_ENV"
 printf 'B300_STAGEI_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"; echo "b300-nextgen-hybrid8-nextself-evict-staged-calibrate OK validated=$VALIDATED hint=$FINAL_HINT width=$W distance=$D speedup=$FINAL_SPEED" >&2
