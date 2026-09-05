#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"; BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageo-cg-l2-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$SWEEP"; do [[ -s "$f" ]] || { echo "missing Stage-O staged input=$f" >&2; exit 2; }; done
[[ "$SEARCH_ROWS" =~ ^[1-9][0-9]*$ ]] && ((SEARCH_ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for x in SEARCH_REPEATS VALIDATE_REPEATS; do [[ "${!x}" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

# Pin Stage-N identity and its exact final residue before any Stage-O search.
# shellcheck disable=SC1090
source "$STAGEN_WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_NGPU B300_STAGEN_PAIR_POLICY B300_STAGEN_BLOCK_POLICY B300_STAGEN_MATE_LOAD_POLICY B300_STAGEN_FINAL_BIN B300_STAGEN_FINAL_THREADS B300_STAGEN_FINAL_SPILL_FREE B300_STAGEN_FINAL_STAGE_ROWS B300_STAGEN_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-N winner missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_FINAL_SPILL_FREE" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || exit 4
[[ "$B300_STAGEN_PAIR_POLICY" == cg || "$B300_STAGEN_BLOCK_POLICY" == cg ]] || { echo 'Stage O not applicable: Stage N has no CG Count axis' >&2; exit 4; }
N_ROWS="$B300_STAGEN_FINAL_STAGE_ROWS"; N_RES="$B300_STAGEN_FINAL_STAGE_RESIDUE"; N_BIN="$B300_STAGEN_FINAL_BIN"
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
[[ "${B300_STAGEN_PREPARED:-0}" == 1 && "${B300_STAGEN_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEN_PREPARED_NGPU:-0}" == "$NGPU" && "$B300_STAGEN_PREPARED_BIN" == "$N_BIN" ]] || { echo 'Stage-O staged Stage-N prepare drift' >&2; exit 3; }
[[ -s "${B300_STAGEN_PREPARED_MANIFEST:-}" ]] || exit 3
sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-N manifest mismatch before Stage-O staged search' >&2; exit 3; }

passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" pair_list="$2" block_list="$3" threads="$4" repeats="$5" tag="$6"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" \
    ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" \
    PAIR_L2_LIST="$pair_list" BLOCK_L2_LIST="$block_list" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" \
    bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stageo_exact_match=1' "$log" || { echo "Stage-O exact gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$envf" ]] || exit 4
  printf '%s\n' "$envf"
}
check_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$N_ROWS" && "$got" != "$N_RES" ]]; then echo "FATAL Stage-O/Stage-N residue mismatch rows=$rows got=$got expected=$N_RES" >&2; exit 4; fi; }

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$PAIR_L2_LIST" "$BLOCK_L2_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"; check_residue "$SEARCH_ROWS" "$B300_STAGEO_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"; SELECTED_PAIR_L2="$B300_STAGEO_PAIR_L2_BYTES"; SELECTED_BLOCK_L2="$B300_STAGEO_BLOCK_L2_BYTES"
BASE_PAIR_L2="$B300_STAGEO_BASE_PAIR_L2_BYTES"; BASE_BLOCK_L2="$B300_STAGEO_BASE_BLOCK_L2_BYTES"
if [[ "$B300_STAGEO_BEST_ENABLED" == 1 && "$B300_STAGEO_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEO_SPILL_FREE" == 1 && "$(passes "$B300_STAGEO_SPEEDUP")" == 1 ]]; then
  VALIDATED=1
  control_threads="$B300_STAGEO_CONTROL_THREADS"; test_threads="$B300_STAGEO_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  pair_validation="$BASE_PAIR_L2"; [[ "$SELECTED_PAIR_L2" == "$BASE_PAIR_L2" ]] || pair_validation+=" $SELECTED_PAIR_L2"
  block_validation="$BASE_BLOCK_L2"; [[ "$SELECTED_BLOCK_L2" == "$BASE_BLOCK_L2" ]] || block_validation+=" $SELECTED_BLOCK_L2"
  validate_rows=()
  for rows in $VALIDATE_ROWS "$N_ROWS"; do
    [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue; [[ "$rows" == "$SEARCH_ROWS" ]] && continue
    seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || validate_rows+=("$rows")
  done
  stage=0
  for rows in "${validate_rows[@]}"; do
    ((stage+=1))
    CURRENT_ENV="$(run_stage "$rows" "$pair_validation" "$block_validation" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_residue "$rows" "$B300_STAGEO_RESIDUE"
    if [[ "$B300_STAGEO_PAIR_L2_BYTES" != "$SELECTED_PAIR_L2" || "$B300_STAGEO_BLOCK_L2_BYTES" != "$SELECTED_BLOCK_L2" || "$B300_STAGEO_BEST_ENABLED" != 1 || "$B300_STAGEO_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEO_SPILL_FREE" != 1 || "$(passes "$B300_STAGEO_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    control_threads="$B300_STAGEO_CONTROL_THREADS"; test_threads="$B300_STAGEO_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_PAIR_L2="$B300_STAGEO_PAIR_L2_BYTES"; FINAL_BLOCK_L2="$B300_STAGEO_BLOCK_L2_BYTES"; FINAL_BIN="$B300_STAGEO_BIN"; FINAL_THREADS="$B300_STAGEO_THREADS"; FINAL_WALL="$B300_STAGEO_WALL_S"; FINAL_HIGH="$B300_STAGEO_HIGH_S"; FINAL_SPEED="$B300_STAGEO_SPEEDUP"
else
  FINAL_PAIR_L2="$B300_STAGEO_BASE_PAIR_L2_BYTES"; FINAL_BLOCK_L2="$B300_STAGEO_BASE_BLOCK_L2_BYTES"; FINAL_BIN="$B300_STAGEO_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEO_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEO_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEO_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000
fi
{
  printf 'B300_STAGEO_STAGED_VALIDATED=%q\n' "$VALIDATED"; printf 'B300_STAGEO_FINAL_ENABLED=%q\n' "$VALIDATED"; printf 'B300_STAGEO_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEO_PAIR_POLICY=%q\n' "$B300_STAGEO_PAIR_POLICY"; printf 'B300_STAGEO_BLOCK_POLICY=%q\n' "$B300_STAGEO_BLOCK_POLICY"; printf 'B300_STAGEO_MATE_LOAD_POLICY=%q\n' "$B300_STAGEO_MATE_LOAD_POLICY"
  printf 'B300_STAGEO_BASE_CG_L2_BYTES=%q\n' "$B300_STAGEO_BASE_CG_L2_BYTES"; printf 'B300_STAGEO_BASE_PAIR_L2_BYTES=%q\n' "$B300_STAGEO_BASE_PAIR_L2_BYTES"; printf 'B300_STAGEO_BASE_BLOCK_L2_BYTES=%q\n' "$B300_STAGEO_BASE_BLOCK_L2_BYTES"
  printf 'B300_STAGEO_PAIR_L2_BYTES=%q\n' "$FINAL_PAIR_L2"; printf 'B300_STAGEO_BLOCK_L2_BYTES=%q\n' "$FINAL_BLOCK_L2"; printf 'B300_STAGEO_FINAL_BIN=%q\n' "$FINAL_BIN"; printf 'B300_STAGEO_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_STAGEO_FINAL_WALL_S=%q\n' "$FINAL_WALL"; printf 'B300_STAGEO_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"; printf 'B300_STAGEO_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"; printf 'B300_STAGEO_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEO_CONTROL_BIN=%q\n' "$B300_STAGEO_CONTROL_BIN"; printf 'B300_STAGEO_CONTROL_THREADS=%q\n' "$B300_STAGEO_CONTROL_THREADS"
  printf 'B300_STAGEO_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEO_ROWS"; printf 'B300_STAGEO_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEO_RESIDUE"
  printf 'B300_STAGEO_SEARCH_PAIR_L2=%q\n' "$PAIR_L2_LIST"; printf 'B300_STAGEO_SEARCH_BLOCK_L2=%q\n' "$BLOCK_L2_LIST"; printf 'B300_STAGEO_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  printf 'B300_STAGEO_INPUT_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"; printf 'B300_STAGEO_INPUT_STAGEN_WINNER_ENV=%q\n' "$STAGEN_WINNER_ENV"; printf 'B300_STAGEO_INPUT_STAGEN_PREPARE_ENV=%q\n' "$STAGEN_PREPARE_ENV"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-stageo-cg-l2-staged-calibrate OK stage=O validated=$VALIDATED pair_l2=$FINAL_PAIR_L2 block_l2=$FINAL_BLOCK_L2 speedup=${FINAL_SPEED}x final_rows=$B300_STAGEO_ROWS ngpu=$NGPU" >&2
