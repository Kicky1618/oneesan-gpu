#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEL_GUARD_ENV="${STAGEL_GUARD_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g${NGPU}_winner.env}"
STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_g${NGPU}_winner.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-pair-block-load-policy-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$STAGEL_GUARD_ENV" "$SWEEP"; do [[ -s "$f" ]] || { echo "missing Stage-N staged input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagel|stagem) ;; *) exit 2;; esac
for x in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
[[ "$SEARCH_ROWS" =~ ^[1-9][0-9]*$ ]] && ((SEARCH_ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

# Resolve the upstream exactly once. Stage N never changes geometry, guards,
# mate-load policy, or upstream identity while validating row slices.
# shellcheck disable=SC1090
source "$STAGEL_GUARD_ENV"
for k in B300_STAGEL_NGPU B300_STAGEL_FINAL_SPILL_FREE B300_STAGEL_FINAL_STAGE_ROWS B300_STAGEL_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || exit 3; done
[[ "$B300_STAGEL_NGPU" == "$NGPU" && "$B300_STAGEL_FINAL_SPILL_FREE" == 1 ]] || exit 4
L_ROWS="$B300_STAGEL_FINAL_STAGE_ROWS"; L_RES="$B300_STAGEL_FINAL_STAGE_RESIDUE"; RESOLVED_UPSTREAM=stagel; U_ROWS="$L_ROWS"; U_RES="$L_RES"
if [[ "$UPSTREAM_KIND" == stagem || ( "$UPSTREAM_KIND" == auto && -s "$STAGEM_WINNER_ENV" && "$(grep -c '^B300_STAGEM_STAGED_VALIDATED=1$' "$STAGEM_WINNER_ENV" || true)" != 0 && "$(grep -c '^B300_STAGEM_FINAL_ENABLED=1$' "$STAGEM_WINNER_ENV" || true)" != 0 ) ]]; then
  [[ -s "$STAGEM_WINNER_ENV" ]] || exit 3
  # shellcheck disable=SC1090
  source "$STAGEM_WINNER_ENV"
  [[ "${B300_STAGEM_STAGED_VALIDATED:-0}" == 1 && "${B300_STAGEM_FINAL_ENABLED:-0}" == 1 && "${B300_STAGEM_FINAL_SPILL_FREE:-0}" == 1 && "${B300_STAGEM_NGPU:-0}" == "$NGPU" ]] || exit 4
  RESOLVED_UPSTREAM=stagem; U_ROWS="$B300_STAGEM_FINAL_STAGE_ROWS"; U_RES="$B300_STAGEM_FINAL_STAGE_RESIDUE"
fi
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
F_ROWS="${B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS:-0}"; F_RES="${B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE:-}"; E_ROWS="${B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS:-0}"; E_RES="${B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE:-}"

check_known_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$U_ROWS" && "$got" != "$U_RES" ]]; then echo "FATAL Stage-N/upstream residue mismatch rows=$rows got=$got expected=$U_RES" >&2; exit 4; fi; if [[ "$rows" == "$L_ROWS" && "$got" != "$L_RES" ]]; then echo "FATAL Stage-N/Stage-L residue mismatch rows=$rows got=$got expected=$L_RES" >&2; exit 4; fi; if [[ "$rows" == "$F_ROWS" && -n "$F_RES" && "$got" != "$F_RES" ]]; then echo "FATAL Stage-N/Stage-F residue mismatch rows=$rows got=$got expected=$F_RES" >&2; exit 4; fi; if [[ "$rows" == "$E_ROWS" && -n "$E_RES" && "$got" != "$E_RES" ]]; then echo "FATAL Stage-N/Stage-E residue mismatch rows=$rows got=$got expected=$E_RES" >&2; exit 4; fi; }
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){ local rows="$1" pairlist="$2" blocklist="$3" threads="$4" repeats="$5" tag="$6"; local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"; STAGE_F_ENV="$STAGE_F_ENV" STAGEL_GUARD_ENV="$STAGEL_GUARD_ENV" STAGEM_WINNER_ENV="$STAGEM_WINNER_ENV" UPSTREAM_KIND="$RESOLVED_UPSTREAM" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" PAIR_POLICY_LIST="$pairlist" BLOCK_POLICY_LIST="$blocklist" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" bash "$SWEEP" | tee "$log" >&2; grep -Fq 'b300_stagen_exact_match=1' "$log" || { echo "Stage-N exact gate missing rows=$rows" >&2; exit 4; }; [[ -s "$envf" ]] || exit 4; printf '%s\n' "$envf"; }
check_config(){ [[ "$B300_STAGEN_UPSTREAM_KIND" == "$RESOLVED_UPSTREAM" && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage-N upstream/GPU drift' >&2; exit 4; }; }

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$PAIR_POLICY_LIST" "$BLOCK_POLICY_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"; check_config; check_known_residue "$SEARCH_ROWS" "$B300_STAGEN_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"; SELECTED_PAIR="$B300_STAGEN_PAIR_POLICY"; SELECTED_BLOCK="$B300_STAGEN_BLOCK_POLICY"
if [[ "$B300_STAGEN_BEST_ENABLED" == 1 && "$B300_STAGEN_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEN_SPILL_FREE" == 1 && "$(passes "$B300_STAGEN_SPEEDUP")" == 1 ]]; then
  VALIDATED=1; control_threads="$B300_STAGEN_CONTROL_THREADS"; test_threads="$B300_STAGEN_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  validate_rows=(); for rows in $VALIDATE_ROWS "$U_ROWS" "$L_ROWS" "$F_ROWS" "$E_ROWS"; do [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue; [[ "$rows" == "$SEARCH_ROWS" ]] && continue; seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || validate_rows+=("$rows"); done
  stage=0
  for rows in "${validate_rows[@]}"; do ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$SELECTED_PAIR" "$SELECTED_BLOCK" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"; source "$CURRENT_ENV"; check_config; check_known_residue "$rows" "$B300_STAGEN_RESIDUE"; if [[ "$B300_STAGEN_PAIR_POLICY" != "$SELECTED_PAIR" || "$B300_STAGEN_BLOCK_POLICY" != "$SELECTED_BLOCK" || "$B300_STAGEN_BEST_ENABLED" != 1 || "$B300_STAGEN_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEN_SPILL_FREE" != 1 || "$(passes "$B300_STAGEN_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi; control_threads="$B300_STAGEN_CONTROL_THREADS"; test_threads="$B300_STAGEN_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"; done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"; check_config
if [[ "$VALIDATED" == 1 ]]; then FINAL_PAIR="$B300_STAGEN_PAIR_POLICY"; FINAL_BLOCK="$B300_STAGEN_BLOCK_POLICY"; FINAL_BIN="$B300_STAGEN_BIN"; FINAL_THREADS="$B300_STAGEN_THREADS"; FINAL_WALL="$B300_STAGEN_WALL_S"; FINAL_HIGH="$B300_STAGEN_HIGH_S"; FINAL_SPEED="$B300_STAGEN_SPEEDUP"; else FINAL_PAIR="$B300_STAGEN_BASE_COUNT_POLICY"; FINAL_BLOCK="$B300_STAGEN_BASE_COUNT_POLICY"; FINAL_BIN="$B300_STAGEN_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEN_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEN_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEN_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
  printf 'B300_STAGEN_STAGED_VALIDATED=%q\n' "$VALIDATED"; printf 'B300_STAGEN_FINAL_ENABLED=%q\n' "$VALIDATED"; printf 'B300_STAGEN_NGPU=%q\n' "$NGPU"; printf 'B300_STAGEN_UPSTREAM_KIND=%q\n' "$RESOLVED_UPSTREAM"; printf 'B300_STAGEN_MATE_LOAD_POLICY=%q\n' "$B300_STAGEN_MATE_LOAD_POLICY"; printf 'B300_STAGEN_BASE_COUNT_POLICY=%q\n' "$B300_STAGEN_BASE_COUNT_POLICY";
  printf 'B300_STAGEN_PAIR_POLICY=%q\n' "$FINAL_PAIR"; printf 'B300_STAGEN_BLOCK_POLICY=%q\n' "$FINAL_BLOCK"; printf 'B300_STAGEN_FINAL_BIN=%q\n' "$FINAL_BIN"; printf 'B300_STAGEN_FINAL_THREADS=%q\n' "$FINAL_THREADS"; printf 'B300_STAGEN_FINAL_WALL_S=%q\n' "$FINAL_WALL"; printf 'B300_STAGEN_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"; printf 'B300_STAGEN_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"; printf 'B300_STAGEN_FINAL_SPILL_FREE=1\n';
  printf 'B300_STAGEN_CONTROL_BIN=%q\n' "$B300_STAGEN_CONTROL_BIN"; printf 'B300_STAGEN_CONTROL_THREADS=%q\n' "$B300_STAGEN_CONTROL_THREADS"; printf 'B300_STAGEN_SELF_WIDTH=%q\n' "$B300_STAGEN_SELF_WIDTH"; printf 'B300_STAGEN_SELF_DISTANCE=%q\n' "$B300_STAGEN_SELF_DISTANCE"; printf 'B300_STAGEN_SELF_EVICT=%q\n' "$B300_STAGEN_SELF_EVICT"; printf 'B300_STAGEN_SELF_GUARD=%q\n' "$B300_STAGEN_SELF_GUARD"; printf 'B300_STAGEN_MATE_WIDTH=%q\n' "$B300_STAGEN_MATE_WIDTH"; printf 'B300_STAGEN_MATE_DISTANCE=%q\n' "$B300_STAGEN_MATE_DISTANCE"; printf 'B300_STAGEN_MATE_EVICT=%q\n' "$B300_STAGEN_MATE_EVICT"; printf 'B300_STAGEN_MATE_GUARD=%q\n' "$B300_STAGEN_MATE_GUARD";
  printf 'B300_STAGEN_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEN_ROWS"; printf 'B300_STAGEN_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEN_RESIDUE"; printf 'B300_STAGEN_SEARCH_PAIR_POLICIES=%q\n' "$PAIR_POLICY_LIST"; printf 'B300_STAGEN_SEARCH_BLOCK_POLICIES=%q\n' "$BLOCK_POLICY_LIST"; printf 'B300_STAGEN_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"; printf 'B300_STAGEN_INPUT_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"; printf 'B300_STAGEN_INPUT_STAGEL_ENV=%q\n' "$STAGEL_GUARD_ENV"; printf 'B300_STAGEN_INPUT_STAGEM_ENV=%q\n' "$STAGEM_WINNER_ENV";
} >"$FINAL_ENV"
cat "$FINAL_ENV"; echo "b300-nextgen-hybrid8-pair-block-load-policy-staged-calibrate OK stage=N validated=$VALIDATED pair=$FINAL_PAIR block=$FINAL_BLOCK upstream=$RESOLVED_UPSTREAM speedup=${FINAL_SPEED}x final_rows=$B300_STAGEN_ROWS ngpu=$NGPU" >&2
