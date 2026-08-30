#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_86}"
NGPU="${NGPU:-1}"
TARGET_MIB="${TARGET_MIB:-1024}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MOD="${MOD:-4294967291}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_THRESHOLDS="${SEARCH_THRESHOLDS:-0 262144 1048576 4194304}"
# auto validates the threshold selected by the search stage. An explicit
# integer remains available for targeted experiments.
VALIDATE_THRESHOLD="${VALIDATE_THRESHOLD:-auto}"
# A separate deep threshold=0 run exercises the ILP8 kernel regardless of the
# selected policy. Set to 0 to disable or to another row count to relocate it.
FORCE_ILP8_ROWS="${FORCE_ILP8_ROWS:-8}"
THREADS_LIST="${THREADS_LIST:-128 256}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_local_sm86_hybrid8_staged}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
mkdir -p "$(dirname "$PREFIX")"

[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$VALIDATE_THRESHOLD" == auto || "$VALIDATE_THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'VALIDATE_THRESHOLD must be auto or non-negative integer' >&2; exit 2; }
[[ "$FORCE_ILP8_ROWS" =~ ^[0-9]+$ ]] && ((FORCE_ILP8_ROWS<=28)) || { echo 'FORCE_ILP8_ROWS must be 0..28' >&2; exit 2; }
for rows in "$SEARCH_ROWS" $VALIDATE_ROWS; do
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || { echo "bad rows=$rows" >&2; exit 2; }
done

run_stage(){
  local rows="$1" thresholds="$2" tag="$3"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"
  echo "=== local staged hybrid8 tag=$tag rows=$rows thresholds=[$thresholds] ngpu=$NGPU ===" >&2
  ARCH="$ARCH" NGPU="$NGPU" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
    ILP8_THRESHOLDS="$thresholds" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" \
    PREFIX="$p" WINNER_ENV="$envf" bash "$SWEEP" | tee "$log"
  grep -Fq 'b300_nextgen_hybrid8_exact_intermediate_match=1' "$log" || { echo "exact residue gate missing rows=$rows" >&2; exit 3; }
  grep -Fq "b300_nextgen_hybrid8_ngpu=$NGPU" "$log" || { echo "ngpu provenance missing rows=$rows" >&2; exit 3; }
  [[ -s "$envf" ]] || { echo "winner env missing rows=$rows" >&2; exit 3; }
  printf '%s\n' "$envf"
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$SEARCH_THRESHOLDS" search | tail -n1)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
SEARCH_RES="$B300_HYBRID8_RESIDUE"
SEARCH_PROFILE="$B300_HYBRID8_WINNER_PROFILE"
SEARCH_MODE="$B300_HYBRID8_WINNER_MODE"
SEARCH_THRESHOLD="$B300_HYBRID8_WINNER_THRESHOLD"
SEARCH_SPEED="$B300_HYBRID8_WINNER_SPEEDUP_VS_BASELINE"

if [[ "$VALIDATE_THRESHOLD" == auto ]]; then
  SELECTED_THRESHOLD="$SEARCH_THRESHOLD"
else
  SELECTED_THRESHOLD="$VALIDATE_THRESHOLD"
fi
[[ "$SELECTED_THRESHOLD" =~ ^[0-9]+$ ]] || { echo "invalid selected threshold=$SELECTED_THRESHOLD" >&2; exit 3; }

last_env="$SEARCH_ENV"
stage=0
for rows in $VALIDATE_ROWS; do
  ((stage+=1))
  last_env="$(run_stage "$rows" "$SELECTED_THRESHOLD" "validate${stage}" | tail -n1)"
done

FORCED_ENV=""
if (( FORCE_ILP8_ROWS > 0 )); then
  FORCED_ENV="$(run_stage "$FORCE_ILP8_ROWS" 0 forced_ilp8 | tail -n1)"
  # shellcheck disable=SC1090
  source "$FORCED_ENV"
  [[ "$B300_HYBRID8_RESIDUE" =~ ^[0-9]+$ ]] || { echo 'forced ILP8 residue missing' >&2; exit 3; }
fi

# Report the winner from the selected-policy validation, not the optional
# forced-ILP8 correctness run.
# shellcheck disable=SC1090
source "$last_env"
echo "b300_local_sm86_hybrid8_staged=OK search_rows=$SEARCH_ROWS validate_rows=[$VALIDATE_ROWS] search_profile=$SEARCH_PROFILE search_mode=$SEARCH_MODE search_threshold=$SEARCH_THRESHOLD search_speedup=${SEARCH_SPEED}x search_residue=$SEARCH_RES selected_validate_threshold=$SELECTED_THRESHOLD forced_ilp8_rows=$FORCE_ILP8_ROWS final_profile=$B300_HYBRID8_WINNER_PROFILE final_mode=$B300_HYBRID8_WINNER_MODE final_residue=$B300_HYBRID8_RESIDUE ngpu=$NGPU arch=$ARCH"
