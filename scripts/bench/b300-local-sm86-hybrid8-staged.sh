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
VALIDATE_THRESHOLD="${VALIDATE_THRESHOLD:-0}"
THREADS_LIST="${THREADS_LIST:-128 256}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_local_sm86_hybrid8_staged}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
mkdir -p "$(dirname "$PREFIX")"

[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$VALIDATE_THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'VALIDATE_THRESHOLD must be non-negative' >&2; exit 2; }
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
SEARCH_SPEED="$B300_HYBRID8_WINNER_SPEEDUP_VS_BASELINE"

last_env="$SEARCH_ENV"
stage=0
for rows in $VALIDATE_ROWS; do
  ((stage+=1))
  last_env="$(run_stage "$rows" "$VALIDATE_THRESHOLD" "validate${stage}" | tail -n1)"
done

# shellcheck disable=SC1090
source "$last_env"
echo "b300_local_sm86_hybrid8_staged=OK search_rows=$SEARCH_ROWS validate_rows=[$VALIDATE_ROWS] search_profile=$SEARCH_PROFILE search_speedup=${SEARCH_SPEED}x search_residue=$SEARCH_RES validate_threshold=$VALIDATE_THRESHOLD final_residue=$B300_HYBRID8_RESIDUE ngpu=$NGPU arch=$ARCH"
