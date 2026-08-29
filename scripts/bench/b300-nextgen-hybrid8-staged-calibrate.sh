#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"
REGCAP_LIST="${REGCAP_LIST:-0 96 128 160 192 224}"
L2_SIZES="${L2_SIZES:-0 64 128 256}"
HYBRID_ILP8_THRESHOLDS="${HYBRID_ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
CORE_REPEATS="${CORE_REPEATS:-1}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
MAIN_MIN_SPEEDUP="${MAIN_MIN_SPEEDUP:-1.01}"
BLOCK_MIN_SPEEDUP="${BLOCK_MIN_SPEEDUP:-1.01}"
LATENCY_MIN_SPEEDUP="${LATENCY_MIN_SPEEDUP:-1.01}"
CGL2_MIN_SPEEDUP="${CGL2_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_staged}"
CORE_LOG="${CORE_LOG:-${PREFIX}.abcd.log}"
FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
CANONICAL_SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"

[[ -f "$CANONICAL_SWEEP" ]] || { echo "missing canonical hybrid sweep=$CANONICAL_SWEEP" >&2; exit 2; }
for rows in "$SEARCH_ROWS" $VALIDATE_ROWS; do
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || { echo "bad stage rows=$rows" >&2; exit 2; }
done
for n in CORE_REPEATS SEARCH_REPEATS VALIDATE_REPEATS; do
  v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$n must be >=1" >&2; exit 2; }
done
python3 - "$HYBRID_MIN_SPEEDUP" "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1])<1.0: raise SystemExit('HYBRID_MIN_SPEEDUP must be >=1')
if float(sys.argv[2])<=0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY

getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }

echo "=== hybrid8 staged calibration A-D search rows=$SEARCH_ROWS ===" >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$SEARCH_ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" REPEATS="$CORE_REPEATS" REGCAP_LIST="$REGCAP_LIST" L2_SIZES="$L2_SIZES" \
  MAIN_MIN_SPEEDUP="$MAIN_MIN_SPEEDUP" BLOCK_MIN_SPEEDUP="$BLOCK_MIN_SPEEDUP" LATENCY_MIN_SPEEDUP="$LATENCY_MIN_SPEEDUP" CGL2_MIN_SPEEDUP="$CGL2_MIN_SPEEDUP" \
  PREFIX="${PREFIX}.abcd" bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-calibrate-cgl2.sh" | tee "$CORE_LOG"
[[ "$(getv b300_nextgen_cgl2_calibrate_exact_gates "$CORE_LOG")" == 1 ]] || { echo 'A-D exact gate missing' >&2; exit 3; }
CORE_RES="$(getv b300_nextgen_cgl2_calibrate_residue "$CORE_LOG")"
H="$(getv b300_nextgen_cgl2_calibrate_final_high_drop "$CORE_LOG")"
BASE_ILP="$(getv b300_nextgen_cgl2_calibrate_final_ilp "$CORE_LOG")"
CG="$(getv b300_nextgen_cgl2_calibrate_final_random_cg "$CORE_LOG")"
CGL2="$(getv b300_nextgen_cgl2_calibrate_final_random_cg_l2_fetch_bytes "$CORE_LOG")"
PRE="$(getv b300_nextgen_cgl2_calibrate_final_prefetch_l2 "$CORE_LOG")"
DUAL="$(getv b300_nextgen_cgl2_calibrate_final_dualmask "$CORE_LOG")"
BATCH="$(getv b300_nextgen_cgl2_calibrate_final_closure_batch "$CORE_LOG")"
CAP="$(getv b300_nextgen_cgl2_calibrate_final_maxrregcount "$CORE_LOG")"
CORE_THREADS="$(getv b300_nextgen_cgl2_calibrate_final_threads "$CORE_LOG")"
CORE_WALL="$(getv b300_nextgen_cgl2_calibrate_final_wall_s "$CORE_LOG")"
case "$BASE_ILP" in 2|4|8) ;; *) echo "bad A-D ILP=$BASE_ILP" >&2; exit 3;; esac
[[ "$CG" == 0 || "$CG" == 1 ]] || exit 3
case "$CGL2" in 0|64|128|256) ;; *) exit 3;; esac

normalize_stage(){
  local raw_env="$1" result="$2" binaries="$3" normalized="$4"
  # shellcheck disable=SC1090
  source "$raw_env"
  local enabled=0
  [[ "$B300_HYBRID8_WINNER_MODE" == hybrid ]] && enabled=1
  [[ "$B300_HYBRID8_WINNER_SPILL_STORE_BYTES" == 0 && "$B300_HYBRID8_WINNER_SPILL_LOAD_BYTES" == 0 ]] || {
    echo 'canonical hybrid sweep returned a spilling winner' >&2; exit 4
  }
  local base_bin base_threads base_wall
  base_bin="$(awk -F $'\t' '$1=="baseline"{print $5;exit}' "$binaries")"
  [[ -n "$base_bin" && -x "$base_bin" ]] || { echo "baseline binary missing from $binaries" >&2; exit 4; }
  read -r base_threads base_wall < <(python3 - "$result" <<'PY'
import csv,statistics,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['mode']=='baseline']
if not rows: raise SystemExit('baseline rows missing')
by={}
for r in rows: by.setdefault(int(r['threads']),[]).append(float(r['wall_s']))
best=min((statistics.median(v),t) for t,v in by.items())
print(best[1],f'{best[0]:.9f}')
PY
)
  {
    printf 'B300_HYBRID8_WINNER_ENABLED=%q\n' "$enabled"
    printf 'B300_HYBRID8_WINNER_MODE=%q\n' "$B300_HYBRID8_WINNER_MODE"
    printf 'B300_HYBRID8_WINNER_THRESHOLD=%q\n' "$B300_HYBRID8_WINNER_THRESHOLD"
    printf 'B300_HYBRID8_WINNER_BIN=%q\n' "$B300_HYBRID8_WINNER_BIN"
    printf 'B300_HYBRID8_WINNER_THREADS=%q\n' "$B300_HYBRID8_WINNER_THREADS"
    printf 'B300_HYBRID8_WINNER_WALL_S=%q\n' "$B300_HYBRID8_WINNER_WALL_S"
    printf 'B300_HYBRID8_WINNER_SPILL_FREE=1\n'
    printf 'B300_HYBRID8_SPEEDUP_VS_BASE=%q\n' "$B300_HYBRID8_WINNER_SPEEDUP_VS_BASELINE"
    printf 'B300_HYBRID8_BASE_BIN=%q\n' "$base_bin"
    printf 'B300_HYBRID8_BASE_THREADS=%q\n' "$base_threads"
    printf 'B300_HYBRID8_BASE_WALL_S=%q\n' "$base_wall"
    printf 'B300_HYBRID8_RESIDUE=%q\n' "$B300_HYBRID8_RESIDUE"
  } >"$normalized"
}

run_hybrid_stage(){
  local rows="$1" thresholds="$2" threads="$3" repeats="$4" tag="$5"
  local stage_prefix="${PREFIX}.${tag}.r${rows}"
  local stage_log="${stage_prefix}.log" raw_env="${stage_prefix}_raw-winner.env" stage_env="${stage_prefix}_winner.env"
  local result="${stage_prefix}.tsv" binaries="${stage_prefix}_logs/binaries.tsv"
  echo "=== hybrid8 Stage E rows=$rows thresholds=[$thresholds] threads=[$threads] repeats=$repeats ===" >&2
  ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    HIGH_DROP_CHUNK="$H" BASE_RECURRENCE_ILP="$BASE_ILP" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" \
    PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" \
    ILP8_THRESHOLDS="$thresholds" THREADS_LIST="$threads" REPEATS="$repeats" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" \
    PREFIX="$stage_prefix" RESULT="$result" WINNER_ENV="$raw_env" \
    bash "$CANONICAL_SWEEP" | tee "$stage_log" >&2
  [[ "$(getv b300_nextgen_hybrid8_exact_intermediate_match "$stage_log")" == 1 ]] || { echo "Stage E exact gate missing rows=$rows" >&2; exit 4; }
  [[ "$(getv b300_nextgen_hybrid8_residue "$stage_log")" == "$CORE_RES" ]] || { echo "FATAL A-D/E residue mismatch rows=$rows" >&2; exit 4; }
  [[ -s "$raw_env" && -s "$result" && -s "$binaries" ]] || { echo "Stage E artifacts missing rows=$rows" >&2; exit 4; }
  normalize_stage "$raw_env" "$result" "$binaries" "$stage_env"
  printf '%s\n' "$stage_env"
}

passes_margin(){
  python3 - "$1" "$HYBRID_MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1])>=float(sys.argv[2]) else 0)
PY
}

SEARCH_ENV="$(run_hybrid_stage "$SEARCH_ROWS" "$HYBRID_ILP8_THRESHOLDS" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
VALIDATED=0
CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_HYBRID8_WINNER_ENABLED" == 1 && "$(passes_margin "$B300_HYBRID8_SPEEDUP_VS_BASE")" == 1 ]]; then
  VALIDATED=1
  threshold="$B300_HYBRID8_WINNER_THRESHOLD"
  base_threads="$B300_HYBRID8_BASE_THREADS"
  winner_threads="$B300_HYBRID8_WINNER_THREADS"
  validation_threads="$base_threads"
  [[ "$winner_threads" == "$base_threads" ]] || validation_threads+=" $winner_threads"
  stage=0
  for rows in $VALIDATE_ROWS; do
    ((stage+=1))
    CURRENT_ENV="$(run_hybrid_stage "$rows" "$threshold" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    if [[ "$B300_HYBRID8_WINNER_ENABLED" != 1 || "$B300_HYBRID8_WINNER_THRESHOLD" != "$threshold" || "$B300_HYBRID8_WINNER_SPILL_FREE" != 1 || "$(passes_margin "$B300_HYBRID8_SPEEDUP_VS_BASE")" != 1 ]]; then
      VALIDATED=0
      break
    fi
    base_threads="$B300_HYBRID8_BASE_THREADS"
    winner_threads="$B300_HYBRID8_WINNER_THREADS"
    validation_threads="$base_threads"
    [[ "$winner_threads" == "$base_threads" ]] || validation_threads+=" $winner_threads"
  done
fi

# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_ENABLED=1
  FINAL_THRESHOLD="$B300_HYBRID8_WINNER_THRESHOLD"
  FINAL_BIN="$B300_HYBRID8_WINNER_BIN"
  FINAL_THREADS="$B300_HYBRID8_WINNER_THREADS"
  FINAL_WALL="$B300_HYBRID8_WINNER_WALL_S"
  FINAL_SPEED="$B300_HYBRID8_SPEEDUP_VS_BASE"
  FINAL_SPILL_FREE="$B300_HYBRID8_WINNER_SPILL_FREE"
else
  FINAL_ENABLED=0
  FINAL_THRESHOLD=0
  FINAL_BIN="$B300_HYBRID8_BASE_BIN"
  FINAL_THREADS="$B300_HYBRID8_BASE_THREADS"
  FINAL_WALL="$B300_HYBRID8_BASE_WALL_S"
  FINAL_SPEED=1.000000000
  FINAL_SPILL_FREE=1
fi

{
  printf 'B300_HYBRID8_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_HYBRID8_FINAL_ENABLED=%q\n' "$FINAL_ENABLED"
  printf 'B300_HYBRID8_FINAL_THRESHOLD=%q\n' "$FINAL_THRESHOLD"
  printf 'B300_HYBRID8_FINAL_BIN=%q\n' "$FINAL_BIN"
  printf 'B300_HYBRID8_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_HYBRID8_FINAL_WALL_S=%q\n' "$FINAL_WALL"
  printf 'B300_HYBRID8_FINAL_SPEEDUP_VS_BASE=%q\n' "$FINAL_SPEED"
  printf 'B300_HYBRID8_FINAL_SPILL_FREE=%q\n' "$FINAL_SPILL_FREE"
  printf 'B300_HYBRID8_BASE_BIN=%q\n' "$B300_HYBRID8_BASE_BIN"
  printf 'B300_HYBRID8_BASE_THREADS=%q\n' "$B300_HYBRID8_BASE_THREADS"
  printf 'B300_HYBRID8_BASE_WALL_S=%q\n' "$B300_HYBRID8_BASE_WALL_S"
  printf 'B300_HYBRID8_RESIDUE=%q\n' "$CORE_RES"
  printf 'B300_HYBRID8_HIGH_DROP_CHUNK=%q\n' "$H"
  printf 'B300_HYBRID8_BASE_RECURRENCE_ILP=%q\n' "$BASE_ILP"
  printf 'B300_HYBRID8_RANDOM_CG=%q\n' "$CG"
  printf 'B300_HYBRID8_RANDOM_CG_L2_FETCH_BYTES=%q\n' "$CGL2"
  printf 'B300_HYBRID8_PREFETCH_L2=%q\n' "$PRE"
  printf 'B300_HYBRID8_DUALMASK=%q\n' "$DUAL"
  printf 'B300_HYBRID8_CLOSURE_BATCH=%q\n' "$BATCH"
  printf 'B300_HYBRID8_MAXRREGCOUNT=%q\n' "$CAP"
  printf 'B300_HYBRID8_CORE_THREADS=%q\n' "$CORE_THREADS"
  printf 'B300_HYBRID8_CORE_WALL_S=%q\n' "$CORE_WALL"
  printf 'B300_HYBRID8_MIN_SPEEDUP=%q\n' "$HYBRID_MIN_SPEEDUP"
  printf 'B300_HYBRID8_CORE_LOG=%q\n' "$CORE_LOG"
} >"$FINAL_ENV"

cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-staged-calibrate OK validated=$VALIDATED final_enabled=$FINAL_ENABLED final_threshold=$FINAL_THRESHOLD winner_env=$FINAL_ENV canonical_sweep=1" >&2
