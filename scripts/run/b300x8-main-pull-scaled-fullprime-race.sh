#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'scaled main-pull full-prime race targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ROWS="${ROWS:-1}"
REPEATS="${REPEATS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
ILP_LIST="${ILP_LIST:-1 2 3 4}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"
RUN_SWEEP="${RUN_SWEEP:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
SWEEP_PREFIX="${SWEEP_PREFIX:-$ONEESAN_ROOT/work/b300_main_pull_ilp1234_scaled_row${ROWS}_hd${HIGH_DROP_CHUNK}}"
WINNER_ENV="${WINNER_ENV:-${SWEEP_PREFIX}_winner.env}"
SWEEP_RESULT="${SWEEP_RESULT:-${SWEEP_PREFIX}.tsv}"
SWEEP_LOGDIR="${SWEEP_LOGDIR:-${SWEEP_PREFIX}_logs}"
MANIFEST="${MANIFEST:-${SWEEP_PREFIX}_artifacts.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_main_pull_scaled_fullprime_race_n27}"
for x in RUN_SWEEP SELECT_ONLY REBUILD_BUCKETS; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1 && ROWS<=28)) || exit 2
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }

write_manifest(){
  local tmp="${MANIFEST}.tmp"
  : >"$tmp"
  sha256sum "$WINNER_ENV" "$SWEEP_RESULT" "$SWEEP_LOGDIR/binaries.tsv" >>"$tmp"
  while IFS=$'\t' read -r ilp bin berr; do
    [[ "$ilp" == ilp ]] && continue
    [[ -x "$bin" ]] || { rm -f "$tmp"; echo "manifest binary missing: $bin" >&2; return 3; }
    sha256sum "$bin" >>"$tmp"
  done <"$SWEEP_LOGDIR/binaries.tsv"
  mv "$tmp" "$MANIFEST"
}

if [[ "$RUN_SWEEP" == 1 ]]; then
  echo '=== scaled main-pull ILP1/2/3/4 partial-row calibration ===' >&2
  N=27 ARCH="$ARCH" ROWS="$ROWS" REPEATS="$REPEATS" THREADS_LIST="$THREADS_LIST" ILP_LIST="$ILP_LIST" \
    HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    PREFIX="$SWEEP_PREFIX" LOGDIR="$SWEEP_LOGDIR" RESULT="$SWEEP_RESULT" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-ilp1234-scaled-ab.sh"
  write_manifest
fi

[[ -f "$WINNER_ENV" && -f "$SWEEP_RESULT" && -f "$SWEEP_LOGDIR/binaries.tsv" && -f "$MANIFEST" ]] || {
  echo 'missing scaled main-pull sweep artifacts/manifest; use RUN_SWEEP=1' >&2; exit 3;
}
if ! sha256sum -c "$MANIFEST" >/dev/null; then
  echo 'scaled main-pull calibration fingerprint mismatch; rerun with RUN_SWEEP=1' >&2
  exit 3
fi

# shellcheck disable=SC1090
source "$WINNER_ENV"
for k in B300_MAIN_PULL_SCALED_WINNER_ILP B300_MAIN_PULL_SCALED_WINNER_THREADS B300_MAIN_PULL_SCALED_WINNER_BIN B300_MAIN_PULL_SCALED_WINNER_SPILL_STORE_BYTES B300_MAIN_PULL_SCALED_WINNER_SPILL_LOAD_BYTES B300_MAIN_PULL_SCALED_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "winner env missing $k" >&2; exit 3; }
done
[[ -x "$B300_MAIN_PULL_SCALED_WINNER_BIN" ]] || { echo 'scaled winner binary missing' >&2; exit 3; }
[[ "$B300_MAIN_PULL_SCALED_WINNER_SPILL_STORE_BYTES" == 0 && "$B300_MAIN_PULL_SCALED_WINNER_SPILL_LOAD_BYTES" == 0 ]] || {
  echo "refusing full-prime promotion of spilling winner store=$B300_MAIN_PULL_SCALED_WINNER_SPILL_STORE_BYTES load=$B300_MAIN_PULL_SCALED_WINNER_SPILL_LOAD_BYTES" >&2
  exit 4
}

BASE_BIN="$(awk -F '\t' '$1==1{print $2}' "$SWEEP_LOGDIR/binaries.tsv" | tail -n1)"
[[ -x "$BASE_BIN" ]] || { echo 'ILP1 baseline binary missing' >&2; exit 3; }
BASE_THREADS="$(python3 - "$SWEEP_RESULT" <<'PY'
import csv,statistics,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['ilp']=='1']
if not rows: raise SystemExit('missing ILP1 baseline rows')
by={}
for r in rows: by.setdefault(int(r['threads']),[]).append(float(r['wall_s']))
print(min(by,key=lambda t:statistics.median(by[t])))
PY
)"
[[ "$BASE_THREADS" =~ ^[0-9]+$ ]] || exit 3
RECORDED_WINNER="$(awk -F '\t' -v x="$B300_MAIN_PULL_SCALED_WINNER_ILP" '$1==x{print $2}' "$SWEEP_LOGDIR/binaries.tsv" | tail -n1)"
[[ -n "$RECORDED_WINNER" && "$(readlink -f "$RECORDED_WINNER")" == "$(readlink -f "$B300_MAIN_PULL_SCALED_WINNER_BIN")" ]] || {
  echo 'winner env / binaries.tsv mismatch; rerun calibration' >&2; exit 3;
}

label="mainpull_scaled_ilp${B300_MAIN_PULL_SCALED_WINNER_ILP}"
echo "=== full-prime race winner=$label threads=$B300_MAIN_PULL_SCALED_WINNER_THREADS vs ilp1 baseline threads=$BASE_THREADS + profiled warp/orbit ===" >&2
echo "calibration_manifest=$MANIFEST" >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
FORCED_OVERRIDE_BIN="$B300_MAIN_PULL_SCALED_WINNER_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_MAIN_PULL_SCALED_WINNER_THREADS" \
FORCED_BASE_BIN="$BASE_BIN" FORCED_BASE_LABEL=mainpull_ilp1_base FORCED_BASE_THREADS="$BASE_THREADS" \
REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
