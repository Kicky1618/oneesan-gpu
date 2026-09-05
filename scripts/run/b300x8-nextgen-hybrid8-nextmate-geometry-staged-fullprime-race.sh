#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'independent mate-geometry full-prime race targets n=27' >&2; exit 2; }
INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
RUN_STAGED="${RUN_STAGED:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"
MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"
SELF_EVICT="${SELF_EVICT:-default}"
MATE_EVICT="${MATE_EVICT:-default}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_staged}"
WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_fullprime_n27}"
PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"

for x in RUN_STAGED SELECT_ONLY REBUILD_BUCKETS PREPARE_ONLY; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
for x in TARGET_MIB MAX_WINDOW; do v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
for ev in SELF_EVICT MATE_EVICT; do case "${!ev}" in default|normal|last) ;; *) exit 2;; esac; done
[[ -s "$INPUT_ENV" ]] || { echo "missing Stage-F INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
[[ -s "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

if [[ "$RUN_STAGED" == 1 ]]; then
  echo '=== Stage I independent mate-geometry staged calibration ===' >&2
  INPUT_ENV="$INPUT_ENV" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" \
    MIN_SPEEDUP="$MIN_SPEEDUP" PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-geometry-staged-calibrate.sh"
fi

[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-I WINNER_ENV=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for key in \
  B300_STAGEI_STAGED_VALIDATED B300_STAGEI_FINAL_ENABLED \
  B300_STAGEI_SELF_WIDTH B300_STAGEI_SELF_DISTANCE B300_STAGEI_SELF_EVICT \
  B300_STAGEI_FINAL_MATE_WIDTH B300_STAGEI_FINAL_MATE_DISTANCE B300_STAGEI_FINAL_MATE_EVICT \
  B300_STAGEI_FINAL_BIN B300_STAGEI_FINAL_THREADS B300_STAGEI_FINAL_SPEEDUP_VS_SELF B300_STAGEI_FINAL_SPILL_FREE \
  B300_STAGEI_CONTROL_BIN B300_STAGEI_CONTROL_THREADS \
  B300_STAGEI_FINAL_STAGE_ROWS B300_STAGEI_FINAL_STAGE_RESIDUE B300_STAGEI_INPUT_STAGE_F_ENV; do
  [[ -n "${!key+x}" ]] || { echo "Stage-I winner env missing $key" >&2; exit 3; }
done
[[ "$B300_STAGEI_STAGED_VALIDATED" == 1 && "$B300_STAGEI_FINAL_ENABLED" == 1 && "$B300_STAGEI_FINAL_SPILL_FREE" == 1 ]] || {
  echo 'Stage-I independent mate geometry did not survive staged validation' >&2; exit 4;
}
for w in B300_STAGEI_SELF_WIDTH B300_STAGEI_FINAL_MATE_WIDTH; do case "${!w}" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in B300_STAGEI_SELF_DISTANCE B300_STAGEI_FINAL_MATE_DISTANCE; do case "${!d}" in 1|2|4) ;; *) exit 3;; esac; done
for ev in B300_STAGEI_SELF_EVICT B300_STAGEI_FINAL_MATE_EVICT; do case "${!ev}" in default|normal|last) ;; *) exit 3;; esac; done
for th in B300_STAGEI_FINAL_THREADS B300_STAGEI_CONTROL_THREADS; do v="${!th}"; [[ "$v" =~ ^[0-9]+$ ]] && ((v>=32&&v<=1024&&v%32==0)) || exit 3; done
[[ -x "$B300_STAGEI_FINAL_BIN" && -x "$B300_STAGEI_CONTROL_BIN" ]] || { echo 'Stage-I final/control binary missing' >&2; exit 3; }
[[ "$B300_STAGEI_INPUT_STAGE_F_ENV" == "$INPUT_ENV" ]] || { echo 'Stage-I input Stage-F env drifted' >&2; exit 3; }
python3 - "$B300_STAGEI_FINAL_SPEEDUP_VS_SELF" "$MIN_SPEEDUP" <<'PY'
import sys
speed,need=map(float,sys.argv[1:])
if speed < need: raise SystemExit(f'Stage-I speedup {speed:.9f}x below {need:.9f}x')
PY

if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"
  sha256sum "$WINNER_ENV" "$INPUT_ENV" "$B300_STAGEI_FINAL_BIN" "$B300_STAGEI_CONTROL_BIN" >"$tmp"
  mv "$tmp" "$MANIFEST"
else
  [[ -s "$MANIFEST" ]] || { echo "missing Stage-I promotion manifest=$MANIFEST; rerun with RUN_STAGED=1" >&2; exit 3; }
fi
if ! sha256sum -c "$MANIFEST" >/dev/null; then
  echo 'Stage-I promotion fingerprint mismatch; rerun staged calibration' >&2; exit 3
fi

FINAL_SHA="$(sha256sum "$B300_STAGEI_FINAL_BIN" | awk '{print $1}')"
CONTROL_SHA="$(sha256sum "$B300_STAGEI_CONTROL_BIN" | awk '{print $1}')"
INPUT_SHA="$(sha256sum "$INPUT_ENV" | awk '{print $1}')"
WINNER_SHA="$(sha256sum "$WINNER_ENV" | awk '{print $1}')"
MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
label="nextmate_selfw${B300_STAGEI_SELF_WIDTH}_selfd${B300_STAGEI_SELF_DISTANCE}_matew${B300_STAGEI_FINAL_MATE_WIDTH}_mated${B300_STAGEI_FINAL_MATE_DISTANCE}"
control_label="nextself_selfw${B300_STAGEI_SELF_WIDTH}_selfd${B300_STAGEI_SELF_DISTANCE}"
PROMOTION_ENV="${RACE_PREFIX}_promotion.env"
cat >"$PROMOTION_ENV" <<EOF
B300_STAGEI_PROMOTION_VALIDATED=1
B300_STAGEI_PROMOTION_BIN=$(printf '%q' "$B300_STAGEI_FINAL_BIN")
B300_STAGEI_PROMOTION_BIN_SHA256=$FINAL_SHA
B300_STAGEI_PROMOTION_THREADS=$B300_STAGEI_FINAL_THREADS
B300_STAGEI_PROMOTION_CONTROL_BIN=$(printf '%q' "$B300_STAGEI_CONTROL_BIN")
B300_STAGEI_PROMOTION_CONTROL_SHA256=$CONTROL_SHA
B300_STAGEI_PROMOTION_CONTROL_THREADS=$B300_STAGEI_CONTROL_THREADS
B300_STAGEI_PROMOTION_SELF_WIDTH=$B300_STAGEI_SELF_WIDTH
B300_STAGEI_PROMOTION_SELF_DISTANCE=$B300_STAGEI_SELF_DISTANCE
B300_STAGEI_PROMOTION_SELF_EVICT=$B300_STAGEI_SELF_EVICT
B300_STAGEI_PROMOTION_MATE_WIDTH=$B300_STAGEI_FINAL_MATE_WIDTH
B300_STAGEI_PROMOTION_MATE_DISTANCE=$B300_STAGEI_FINAL_MATE_DISTANCE
B300_STAGEI_PROMOTION_MATE_EVICT=$B300_STAGEI_FINAL_MATE_EVICT
B300_STAGEI_PROMOTION_SPEEDUP=$B300_STAGEI_FINAL_SPEEDUP_VS_SELF
B300_STAGEI_PROMOTION_FINAL_STAGE_ROWS=$B300_STAGEI_FINAL_STAGE_ROWS
B300_STAGEI_PROMOTION_FINAL_STAGE_RESIDUE=$(printf '%q' "$B300_STAGEI_FINAL_STAGE_RESIDUE")
B300_STAGEI_PROMOTION_INPUT_STAGE_F_ENV=$(printf '%q' "$INPUT_ENV")
B300_STAGEI_PROMOTION_INPUT_STAGE_F_ENV_SHA256=$INPUT_SHA
B300_STAGEI_PROMOTION_WINNER_ENV_SHA256=$WINNER_SHA
B300_STAGEI_PROMOTION_MANIFEST=$(printf '%q' "$MANIFEST")
B300_STAGEI_PROMOTION_MANIFEST_SHA256=$MANIFEST_SHA
EOF

if [[ "$PREPARE_ONLY" == 1 ]]; then
  mkdir -p "$(dirname "$PREPARE_ENV")"
  {
    printf 'B300_STAGEI_PREPARED=1\n'
    printf 'B300_STAGEI_PREPARED_BIN=%q\n' "$B300_STAGEI_FINAL_BIN"
    printf 'B300_STAGEI_PREPARED_LABEL=%q\n' "$label"
    printf 'B300_STAGEI_PREPARED_THREADS=%q\n' "$B300_STAGEI_FINAL_THREADS"
    printf 'B300_STAGEI_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGEI_CONTROL_BIN"
    printf 'B300_STAGEI_PREPARED_CONTROL_LABEL=%q\n' "$control_label"
    printf 'B300_STAGEI_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGEI_CONTROL_THREADS"
    printf 'B300_STAGEI_PREPARED_SELF_WIDTH=%q\n' "$B300_STAGEI_SELF_WIDTH"
    printf 'B300_STAGEI_PREPARED_SELF_DISTANCE=%q\n' "$B300_STAGEI_SELF_DISTANCE"
    printf 'B300_STAGEI_PREPARED_MATE_WIDTH=%q\n' "$B300_STAGEI_FINAL_MATE_WIDTH"
    printf 'B300_STAGEI_PREPARED_MATE_DISTANCE=%q\n' "$B300_STAGEI_FINAL_MATE_DISTANCE"
    printf 'B300_STAGEI_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGEI_FINAL_SPEEDUP_VS_SELF"
    printf 'B300_STAGEI_PREPARED_MANIFEST=%q\n' "$MANIFEST"
    printf 'B300_STAGEI_PREPARED_PROMOTION_ENV=%q\n' "$PROMOTION_ENV"
  } >"$PREPARE_ENV"
  cat "$PREPARE_ENV"
  echo "STAGE I PREPARED env=$PREPARE_ENV label=$label control=$control_label speedup=${B300_STAGEI_FINAL_SPEEDUP_VS_SELF}x" >&2
  exit 0
fi

echo "=== Stage I full-prime race: $label vs $control_label vs profiled warp/orbit ===" >&2
exec env \
  PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_STAGEI_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGEI_FINAL_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEI_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGEI_CONTROL_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
