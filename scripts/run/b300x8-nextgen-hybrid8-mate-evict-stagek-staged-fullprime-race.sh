#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-K mate-eviction race targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEJ_WINNER_ENV="${STAGEJ_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_stagej_staged_winner.env}"
STAGEJ_PREPARE_ENV="${STAGEJ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_stagej_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
RUN_STAGED="${RUN_STAGED:-1}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; PREPARE_ONLY="${PREPARE_ONLY:-0}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; EVICT_LIST="${EVICT_LIST:-default normal last}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_staged}"
WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_fullprime_n27}"
PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"
PROMOTION_ENV="${PROMOTION_ENV:-${RACE_PREFIX}_promotion.env}"
for x in RUN_STAGED SELECT_ONLY REBUILD_BUCKETS PREPARE_ONLY; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
for x in MOD TARGET_MIB MAX_WINDOW; do v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$x must be positive integer" >&2; exit 2; }; done
for f in "$PROFILE_FILE" "$STAGE_F_ENV" "$STAGEJ_WINNER_ENV" "$STAGEJ_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-K input=$f" >&2; exit 2; }; done
command -v sha256sum >/dev/null || exit 2
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

# Stage K is a refinement of one exact Stage-J artifact. Bind the refinement to
# the same smoke modulus and to the exact Stage-J promotion manifest before any
# GPU work, so a stale/same-path rebuilt Stage-J binary cannot be reused here.
# shellcheck disable=SC1090
source "$STAGEJ_PREPARE_ENV"
for k in B300_STAGEJ_PREPARED B300_STAGEJ_PREPARED_MOD B300_STAGEJ_PREPARED_MANIFEST; do
  [[ -n "${!k+x}" ]] || { echo "Stage-J prepare env missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEJ_PREPARED" == 1 ]] || { echo 'Stage-J prepare marker missing' >&2; exit 3; }
[[ "$B300_STAGEJ_PREPARED_MOD" == "$MOD" ]] || {
  echo "Stage-K/Stage-J modulus mismatch stagek=$MOD stagej=$B300_STAGEJ_PREPARED_MOD" >&2
  exit 3
}
[[ -s "$B300_STAGEJ_PREPARED_MANIFEST" ]] || { echo 'Stage-J promotion manifest missing' >&2; exit 3; }
sha256sum -c "$B300_STAGEJ_PREPARED_MANIFEST" >/dev/null || {
  echo 'Stage-J promotion manifest mismatch before Stage K' >&2
  exit 3
}
STAGEJ_MANIFEST_SHA="$(sha256sum "$B300_STAGEJ_PREPARED_MANIFEST" | awk '{print $1}')"

if [[ "$RUN_STAGED" == 1 ]]; then
  echo '=== Stage K mate-eviction staged calibration ===' >&2
  STAGE_F_ENV="$STAGE_F_ENV" STAGEJ_WINNER_ENV="$STAGEJ_WINNER_ENV" STAGEJ_PREPARE_ENV="$STAGEJ_PREPARE_ENV" \
    ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" EVICT_LIST="$EVICT_LIST" MIN_SPEEDUP="$MIN_SPEEDUP" \
    PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-evict-staged-calibrate.sh"
fi
[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-K winner env=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for k in B300_STAGEK_STAGED_VALIDATED B300_STAGEK_FINAL_ENABLED B300_STAGEK_SELF_WIDTH B300_STAGEK_SELF_DISTANCE B300_STAGEK_SELF_EVICT B300_STAGEK_MATE_WIDTH B300_STAGEK_MATE_DISTANCE B300_STAGEK_BASE_MATE_EVICT B300_STAGEK_FINAL_MATE_EVICT B300_STAGEK_FINAL_BIN B300_STAGEK_FINAL_THREADS B300_STAGEK_FINAL_SPEEDUP B300_STAGEK_FINAL_SPILL_FREE B300_STAGEK_CONTROL_BIN B300_STAGEK_CONTROL_THREADS B300_STAGEK_FINAL_STAGE_ROWS B300_STAGEK_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-K winner env missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEK_STAGED_VALIDATED" == 1 && "$B300_STAGEK_FINAL_ENABLED" == 1 && "$B300_STAGEK_FINAL_SPILL_FREE" == 1 ]] || { echo 'Stage K did not survive staged validation' >&2; exit 4; }
for w in "$B300_STAGEK_SELF_WIDTH" "$B300_STAGEK_MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$B300_STAGEK_SELF_DISTANCE" "$B300_STAGEK_MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for ev in "$B300_STAGEK_SELF_EVICT" "$B300_STAGEK_BASE_MATE_EVICT" "$B300_STAGEK_FINAL_MATE_EVICT"; do case "$ev" in default|normal|last) ;; *) exit 3;; esac; done
[[ "$B300_STAGEK_FINAL_MATE_EVICT" != "$B300_STAGEK_BASE_MATE_EVICT" ]] || { echo 'Stage-K promoted hint must differ from baseline' >&2; exit 4; }
[[ -x "$B300_STAGEK_FINAL_BIN" && -x "$B300_STAGEK_CONTROL_BIN" ]] || { echo 'Stage-K final/control binary missing' >&2; exit 3; }
for t in "$B300_STAGEK_FINAL_THREADS" "$B300_STAGEK_CONTROL_THREADS"; do [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || exit 3; done
python3 - "$B300_STAGEK_FINAL_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('Stage-K staged speedup below promotion threshold')
PY

if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"
  sha256sum "$WINNER_ENV" "$STAGE_F_ENV" "$STAGEJ_WINNER_ENV" "$STAGEJ_PREPARE_ENV" "$B300_STAGEJ_PREPARED_MANIFEST" "$B300_STAGEK_FINAL_BIN" "$B300_STAGEK_CONTROL_BIN" >"$tmp"
  mv "$tmp" "$MANIFEST"
else
  [[ -s "$MANIFEST" ]] || { echo "missing Stage-K manifest=$MANIFEST; rerun staged calibration" >&2; exit 3; }
fi
sha256sum -c "$MANIFEST" >/dev/null || { echo 'Stage-K promotion fingerprint mismatch' >&2; exit 3; }
FINAL_SHA="$(sha256sum "$B300_STAGEK_FINAL_BIN" | awk '{print $1}')"; CONTROL_SHA="$(sha256sum "$B300_STAGEK_CONTROL_BIN" | awk '{print $1}')"
WINNER_SHA="$(sha256sum "$WINNER_ENV" | awk '{print $1}')"; STAGEF_SHA="$(sha256sum "$STAGE_F_ENV" | awk '{print $1}')"; STAGEJ_WINNER_SHA="$(sha256sum "$STAGEJ_WINNER_ENV" | awk '{print $1}')"; STAGEJ_PREPARE_SHA="$(sha256sum "$STAGEJ_PREPARE_ENV" | awk '{print $1}')"; MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
label="stagek_mateevict_${B300_STAGEK_FINAL_MATE_EVICT}_selfw${B300_STAGEK_SELF_WIDTH}_selfd${B300_STAGEK_SELF_DISTANCE}_matew${B300_STAGEK_MATE_WIDTH}_mated${B300_STAGEK_MATE_DISTANCE}"
control_label="stagek_mateevict_${B300_STAGEK_BASE_MATE_EVICT}_control"
cat >"$PROMOTION_ENV" <<EOF
B300_STAGEK_PROMOTION_VALIDATED=1
B300_STAGEK_PROMOTION_MOD=$MOD
B300_STAGEK_PROMOTION_BIN=$(printf '%q' "$B300_STAGEK_FINAL_BIN")
B300_STAGEK_PROMOTION_BIN_SHA256=$FINAL_SHA
B300_STAGEK_PROMOTION_THREADS=$B300_STAGEK_FINAL_THREADS
B300_STAGEK_PROMOTION_CONTROL_BIN=$(printf '%q' "$B300_STAGEK_CONTROL_BIN")
B300_STAGEK_PROMOTION_CONTROL_SHA256=$CONTROL_SHA
B300_STAGEK_PROMOTION_CONTROL_THREADS=$B300_STAGEK_CONTROL_THREADS
B300_STAGEK_PROMOTION_SELF_WIDTH=$B300_STAGEK_SELF_WIDTH
B300_STAGEK_PROMOTION_SELF_DISTANCE=$B300_STAGEK_SELF_DISTANCE
B300_STAGEK_PROMOTION_SELF_EVICT=$B300_STAGEK_SELF_EVICT
B300_STAGEK_PROMOTION_MATE_WIDTH=$B300_STAGEK_MATE_WIDTH
B300_STAGEK_PROMOTION_MATE_DISTANCE=$B300_STAGEK_MATE_DISTANCE
B300_STAGEK_PROMOTION_BASE_MATE_EVICT=$B300_STAGEK_BASE_MATE_EVICT
B300_STAGEK_PROMOTION_MATE_EVICT=$B300_STAGEK_FINAL_MATE_EVICT
B300_STAGEK_PROMOTION_SPEEDUP=$B300_STAGEK_FINAL_SPEEDUP
B300_STAGEK_PROMOTION_FINAL_STAGE_ROWS=$B300_STAGEK_FINAL_STAGE_ROWS
B300_STAGEK_PROMOTION_FINAL_STAGE_RESIDUE=$(printf '%q' "$B300_STAGEK_FINAL_STAGE_RESIDUE")
B300_STAGEK_PROMOTION_WINNER_ENV_SHA256=$WINNER_SHA
B300_STAGEK_PROMOTION_STAGE_F_ENV_SHA256=$STAGEF_SHA
B300_STAGEK_PROMOTION_STAGEJ_WINNER_ENV_SHA256=$STAGEJ_WINNER_SHA
B300_STAGEK_PROMOTION_STAGEJ_PREPARE_ENV_SHA256=$STAGEJ_PREPARE_SHA
B300_STAGEK_PROMOTION_STAGEJ_MANIFEST=$(printf '%q' "$B300_STAGEJ_PREPARED_MANIFEST")
B300_STAGEK_PROMOTION_STAGEJ_MANIFEST_SHA256=$STAGEJ_MANIFEST_SHA
B300_STAGEK_PROMOTION_MANIFEST=$(printf '%q' "$MANIFEST")
B300_STAGEK_PROMOTION_MANIFEST_SHA256=$MANIFEST_SHA
EOF
{
  printf 'B300_STAGEK_PREPARED=1\n'
  printf 'B300_STAGEK_PREPARED_MOD=%q\n' "$MOD"
  printf 'B300_STAGEK_PREPARED_BIN=%q\n' "$B300_STAGEK_FINAL_BIN"
  printf 'B300_STAGEK_PREPARED_LABEL=%q\n' "$label"
  printf 'B300_STAGEK_PREPARED_THREADS=%q\n' "$B300_STAGEK_FINAL_THREADS"
  printf 'B300_STAGEK_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGEK_CONTROL_BIN"
  printf 'B300_STAGEK_PREPARED_CONTROL_LABEL=%q\n' "$control_label"
  printf 'B300_STAGEK_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGEK_CONTROL_THREADS"
  printf 'B300_STAGEK_PREPARED_SELF_WIDTH=%q\n' "$B300_STAGEK_SELF_WIDTH"
  printf 'B300_STAGEK_PREPARED_SELF_DISTANCE=%q\n' "$B300_STAGEK_SELF_DISTANCE"
  printf 'B300_STAGEK_PREPARED_SELF_EVICT=%q\n' "$B300_STAGEK_SELF_EVICT"
  printf 'B300_STAGEK_PREPARED_MATE_WIDTH=%q\n' "$B300_STAGEK_MATE_WIDTH"
  printf 'B300_STAGEK_PREPARED_MATE_DISTANCE=%q\n' "$B300_STAGEK_MATE_DISTANCE"
  printf 'B300_STAGEK_PREPARED_BASE_MATE_EVICT=%q\n' "$B300_STAGEK_BASE_MATE_EVICT"
  printf 'B300_STAGEK_PREPARED_MATE_EVICT=%q\n' "$B300_STAGEK_FINAL_MATE_EVICT"
  printf 'B300_STAGEK_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGEK_FINAL_SPEEDUP"
  printf 'B300_STAGEK_PREPARED_STAGEJ_MANIFEST=%q\n' "$B300_STAGEJ_PREPARED_MANIFEST"
  printf 'B300_STAGEK_PREPARED_MANIFEST=%q\n' "$MANIFEST"
  printf 'B300_STAGEK_PREPARED_PROMOTION_ENV=%q\n' "$PROMOTION_ENV"
} >"$PREPARE_ENV"
if [[ "$PREPARE_ONLY" == 1 ]]; then
  cat "$PREPARE_ENV"
  echo "STAGE K PREPARED mod=$MOD mate_evict=${B300_STAGEK_FINAL_MATE_EVICT} baseline=${B300_STAGEK_BASE_MATE_EVICT} speedup=${B300_STAGEK_FINAL_SPEEDUP}x env=$PREPARE_ENV" >&2
  exit 0
fi

echo "=== Stage K full-prime race: $label vs $control_label vs profiled warp/orbit ===" >&2
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$MOD" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_STAGEK_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGEK_FINAL_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEK_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGEK_CONTROL_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
