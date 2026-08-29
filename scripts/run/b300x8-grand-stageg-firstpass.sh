#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand Stage-G first-pass targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -f "$PROFILE_FILE" ]] || exit 2
ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"; SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"; STAGEG_MIN_SPEEDUP="${STAGEG_MIN_SPEEDUP:-1.002}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_stageg_firstpass_n27}"
BASE_PREFIX="${BASE_PREFIX:-${PREFIX}.base}"; BASE_SELECTED_ENV="${BASE_SELECTED_ENV:-${BASE_PREFIX}.selected.env}"
STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"
STAGEG_PREFIX="${STAGEG_PREFIX:-${PREFIX}.stageg}"; STAGEG_ENV="${STAGEG_ENV:-${STAGEG_PREFIX}_winner.env}"
STAGEG_PREPARE_ENV="${STAGEG_PREPARE_ENV:-${PREFIX}.stageg.prepared.env}"; STAGEG_RACE_PREFIX="${STAGEG_RACE_PREFIX:-${PREFIX}.stageg.promote}"
FINAL_RACE_PREFIX="${FINAL_RACE_PREFIX:-${PREFIX}.race}"; FINAL_RESULT="${FINAL_RESULT:-${FINAL_RACE_PREFIX}.tsv}"; FINAL_LOG="${FINAL_LOG:-${PREFIX}.log}"; SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"
mkdir -p "$(dirname "$SELECTED_ENV")" "$(dirname "$FINAL_LOG")"

# First choose the best candidate from the existing grand family.
echo '=== Stage G grand: existing grand first-pass ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$BASE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh" 27
[[ -s "$BASE_SELECTED_ENV" ]] || { echo "base grand selected env missing: $BASE_SELECTED_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$BASE_SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]] || exit 3
BASE_BIN="$B300_GRAND_SELECTED_BINARY"; BASE_BACKEND="$B300_GRAND_SELECTED_BACKEND"; BASE_RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; BASE_THREADS="$B300_GRAND_SELECTED_THREADS"
[[ -x "$BASE_BIN" ]] || exit 3

# Stage G is meaningful only when Stage F produced a staged-valid composed candidate.
if [[ ! -s "$STAGE_F_ENV" ]]; then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEG_APPLICABLE=0\nB300_GRAND_STAGEG_REASON=no_stage_f_env\n' >>"$SELECTED_ENV"
  echo "grand Stage-G: no Stage-F env; existing grand winner retained: $BASE_BACKEND" >&2
  exit 0
fi

# New Stage F searches width and distance jointly against the plain hybrid
# control and already sends its selected geometry through the complete-prime
# grand race. Re-running the legacy fixed-width distance stage would duplicate
# B300 work without introducing a candidate that was not already considered.
# Keep this runner compatible with old width-only Stage-F artifacts: only skip
# when both geometry provenance fields are present.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
if [[ -n "${B300_HYBRID8_NEXTSELF_FINAL_DISTANCE+x}" && -n "${B300_HYBRID8_NEXTSELF_SEARCH_DISTANCES+x}" ]]; then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  {
    printf 'B300_GRAND_STAGEG_APPLICABLE=0\n'
    printf 'B300_GRAND_STAGEG_REASON=stage_f_geometry_complete\n'
    printf 'B300_GRAND_STAGEG_GEOMETRY_VALIDATED=%q\n' "${B300_HYBRID8_NEXTSELF_STAGED_VALIDATED:-0}"
    printf 'B300_GRAND_STAGEG_GEOMETRY_WIDTH=%q\n' "${B300_HYBRID8_NEXTSELF_FINAL_WIDTH:-0}"
    printf 'B300_GRAND_STAGEG_GEOMETRY_DISTANCE=%q\n' "${B300_HYBRID8_NEXTSELF_FINAL_DISTANCE:-0}"
    printf 'B300_GRAND_STAGEG_GEOMETRY_SEARCH_DISTANCES=%q\n' "${B300_HYBRID8_NEXTSELF_SEARCH_DISTANCES:-}"
  } >>"$SELECTED_ENV"
  echo "grand Stage-G: joint geometry already searched in Stage F; existing grand winner retained: $BASE_BACKEND" >&2
  exit 0
fi

echo '=== Stage G grand: legacy fixed-width distance staged prepare ===' >&2
set +e
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$SMOKE_PRIME" \
  RUN_STAGED=1 RUN_STAGE_F=0 PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEG_MIN_SPEEDUP" DISTANCE_LIST="$DISTANCE_LIST" \
  STAGED_PREFIX="$STAGEG_PREFIX" STAGE_F_ENV="$STAGE_F_ENV" WINNER_ENV="$STAGEG_ENV" RACE_PREFIX="$STAGEG_RACE_PREFIX" PREPARE_ENV="$STAGEG_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-distance-staged-fullprime-race.sh" 27
rc=$?
set -e
if ((rc==4)); then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEG_APPLICABLE=1\nB300_GRAND_STAGEG_PROMOTED=0\nB300_GRAND_STAGEG_REASON=staged_distance_rejected\n' >>"$SELECTED_ENV"
  echo "grand Stage-G: staged distance rejected; existing grand winner retained: $BASE_BACKEND" >&2
  exit 0
fi
((rc==0)) || exit "$rc"
[[ -s "$STAGEG_PREPARE_ENV" ]] || exit 3
# shellcheck disable=SC1090
source "$STAGEG_PREPARE_ENV"
[[ "${B300_STAGEG_PREPARED:-0}" == 1 ]] || exit 3

EXTRA_BIN=""; EXTRA_LABEL=""; EXTRA_THREADS=256
if [[ "$BASE_RUNTIME" == forced ]]; then
  EXTRA_BIN="$BASE_BIN"; EXTRA_LABEL="grand_previous_${BASE_BACKEND}"; EXTRA_THREADS="$BASE_THREADS"
fi

echo "=== Stage G grand final race: stageg w${B300_STAGEG_PREPARED_WIDTH} d${B300_STAGEG_PREPARED_DISTANCE} vs previous=$BASE_BACKEND ===" >&2
set +e
env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$SMOKE_PRIME" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_STAGEG_PREPARED_BIN" FORCED_OVERRIDE_LABEL="$B300_STAGEG_PREPARED_LABEL" FORCED_OVERRIDE_THREADS="$B300_STAGEG_PREPARED_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEG_PREPARED_REFERENCE_BIN" FORCED_BASE_LABEL="$B300_STAGEG_PREPARED_REFERENCE_LABEL" FORCED_BASE_THREADS="$B300_STAGEG_PREPARED_REFERENCE_THREADS" \
  FORCED_EXTRA_BIN="$EXTRA_BIN" FORCED_EXTRA_LABEL="$EXTRA_LABEL" FORCED_EXTRA_THREADS="$EXTRA_THREADS" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$FINAL_RACE_PREFIX" RESULT="$FINAL_RESULT" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@" 2>&1 | tee "$FINAL_LOG"
rc=${PIPESTATUS[0]}
set -e
((rc==0)) || exit "$rc"
[[ -s "$FINAL_RESULT" ]] || exit 4

WIN="$(python3 - "$FINAL_RESULT" <<'PY'
import csv,sys
ok=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['status']=='ok']
if not ok: raise SystemExit('no Stage-G grand final candidates')
res={r['residue'] for r in ok}
if len(res)!=1: raise SystemExit('Stage-G grand residue mismatch')
b=min(ok,key=lambda r:float(r['wall_s']))
print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"
[[ -x "$BEST_BIN" ]] || exit 4
BEST_SHA="$(sha256sum "$BEST_BIN" | awk '{print $1}')"; SHA12="${BEST_SHA:0:12}"; BEST_WORK="$ONEESAN_ROOT/work/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; CHECKPOINT="$BEST_WORK/checkpoint.json"
[[ -s "$CHECKPOINT" ]] || exit 4
RUNTIME=forced; THREADS=0; RUN_TARGET="$FORCED_TARGET_MIB"
if [[ "$BEST_BIN" == "$B300_STAGEG_PREPARED_BIN" ]]; then THREADS="$B300_STAGEG_PREPARED_THREADS";
elif [[ "$BEST_BIN" == "$B300_STAGEG_PREPARED_REFERENCE_BIN" ]]; then THREADS="$B300_STAGEG_PREPARED_REFERENCE_THREADS";
elif [[ -n "$EXTRA_BIN" && "$BEST_BIN" == "$EXTRA_BIN" ]]; then THREADS="$EXTRA_THREADS";
elif [[ "$BEST" == warp_tuned ]]; then RUNTIME=warp; RUN_TARGET="$BUCKET_TARGET_MIB";
else RUNTIME=orbit; RUN_TARGET="$BUCKET_TARGET_MIB"; fi
{
  printf 'B300_GRAND_STAGEG_SELECTED_SCHEMA=1\n'
  printf 'B300_GRAND_STAGEG_SELECTED_VALIDATED=1\n'
  printf 'B300_GRAND_STAGEG_APPLICABLE=1\n'
  printf 'B300_GRAND_STAGEG_PROMOTED=%q\n' "$([[ "$BEST_BIN" == "$B300_STAGEG_PREPARED_BIN" ]] && echo 1 || echo 0)"
  printf 'B300_GRAND_STAGEG_WIDTH=%q\n' "$B300_STAGEG_PREPARED_WIDTH"
  printf 'B300_GRAND_STAGEG_DISTANCE=%q\n' "$B300_STAGEG_PREPARED_DISTANCE"
  printf 'B300_GRAND_STAGEG_PREPARE_ENV=%q\n' "$STAGEG_PREPARE_ENV"
  printf 'B300_GRAND_STAGEG_BASE_SELECTED_ENV=%q\n' "$BASE_SELECTED_ENV"
  printf 'B300_GRAND_STAGEG_SELECTED_BACKEND=%q\n' "$BEST"
  printf 'B300_GRAND_STAGEG_SELECTED_PROFILE=%q\n' "$BEST_PROFILE"
  printf 'B300_GRAND_STAGEG_SELECTED_BINARY=%q\n' "$BEST_BIN"
  printf 'B300_GRAND_STAGEG_SELECTED_BINARY_SHA256=%q\n' "$BEST_SHA"
  printf 'B300_GRAND_STAGEG_SELECTED_RESIDUE=%q\n' "$BEST_RES"
  printf 'B300_GRAND_STAGEG_SELECTED_WALL_S=%q\n' "$BEST_WALL"
  printf 'B300_GRAND_STAGEG_SELECTED_RUNTIME_KIND=%q\n' "$RUNTIME"
  printf 'B300_GRAND_STAGEG_SELECTED_THREADS=%q\n' "$THREADS"
  printf 'B300_GRAND_STAGEG_SELECTED_TARGET_MIB=%q\n' "$RUN_TARGET"
  printf 'B300_GRAND_STAGEG_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"
  printf 'B300_GRAND_STAGEG_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'B300_GRAND_STAGEG_SELECTED_RACE_RESULT=%q\n' "$FINAL_RESULT"
} >"$SELECTED_ENV"
cat "$SELECTED_ENV"
echo "b300x8-grand-stageg-firstpass OK backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL stageg_width=$B300_STAGEG_PREPARED_WIDTH stageg_distance=$B300_STAGEG_PREPARED_DISTANCE selected_env=$SELECTED_ENV" >&2
