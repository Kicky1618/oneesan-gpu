#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand Stage-H first-pass targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -f "$PROFILE_FILE" ]] || exit 2
ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"; SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
STAGEH_MIN_SPEEDUP="${STAGEH_MIN_SPEEDUP:-1.002}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_stageh_firstpass_n27}"
BASE_PREFIX="${BASE_PREFIX:-${PREFIX}.base}"; BASE_SELECTED_ENV="${BASE_SELECTED_ENV:-${BASE_PREFIX}.selected.env}"; STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"
BASE_GRAND_ENV="${BASE_GRAND_ENV:-${BASE_PREFIX}.race_grand.env}"
INTEGRATED_STAGEJ_PREPARE="${INTEGRATED_STAGEJ_PREPARE:-${BASE_PREFIX}.stagej-mategeo.prepared.env}"
INTEGRATED_LEGACY_STAGEH_PREPARE="${INTEGRATED_LEGACY_STAGEH_PREPARE:-${BASE_PREFIX}.stageh-nextmate.prepared.env}"
STAGEH_PREFIX="${STAGEH_PREFIX:-${PREFIX}.stageh}"; STAGEH_ENV="${STAGEH_ENV:-${STAGEH_PREFIX}_winner.env}"; STAGEH_PREPARE_ENV="${STAGEH_PREPARE_ENV:-${PREFIX}.stageh.prepared.env}"; STAGEH_RACE_PREFIX="${STAGEH_RACE_PREFIX:-${PREFIX}.stageh.promote}"
FINAL_RACE_PREFIX="${FINAL_RACE_PREFIX:-${PREFIX}.race}"; FINAL_RESULT="${FINAL_RESULT:-${FINAL_RACE_PREFIX}.tsv}"; FINAL_LOG="${FINAL_LOG:-${PREFIX}.log}"; SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"
STAGEH_META="${STAGEH_META:-${PREFIX}.meta}"
mkdir -p "$(dirname "$SELECTED_ENV")" "$(dirname "$FINAL_LOG")" "$WORK_ROOT"

echo '=== Stage H compatibility: existing grand first-pass ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" REBUILD_BUCKETS="$REBUILD_BUCKETS" WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX" \
  STAGEH_MIN_SPEEDUP="$STAGEH_MIN_SPEEDUP" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh" 27
[[ -s "$BASE_SELECTED_ENV" ]] || { echo "base selected env missing: $BASE_SELECTED_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$BASE_SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]] || exit 3
BASE_BIN="$B300_GRAND_SELECTED_BINARY"; BASE_BACKEND="$B300_GRAND_SELECTED_BACKEND"; BASE_RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; BASE_THREADS="$B300_GRAND_SELECTED_THREADS"
BASE_HEAD_SHA="$B300_GRAND_SELECTED_HEAD_SHA"; BASE_HEAD_DIRTY="$B300_GRAND_SELECTED_HEAD_DIRTY"
BASE_PROFILE_FILE="$B300_GRAND_SELECTED_PROFILE_FILE"; BASE_PROFILE_SHA="$B300_GRAND_SELECTED_PROFILE_SHA256"
BASE_SMOKE_PRIME="$B300_GRAND_SELECTED_SMOKE_PRIME"; BASE_MAX_WINDOW="$B300_GRAND_SELECTED_MAX_WINDOW"
[[ -x "$BASE_BIN" ]] || exit 3

# Current grand selectors integrate mate geometry as Stage J and expose the old
# B300_GRAND_STAGEH_* keys as compatibility aliases. If that provenance is
# present, never stage/race the legacy Stage-H path a second time.
if [[ -s "$BASE_GRAND_ENV" ]] && grep -q '^B300_GRAND_STAGEH_OK=' "$BASE_GRAND_ENV"; then
  # shellcheck disable=SC1090
  source "$BASE_GRAND_ENV"
  APPLICABLE="${B300_GRAND_HYBRID8_NEXTSELF_OK:-0}"
  PROMOTED=0
  if [[ "${B300_GRAND_STAGEH_OK:-0}" == 1 && "$BASE_BIN" == "$B300_GRAND_PRIMARY_BIN" ]]; then
    case "$B300_GRAND_MODE" in
      stagej_mategeo_*|stageh_nextmate_*) PROMOTED=1 ;;
    esac
  fi
  H_SPEED=0
  H_HIGH=0
  H_PREPARE=""
  if [[ -s "$INTEGRATED_STAGEJ_PREPARE" ]]; then
    # shellcheck disable=SC1090
    source "$INTEGRATED_STAGEJ_PREPARE"
    if [[ "${B300_STAGEJ_PREPARED:-0}" == 1 ]]; then
      H_SPEED="${B300_STAGEJ_PREPARED_STAGED_SPEEDUP:-0}"
      H_PREPARE="$INTEGRATED_STAGEJ_PREPARE"
    fi
  elif [[ -s "$INTEGRATED_LEGACY_STAGEH_PREPARE" ]]; then
    # shellcheck disable=SC1090
    source "$INTEGRATED_LEGACY_STAGEH_PREPARE"
    if [[ "${B300_STAGEH_PREPARED:-0}" == 1 ]]; then
      H_SPEED="${B300_STAGEH_PREPARED_SPEEDUP:-0}"
      H_HIGH="${B300_STAGEH_PREPARED_HIGH_S:-0}"
      H_PREPARE="$INTEGRATED_LEGACY_STAGEH_PREPARE"
    fi
  fi
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  {
    printf 'B300_GRAND_STAGEH_SELECTED_SCHEMA=1\n'
    printf 'B300_GRAND_STAGEH_SELECTED_VALIDATED=1\n'
    printf 'B300_GRAND_STAGEH_APPLICABLE=%q\n' "$APPLICABLE"
    printf 'B300_GRAND_STAGEH_PROMOTED=%q\n' "$PROMOTED"
    printf 'B300_GRAND_STAGEH_REASON=integrated_in_grand\n'
    printf 'B300_GRAND_STAGEH_WIDTH=%q\n' "${B300_GRAND_STAGEH_WIDTH:-0}"
    printf 'B300_GRAND_STAGEH_DISTANCE=%q\n' "${B300_GRAND_STAGEH_DISTANCE:-0}"
    printf 'B300_GRAND_STAGEH_SPEEDUP=%q\n' "$H_SPEED"
    printf 'B300_GRAND_STAGEH_HIGH_S=%q\n' "$H_HIGH"
    printf 'B300_GRAND_STAGEH_BASE_SELECTED_ENV=%q\n' "$BASE_SELECTED_ENV"
    printf 'B300_GRAND_STAGEH_PREPARE_ENV=%q\n' "$H_PREPARE"
    printf 'B300_GRAND_STAGEH_SELECTED_BACKEND=%q\n' "$B300_GRAND_SELECTED_BACKEND"
    printf 'B300_GRAND_STAGEH_SELECTED_PROFILE=%q\n' "$B300_GRAND_SELECTED_PROFILE"
    printf 'B300_GRAND_STAGEH_SELECTED_BINARY=%q\n' "$B300_GRAND_SELECTED_BINARY"
    printf 'B300_GRAND_STAGEH_SELECTED_BINARY_SHA256=%q\n' "$B300_GRAND_SELECTED_BINARY_SHA256"
    printf 'B300_GRAND_STAGEH_SELECTED_RESIDUE=%q\n' "$B300_GRAND_SELECTED_RESIDUE"
    printf 'B300_GRAND_STAGEH_SELECTED_WALL_S=%q\n' "$B300_GRAND_SELECTED_WALL_S"
    printf 'B300_GRAND_STAGEH_SELECTED_RUNTIME_KIND=%q\n' "$B300_GRAND_SELECTED_RUNTIME_KIND"
    printf 'B300_GRAND_STAGEH_SELECTED_THREADS=%q\n' "$B300_GRAND_SELECTED_THREADS"
    printf 'B300_GRAND_STAGEH_SELECTED_TARGET_MIB=%q\n' "$B300_GRAND_SELECTED_TARGET_MIB"
    printf 'B300_GRAND_STAGEH_SELECTED_WORK_DIR=%q\n' "$B300_GRAND_SELECTED_WORK_DIR"
    printf 'B300_GRAND_STAGEH_SELECTED_CHECKPOINT=%q\n' "$B300_GRAND_SELECTED_CHECKPOINT"
    printf 'B300_GRAND_STAGEH_SELECTED_RACE_RESULT=%q\n' "$B300_GRAND_SELECTED_RACE_RESULT"
    printf 'B300_GRAND_STAGEH_SELECTED_RACE_RESULT_SHA256=%q\n' "$B300_GRAND_SELECTED_RACE_RESULT_SHA256"
  } >>"$SELECTED_ENV"
  echo "b300x8-grand-stageh-firstpass OK backend=$BASE_BACKEND promoted=$PROMOTED reason=integrated_in_grand selected_env=$SELECTED_ENV normalized_contract=1" >&2
  exit 0
fi

# Legacy compatibility for pre-integration grand artifacts.
if [[ ! -s "$STAGE_F_ENV" ]]; then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEH_APPLICABLE=0\nB300_GRAND_STAGEH_REASON=no_stage_f_env\n' >>"$SELECTED_ENV"
  echo "Stage H skipped: no Stage-F env; retained $BASE_BACKEND" >&2
  exit 0
fi

set +e
PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$BASE_MAX_WINDOW" MOD="$BASE_SMOKE_PRIME" INPUT_ENV="$STAGE_F_ENV" \
  RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEH_MIN_SPEEDUP" STAGED_PREFIX="$STAGEH_PREFIX" WINNER_ENV="$STAGEH_ENV" RACE_PREFIX="$STAGEH_RACE_PREFIX" PREPARE_ENV="$STAGEH_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh" 27
rc=$?
set -e
if ((rc==4)); then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEH_APPLICABLE=1\nB300_GRAND_STAGEH_PROMOTED=0\nB300_GRAND_STAGEH_REASON=staged_mate_rejected\n' >>"$SELECTED_ENV"
  echo "Stage H rejected; retained $BASE_BACKEND" >&2
  exit 0
fi
((rc==0)) || exit "$rc"
[[ -s "$STAGEH_PREPARE_ENV" ]] || exit 3
# shellcheck disable=SC1090
source "$STAGEH_PREPARE_ENV"; [[ "${B300_STAGEH_PREPARED:-0}" == 1 ]] || exit 3
EXTRA_BIN=""; EXTRA_LABEL=""; EXTRA_THREADS=256
if [[ "$BASE_RUNTIME" == forced && "$BASE_BIN" != "$B300_STAGEH_PREPARED_BIN" && "$BASE_BIN" != "$B300_STAGEH_PREPARED_CONTROL_BIN" ]]; then
  EXTRA_BIN="$BASE_BIN"; EXTRA_LABEL="grand_previous_${BASE_BACKEND}"; EXTRA_THREADS="$BASE_THREADS"
fi

echo "=== Stage H grand final race: mate w${B300_STAGEH_PREPARED_WIDTH}d${B300_STAGEH_PREPARED_DISTANCE} vs self vs previous=$BASE_BACKEND ===" >&2
set +e
env PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$BASE_SMOKE_PRIME" MAX_WINDOW="$BASE_MAX_WINDOW" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" \
  FORCED_OVERRIDE_BIN="$B300_STAGEH_PREPARED_BIN" FORCED_OVERRIDE_LABEL="$B300_STAGEH_PREPARED_LABEL" FORCED_OVERRIDE_THREADS="$B300_STAGEH_PREPARED_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEH_PREPARED_CONTROL_BIN" FORCED_BASE_LABEL="$B300_STAGEH_PREPARED_CONTROL_LABEL" FORCED_BASE_THREADS="$B300_STAGEH_PREPARED_CONTROL_THREADS" \
  FORCED_EXTRA_BIN="$EXTRA_BIN" FORCED_EXTRA_LABEL="$EXTRA_LABEL" FORCED_EXTRA_THREADS="$EXTRA_THREADS" SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$FINAL_RACE_PREFIX" RESULT="$FINAL_RESULT" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@" 2>&1 | tee "$FINAL_LOG"
rc=${PIPESTATUS[0]}; set -e; ((rc==0)) || exit "$rc"; [[ -s "$FINAL_RESULT" ]] || exit 4

WIN="$(python3 - "$FINAL_RESULT" <<'PY'
import csv,sys
ok=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['status']=='ok']
if not ok: raise SystemExit('no Stage-H final candidates')
if len({r['residue'] for r in ok})!=1: raise SystemExit('Stage-H final residue mismatch')
b=min(ok,key=lambda r:float(r['wall_s']))
print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"; [[ -x "$BEST_BIN" ]] || exit 4
BEST_SHA="$(sha256sum "$BEST_BIN" | awk '{print $1}')"; SHA12="${BEST_SHA:0:12}"; BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; CHECKPOINT="$BEST_WORK/checkpoint.json"; [[ -s "$CHECKPOINT" ]] || exit 4
FINAL_RESULT_SHA="$(sha256sum "$FINAL_RESULT" | awk '{print $1}')"
RUNTIME=forced; THREADS=0; RUN_TARGET="$FORCED_TARGET_MIB"
if [[ "$BEST_BIN" == "$B300_STAGEH_PREPARED_BIN" ]]; then THREADS="$B300_STAGEH_PREPARED_THREADS";
elif [[ "$BEST_BIN" == "$B300_STAGEH_PREPARED_CONTROL_BIN" ]]; then THREADS="$B300_STAGEH_PREPARED_CONTROL_THREADS";
elif [[ -n "$EXTRA_BIN" && "$BEST_BIN" == "$EXTRA_BIN" ]]; then THREADS="$EXTRA_THREADS";
elif [[ "$BEST" == warp_tuned ]]; then RUNTIME=warp; RUN_TARGET="$BUCKET_TARGET_MIB";
else RUNTIME=orbit; RUN_TARGET="$BUCKET_TARGET_MIB"; fi
PROMOTED=0; [[ "$BEST_BIN" == "$B300_STAGEH_PREPARED_BIN" ]] && PROMOTED=1

python3 - "$CHECKPOINT" "$BEST_SHA" "$BASE_PROFILE_SHA" "$BASE_SMOKE_PRIME" "$BEST_RES" <<'PY'
import json,sys
cp,bsha,psha,prime,residue=sys.argv[1:]
d=json.load(open(cp))
if int(d.get('n',-1)) != 27: raise SystemExit('Stage-H checkpoint n mismatch')
if d.get('solver_fingerprint') != {'schema':3,'binary_sha256':bsha,'profile_sha256':psha}:
    raise SystemExit('Stage-H checkpoint fingerprint mismatch')
r=d.get('residues',{}).get(str(int(prime)))
if not r or int(r.get('residue',-1)) != int(residue): raise SystemExit('Stage-H checkpoint smoke residue mismatch')
PY

{
  printf 'schema=1\n'
  printf 'stageh_validated=1\n'
  printf 'base_selected_env=%s\n' "$BASE_SELECTED_ENV"
  printf 'stageh_prepare_env=%s\n' "$STAGEH_PREPARE_ENV"
  printf 'stageh_width=%s\n' "$B300_STAGEH_PREPARED_WIDTH"
  printf 'stageh_distance=%s\n' "$B300_STAGEH_PREPARED_DISTANCE"
  printf 'stageh_speedup=%s\n' "$B300_STAGEH_PREPARED_SPEEDUP"
  printf 'stageh_high_s=%s\n' "$B300_STAGEH_PREPARED_HIGH_S"
  printf 'selected_backend=%s\n' "$BEST"
  printf 'selected_profile=%s\n' "$BEST_PROFILE"
  printf 'selected_binary=%s\n' "$BEST_BIN"
  printf 'selected_binary_sha256=%s\n' "$BEST_SHA"
  printf 'selected_runtime_kind=%s\n' "$RUNTIME"
  printf 'selected_work_dir=%s\n' "$BEST_WORK"
  printf 'race_result=%s\n' "$FINAL_RESULT"
  printf 'race_result_sha256=%s\n' "$FINAL_RESULT_SHA"
} >"$STAGEH_META"

{
  printf 'B300_GRAND_SELECTED_SCHEMA=1\n'
  printf 'B300_GRAND_SELECTED_VALIDATED=1\n'
  printf 'B300_GRAND_SELECTED_N=27\n'
  printf 'B300_GRAND_SELECTED_HEAD_SHA=%q\n' "$BASE_HEAD_SHA"
  printf 'B300_GRAND_SELECTED_HEAD_DIRTY=%q\n' "$BASE_HEAD_DIRTY"
  printf 'B300_GRAND_SELECTED_PROFILE_FILE=%q\n' "$BASE_PROFILE_FILE"
  printf 'B300_GRAND_SELECTED_PROFILE_SHA256=%q\n' "$BASE_PROFILE_SHA"
  printf 'B300_GRAND_SELECTED_BACKEND=%q\n' "$BEST"
  printf 'B300_GRAND_SELECTED_PROFILE=%q\n' "$BEST_PROFILE"
  printf 'B300_GRAND_SELECTED_BINARY=%q\n' "$BEST_BIN"
  printf 'B300_GRAND_SELECTED_BINARY_SHA256=%q\n' "$BEST_SHA"
  printf 'B300_GRAND_SELECTED_RESIDUE=%q\n' "$BEST_RES"
  printf 'B300_GRAND_SELECTED_WALL_S=%q\n' "$BEST_WALL"
  printf 'B300_GRAND_SELECTED_SMOKE_PRIME=%q\n' "$BASE_SMOKE_PRIME"
  printf 'B300_GRAND_SELECTED_RUNTIME_KIND=%q\n' "$RUNTIME"
  printf 'B300_GRAND_SELECTED_THREADS=%q\n' "$THREADS"
  printf 'B300_GRAND_SELECTED_TARGET_MIB=%q\n' "$RUN_TARGET"
  printf 'B300_GRAND_SELECTED_MAX_WINDOW=%q\n' "$BASE_MAX_WINDOW"
  printf 'B300_GRAND_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"
  printf 'B300_GRAND_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'B300_GRAND_SELECTED_RACE_PREFIX=%q\n' "$FINAL_RACE_PREFIX"
  printf 'B300_GRAND_SELECTED_RACE_RESULT=%q\n' "$FINAL_RESULT"
  printf 'B300_GRAND_SELECTED_RACE_RESULT_SHA256=%q\n' "$FINAL_RESULT_SHA"
  printf 'B300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$STAGEH_META"
  printf 'B300_GRAND_STAGEH_SELECTED_SCHEMA=1\n'
  printf 'B300_GRAND_STAGEH_SELECTED_VALIDATED=1\n'
  printf 'B300_GRAND_STAGEH_APPLICABLE=1\n'
  printf 'B300_GRAND_STAGEH_PROMOTED=%q\n' "$PROMOTED"
  printf 'B300_GRAND_STAGEH_WIDTH=%q\n' "$B300_STAGEH_PREPARED_WIDTH"
  printf 'B300_GRAND_STAGEH_DISTANCE=%q\n' "$B300_STAGEH_PREPARED_DISTANCE"
  printf 'B300_GRAND_STAGEH_SPEEDUP=%q\n' "$B300_STAGEH_PREPARED_SPEEDUP"
  printf 'B300_GRAND_STAGEH_HIGH_S=%q\n' "$B300_STAGEH_PREPARED_HIGH_S"
  printf 'B300_GRAND_STAGEH_BASE_SELECTED_ENV=%q\n' "$BASE_SELECTED_ENV"
  printf 'B300_GRAND_STAGEH_PREPARE_ENV=%q\n' "$STAGEH_PREPARE_ENV"
  printf 'B300_GRAND_STAGEH_SELECTED_BACKEND=%q\n' "$BEST"
  printf 'B300_GRAND_STAGEH_SELECTED_PROFILE=%q\n' "$BEST_PROFILE"
  printf 'B300_GRAND_STAGEH_SELECTED_BINARY=%q\n' "$BEST_BIN"
  printf 'B300_GRAND_STAGEH_SELECTED_BINARY_SHA256=%q\n' "$BEST_SHA"
  printf 'B300_GRAND_STAGEH_SELECTED_RESIDUE=%q\n' "$BEST_RES"
  printf 'B300_GRAND_STAGEH_SELECTED_WALL_S=%q\n' "$BEST_WALL"
  printf 'B300_GRAND_STAGEH_SELECTED_RUNTIME_KIND=%q\n' "$RUNTIME"
  printf 'B300_GRAND_STAGEH_SELECTED_THREADS=%q\n' "$THREADS"
  printf 'B300_GRAND_STAGEH_SELECTED_TARGET_MIB=%q\n' "$RUN_TARGET"
  printf 'B300_GRAND_STAGEH_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"
  printf 'B300_GRAND_STAGEH_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'B300_GRAND_STAGEH_SELECTED_RACE_RESULT=%q\n' "$FINAL_RESULT"
  printf 'B300_GRAND_STAGEH_SELECTED_RACE_RESULT_SHA256=%q\n' "$FINAL_RESULT_SHA"
} >"$SELECTED_ENV"
cat "$SELECTED_ENV"
echo "b300x8-grand-stageh-firstpass OK backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL promoted=$PROMOTED width=$B300_STAGEH_PREPARED_WIDTH distance=$B300_STAGEH_PREPARED_DISTANCE selected_env=$SELECTED_ENV normalized_contract=1" >&2