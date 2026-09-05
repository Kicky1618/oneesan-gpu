#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand Stage-I first-pass targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -s "$PROFILE_FILE" ]] || exit 2
ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"; SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
STAGEI_MIN_SPEEDUP="${STAGEI_MIN_SPEEDUP:-1.002}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"; MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"
SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_stagei_firstpass_n27}"
BASE_PREFIX="${BASE_PREFIX:-${PREFIX}.base}"
BASE_SELECTED_ENV="${BASE_SELECTED_ENV:-${BASE_PREFIX}.selected.env}"
STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"
STAGEI_PREFIX="${STAGEI_PREFIX:-${PREFIX}.stagei}"
STAGEI_ENV="${STAGEI_ENV:-${STAGEI_PREFIX}_winner.env}"
STAGEI_PREPARE_ENV="${STAGEI_PREPARE_ENV:-${PREFIX}.stagei.prepared.env}"
STAGEI_RACE_PREFIX="${STAGEI_RACE_PREFIX:-${PREFIX}.stagei.promote}"
FINAL_RACE_PREFIX="${FINAL_RACE_PREFIX:-${PREFIX}.race}"
FINAL_RESULT="${FINAL_RESULT:-${FINAL_RACE_PREFIX}.tsv}"
FINAL_LOG="${FINAL_LOG:-${PREFIX}.log}"
SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"
STAGEI_META="${STAGEI_META:-${PREFIX}.meta}"
mkdir -p "$(dirname "$SELECTED_ENV")" "$(dirname "$FINAL_LOG")" "$WORK_ROOT"
for ev in SELF_EVICT MATE_EVICT; do case "${!ev}" in default|normal|last) ;; *) echo "$ev must be default,normal,last" >&2; exit 2;; esac; done
python3 - "$STAGEI_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('STAGEI_MIN_SPEEDUP must be >=1')
PY

echo '=== Stage I grand: existing grand first-pass (includes integrated Stage H) ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" \
  FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh" 27
[[ -s "$BASE_SELECTED_ENV" ]] || { echo "base grand selected env missing: $BASE_SELECTED_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$BASE_SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]] || { echo 'base grand selection not validated' >&2; exit 3; }
BASE_BIN="$B300_GRAND_SELECTED_BINARY"; BASE_BACKEND="$B300_GRAND_SELECTED_BACKEND"; BASE_PROFILE="$B300_GRAND_SELECTED_PROFILE"
BASE_RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; BASE_THREADS="$B300_GRAND_SELECTED_THREADS"
BASE_HEAD_SHA="$B300_GRAND_SELECTED_HEAD_SHA"; BASE_HEAD_DIRTY="$B300_GRAND_SELECTED_HEAD_DIRTY"
BASE_PROFILE_FILE="$B300_GRAND_SELECTED_PROFILE_FILE"; BASE_PROFILE_SHA="$B300_GRAND_SELECTED_PROFILE_SHA256"
BASE_SMOKE_PRIME="$B300_GRAND_SELECTED_SMOKE_PRIME"; BASE_MAX_WINDOW="$B300_GRAND_SELECTED_MAX_WINDOW"
[[ -x "$BASE_BIN" ]] || { echo 'base grand selected binary missing' >&2; exit 3; }

if [[ ! -s "$STAGE_F_ENV" ]]; then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEI_APPLICABLE=0\nB300_GRAND_STAGEI_REASON=no_stage_f_env\n' >>"$SELECTED_ENV"
  echo "Stage I skipped: no Stage-F env; retained $BASE_BACKEND" >&2
  exit 0
fi

echo '=== Stage I grand: independent mate-geometry staged prepare ===' >&2
set +e
PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$BASE_MAX_WINDOW" MOD="$BASE_SMOKE_PRIME" \
  INPUT_ENV="$STAGE_F_ENV" RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEI_MIN_SPEEDUP" \
  MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" \
  STAGED_PREFIX="$STAGEI_PREFIX" WINNER_ENV="$STAGEI_ENV" RACE_PREFIX="$STAGEI_RACE_PREFIX" PREPARE_ENV="$STAGEI_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-staged-fullprime-race.sh" 27
rc=$?
set -e
if ((rc==4)); then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEI_APPLICABLE=1\nB300_GRAND_STAGEI_PROMOTED=0\nB300_GRAND_STAGEI_REASON=staged_geometry_rejected\n' >>"$SELECTED_ENV"
  echo "Stage I rejected by staged gates; retained $BASE_BACKEND" >&2
  exit 0
fi
((rc==0)) || exit "$rc"
[[ -s "$STAGEI_PREPARE_ENV" ]] || { echo 'Stage-I prepare env missing' >&2; exit 3; }
# shellcheck disable=SC1090
source "$STAGEI_PREPARE_ENV"
[[ "${B300_STAGEI_PREPARED:-0}" == 1 ]] || { echo 'Stage-I prepared marker missing' >&2; exit 3; }
[[ -x "$B300_STAGEI_PREPARED_BIN" && -x "$B300_STAGEI_PREPARED_CONTROL_BIN" ]] || { echo 'Stage-I prepared binary missing' >&2; exit 3; }

# Keep the already-selected grand forced winner as a third forced candidate when
# it is distinct. If the prior grand winner is warp/orbit it is already present
# through the profiled candidates in the external race, so no duplicate slot is
# consumed.
EXTRA_BIN=""; EXTRA_LABEL=""; EXTRA_THREADS=256
if [[ "$BASE_RUNTIME" == forced && "$BASE_BIN" != "$B300_STAGEI_PREPARED_BIN" && "$BASE_BIN" != "$B300_STAGEI_PREPARED_CONTROL_BIN" ]]; then
  EXTRA_BIN="$BASE_BIN"; EXTRA_LABEL="grand_previous_${BASE_BACKEND}_${BASE_PROFILE}"; EXTRA_THREADS="$BASE_THREADS"
fi

echo "=== Stage I grand final race: self=w${B300_STAGEI_PREPARED_SELF_WIDTH}d${B300_STAGEI_PREPARED_SELF_DISTANCE} mate=w${B300_STAGEI_PREPARED_MATE_WIDTH}d${B300_STAGEI_PREPARED_MATE_DISTANCE} vs previous=$BASE_BACKEND ===" >&2
set +e
env PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$BASE_SMOKE_PRIME" MAX_WINDOW="$BASE_MAX_WINDOW" \
  FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" \
  FORCED_OVERRIDE_BIN="$B300_STAGEI_PREPARED_BIN" FORCED_OVERRIDE_LABEL="$B300_STAGEI_PREPARED_LABEL" FORCED_OVERRIDE_THREADS="$B300_STAGEI_PREPARED_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEI_PREPARED_CONTROL_BIN" FORCED_BASE_LABEL="$B300_STAGEI_PREPARED_CONTROL_LABEL" FORCED_BASE_THREADS="$B300_STAGEI_PREPARED_CONTROL_THREADS" \
  FORCED_EXTRA_BIN="$EXTRA_BIN" FORCED_EXTRA_LABEL="$EXTRA_LABEL" FORCED_EXTRA_THREADS="$EXTRA_THREADS" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$FINAL_RACE_PREFIX" RESULT="$FINAL_RESULT" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@" 2>&1 | tee "$FINAL_LOG"
rc=${PIPESTATUS[0]}
set -e
((rc==0)) || exit "$rc"
[[ -s "$FINAL_RESULT" ]] || { echo 'Stage-I grand final TSV missing' >&2; exit 4; }

WIN="$(python3 - "$FINAL_RESULT" <<'PY'
import csv,sys
ok=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['status']=='ok']
if not ok: raise SystemExit('no Stage-I grand final candidates')
if len({r['residue'] for r in ok}) != 1: raise SystemExit('Stage-I grand final residue mismatch')
b=min(ok,key=lambda r:float(r['wall_s']))
print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"
[[ -x "$BEST_BIN" ]] || { echo 'Stage-I selected binary missing' >&2; exit 4; }
BEST_SHA="$(sha256sum "$BEST_BIN" | awk '{print $1}')"; SHA12="${BEST_SHA:0:12}"
BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"
CHECKPOINT="$BEST_WORK/checkpoint.json"; [[ -s "$CHECKPOINT" ]] || { echo 'Stage-I selected checkpoint missing' >&2; exit 4; }
FINAL_RESULT_SHA="$(sha256sum "$FINAL_RESULT" | awk '{print $1}')"

RUNTIME=forced; THREADS=0; RUN_TARGET="$FORCED_TARGET_MIB"
if [[ "$BEST_BIN" == "$B300_STAGEI_PREPARED_BIN" ]]; then
  THREADS="$B300_STAGEI_PREPARED_THREADS"
elif [[ "$BEST_BIN" == "$B300_STAGEI_PREPARED_CONTROL_BIN" ]]; then
  THREADS="$B300_STAGEI_PREPARED_CONTROL_THREADS"
elif [[ -n "$EXTRA_BIN" && "$BEST_BIN" == "$EXTRA_BIN" ]]; then
  THREADS="$EXTRA_THREADS"
elif [[ "$BEST" == warp_tuned ]]; then
  RUNTIME=warp; RUN_TARGET="$BUCKET_TARGET_MIB"
elif [[ "$BEST" == orbit_tuned ]]; then
  RUNTIME=orbit; RUN_TARGET="$BUCKET_TARGET_MIB"
else
  echo "cannot map Stage-I final winner runtime backend=$BEST profile=$BEST_PROFILE" >&2; exit 4
fi
PROMOTED=0; [[ "$BEST_BIN" == "$B300_STAGEI_PREPARED_BIN" ]] && PROMOTED=1

python3 - "$CHECKPOINT" "$BEST_SHA" "$BASE_PROFILE_SHA" "$BASE_SMOKE_PRIME" "$BEST_RES" <<'PY'
import json,sys
cp,bsha,psha,prime,residue=sys.argv[1:]
d=json.load(open(cp))
if int(d.get('n',-1)) != 27: raise SystemExit('Stage-I grand checkpoint n mismatch')
if d.get('solver_fingerprint') != {'schema':3,'binary_sha256':bsha,'profile_sha256':psha}:
    raise SystemExit('Stage-I grand checkpoint fingerprint mismatch')
r=d.get('residues',{}).get(str(int(prime)))
if not r or int(r.get('residue',-1)) != int(residue): raise SystemExit('Stage-I grand checkpoint smoke residue mismatch')
PY

{
  printf 'schema=1\n'
  printf 'stagei_validated=1\n'
  printf 'base_selected_env=%s\n' "$BASE_SELECTED_ENV"
  printf 'stage_f_env=%s\n' "$STAGE_F_ENV"
  printf 'stagei_prepare_env=%s\n' "$STAGEI_PREPARE_ENV"
  printf 'stagei_self_width=%s\n' "$B300_STAGEI_PREPARED_SELF_WIDTH"
  printf 'stagei_self_distance=%s\n' "$B300_STAGEI_PREPARED_SELF_DISTANCE"
  printf 'stagei_mate_width=%s\n' "$B300_STAGEI_PREPARED_MATE_WIDTH"
  printf 'stagei_mate_distance=%s\n' "$B300_STAGEI_PREPARED_MATE_DISTANCE"
  printf 'stagei_speedup=%s\n' "$B300_STAGEI_PREPARED_STAGED_SPEEDUP"
  printf 'selected_backend=%s\n' "$BEST"
  printf 'selected_profile=%s\n' "$BEST_PROFILE"
  printf 'selected_binary=%s\n' "$BEST_BIN"
  printf 'selected_binary_sha256=%s\n' "$BEST_SHA"
  printf 'selected_runtime_kind=%s\n' "$RUNTIME"
  printf 'selected_work_dir=%s\n' "$BEST_WORK"
  printf 'race_result=%s\n' "$FINAL_RESULT"
  printf 'race_result_sha256=%s\n' "$FINAL_RESULT_SHA"
} >"$STAGEI_META"

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
  printf 'B300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$STAGEI_META"
  printf 'B300_GRAND_STAGEI_SELECTED_SCHEMA=1\n'
  printf 'B300_GRAND_STAGEI_SELECTED_VALIDATED=1\n'
  printf 'B300_GRAND_STAGEI_APPLICABLE=1\n'
  printf 'B300_GRAND_STAGEI_PROMOTED=%q\n' "$PROMOTED"
  printf 'B300_GRAND_STAGEI_SELF_WIDTH=%q\n' "$B300_STAGEI_PREPARED_SELF_WIDTH"
  printf 'B300_GRAND_STAGEI_SELF_DISTANCE=%q\n' "$B300_STAGEI_PREPARED_SELF_DISTANCE"
  printf 'B300_GRAND_STAGEI_MATE_WIDTH=%q\n' "$B300_STAGEI_PREPARED_MATE_WIDTH"
  printf 'B300_GRAND_STAGEI_MATE_DISTANCE=%q\n' "$B300_STAGEI_PREPARED_MATE_DISTANCE"
  printf 'B300_GRAND_STAGEI_SPEEDUP=%q\n' "$B300_STAGEI_PREPARED_STAGED_SPEEDUP"
  printf 'B300_GRAND_STAGEI_BASE_SELECTED_ENV=%q\n' "$BASE_SELECTED_ENV"
  printf 'B300_GRAND_STAGEI_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"
  printf 'B300_GRAND_STAGEI_PREPARE_ENV=%q\n' "$STAGEI_PREPARE_ENV"
  printf 'B300_GRAND_STAGEI_SELECTED_RACE_RESULT_SHA256=%q\n' "$FINAL_RESULT_SHA"
} >"$SELECTED_ENV"
cat "$SELECTED_ENV"
echo "b300x8-grand-stagei-firstpass OK backend=$BEST profile=$BEST_PROFILE promoted=$PROMOTED self=w${B300_STAGEI_PREPARED_SELF_WIDTH}d${B300_STAGEI_PREPARED_SELF_DISTANCE} mate=w${B300_STAGEI_PREPARED_MATE_WIDTH}d${B300_STAGEI_PREPARED_MATE_DISTANCE} selected_env=$SELECTED_ENV normalized_contract=1" >&2
