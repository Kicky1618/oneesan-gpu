#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand Stage-J v2 first-pass targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"; SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"; STAGEJ_MIN_SPEEDUP="${STAGEJ_MIN_SPEEDUP:-1.002}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"; MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"; MATE_EVICT="${MATE_EVICT:-default}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_stagej_v2_firstpass_n27}"
BASE_PREFIX="${BASE_PREFIX:-${PREFIX}.base}"; BASE_SELECTED_ENV="${BASE_SELECTED_ENV:-${BASE_PREFIX}.selected.env}"
STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"; BASE_STAGEI_PREPARE_ENV="${BASE_STAGEI_PREPARE_ENV:-${BASE_PREFIX}.stagei-evict.prepared.env}"
STAGEJ_PREFIX="${STAGEJ_PREFIX:-${PREFIX}.stagej}"; STAGEJ_ENV="${STAGEJ_ENV:-${STAGEJ_PREFIX}_winner.env}"
STAGEJ_PREPARE_ENV="${STAGEJ_PREPARE_ENV:-${PREFIX}.stagej.prepared.env}"; STAGEJ_RACE_PREFIX="${STAGEJ_RACE_PREFIX:-${PREFIX}.stagej.promote}"
FINAL_RACE_PREFIX="${FINAL_RACE_PREFIX:-${PREFIX}.race}"; FINAL_RESULT="${FINAL_RESULT:-${FINAL_RACE_PREFIX}.tsv}"
FINAL_LOG="${FINAL_LOG:-${PREFIX}.log}"; SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"; META="${META:-${PREFIX}.meta}"
STAGEJ_RUNNER="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh"

[[ -s "$PROFILE_FILE" ]] || { echo "missing profile=$PROFILE_FILE" >&2; exit 2; }
case "$MATE_EVICT" in default|normal|last) ;; *) exit 2;; esac
for x in MAX_WINDOW SMOKE_PRIME FORCED_TARGET_MIB BUCKET_TARGET_MIB; do v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$STAGEJ_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('STAGEJ_MIN_SPEEDUP must be >=1')
PY
mkdir -p "$(dirname "$SELECTED_ENV")" "$(dirname "$FINAL_LOG")" "$WORK_ROOT"

# Baseline grand pipeline owns Stage I=self eviction and Stage H=shared-geometry mate prefetch.
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" \
  FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh" 27
[[ -s "$BASE_SELECTED_ENV" ]] || exit 3
# shellcheck disable=SC1090
source "$BASE_SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_SCHEMA:-}" == 1 && "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]] || exit 3
BASE_BIN="$B300_GRAND_SELECTED_BINARY"; BASE_BACKEND="$B300_GRAND_SELECTED_BACKEND"; BASE_PROFILE="$B300_GRAND_SELECTED_PROFILE"
BASE_RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; BASE_THREADS="$B300_GRAND_SELECTED_THREADS"
BASE_HEAD_SHA="$B300_GRAND_SELECTED_HEAD_SHA"; BASE_HEAD_DIRTY="$B300_GRAND_SELECTED_HEAD_DIRTY"
BASE_PROFILE_FILE="$B300_GRAND_SELECTED_PROFILE_FILE"; BASE_PROFILE_SHA="$B300_GRAND_SELECTED_PROFILE_SHA256"
BASE_SMOKE_PRIME="$B300_GRAND_SELECTED_SMOKE_PRIME"; BASE_MAX_WINDOW="$B300_GRAND_SELECTED_MAX_WINDOW"
[[ -x "$BASE_BIN" ]] || exit 3

if [[ ! -s "$STAGE_F_ENV" ]]; then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEJ_APPLICABLE=0\nB300_GRAND_STAGEJ_REASON=no_stage_f_env\n' >>"$SELECTED_ENV"
  echo "Stage J v2 skipped: no Stage-F env; retained $BASE_BACKEND" >&2
  exit 0
fi

# Only these two B300_STAGEI_* variables are read: they belong to the official self-eviction stage.
SELF_EVICT=default
if [[ -s "$BASE_STAGEI_PREPARE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$BASE_STAGEI_PREPARE_ENV"
  if [[ "${B300_STAGEI_PREPARED:-0}" == 1 ]]; then
    case "${B300_STAGEI_PREPARED_HINT:-}" in normal|last) SELF_EVICT="$B300_STAGEI_PREPARED_HINT";; *) exit 3;; esac
  fi
fi

set +e
PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" MOD="$BASE_SMOKE_PRIME" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$BASE_MAX_WINDOW" \
  INPUT_ENV="$STAGE_F_ENV" RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEJ_MIN_SPEEDUP" \
  MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" \
  STAGED_PREFIX="$STAGEJ_PREFIX" WINNER_ENV="$STAGEJ_ENV" RACE_PREFIX="$STAGEJ_RACE_PREFIX" PREPARE_ENV="$STAGEJ_PREPARE_ENV" \
  bash "$STAGEJ_RUNNER" 27
rc=$?
set -e
if ((rc==4)); then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEJ_APPLICABLE=1\nB300_GRAND_STAGEJ_PROMOTED=0\nB300_GRAND_STAGEJ_REASON=staged_geometry_rejected\n' >>"$SELECTED_ENV"
  echo "Stage J v2 rejected; retained $BASE_BACKEND" >&2
  exit 0
fi
((rc==0)) || exit "$rc"
# shellcheck disable=SC1090
source "$STAGEJ_PREPARE_ENV"
[[ "${B300_STAGEJ_PREPARED:-0}" == 1 && "$B300_STAGEJ_PREPARED_MOD" == "$BASE_SMOKE_PRIME" ]] || exit 3
[[ -x "$B300_STAGEJ_PREPARED_BIN" && -x "$B300_STAGEJ_PREPARED_CONTROL_BIN" ]] || exit 3
[[ -s "$B300_STAGEJ_PREPARED_MANIFEST" ]] || exit 3
sha256sum -c "$B300_STAGEJ_PREPARED_MANIFEST" >/dev/null || exit 3
J_BIN="$B300_STAGEJ_PREPARED_BIN"; J_LABEL="$B300_STAGEJ_PREPARED_LABEL"; J_THREADS="$B300_STAGEJ_PREPARED_THREADS"
J_CTL="$B300_STAGEJ_PREPARED_CONTROL_BIN"; J_CTL_LABEL="$B300_STAGEJ_PREPARED_CONTROL_LABEL"; J_CTL_THREADS="$B300_STAGEJ_PREPARED_CONTROL_THREADS"
J_SW="$B300_STAGEJ_PREPARED_SELF_WIDTH"; J_SD="$B300_STAGEJ_PREPARED_SELF_DISTANCE"; J_MW="$B300_STAGEJ_PREPARED_MATE_WIDTH"; J_MD="$B300_STAGEJ_PREPARED_MATE_DISTANCE"
J_SPEED="$B300_STAGEJ_PREPARED_STAGED_SPEEDUP"; J_MANIFEST="$B300_STAGEJ_PREPARED_MANIFEST"; J_BUILD_DIR="$B300_STAGEJ_PREPARED_BUILD_DIR"

EXTRA_BIN=""; EXTRA_LABEL=""; EXTRA_THREADS=256
if [[ "$BASE_RUNTIME" == forced && "$BASE_BIN" != "$J_BIN" && "$BASE_BIN" != "$J_CTL" ]]; then
  EXTRA_BIN="$BASE_BIN"; EXTRA_LABEL="grand_previous_${BASE_BACKEND}_${BASE_PROFILE}"; EXTRA_THREADS="$BASE_THREADS"
fi

set +e
env PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$BASE_SMOKE_PRIME" MAX_WINDOW="$BASE_MAX_WINDOW" \
  FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" \
  FORCED_OVERRIDE_BIN="$J_BIN" FORCED_OVERRIDE_LABEL="$J_LABEL" FORCED_OVERRIDE_THREADS="$J_THREADS" \
  FORCED_BASE_BIN="$J_CTL" FORCED_BASE_LABEL="$J_CTL_LABEL" FORCED_BASE_THREADS="$J_CTL_THREADS" \
  FORCED_EXTRA_BIN="$EXTRA_BIN" FORCED_EXTRA_LABEL="$EXTRA_LABEL" FORCED_EXTRA_THREADS="$EXTRA_THREADS" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$FINAL_RACE_PREFIX" RESULT="$FINAL_RESULT" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@" 2>&1 | tee "$FINAL_LOG"
rc=${PIPESTATUS[0]}; set -e; ((rc==0)) || exit "$rc"; [[ -s "$FINAL_RESULT" ]] || exit 4

WIN="$(python3 - "$FINAL_RESULT" <<'PY'
import csv,sys
ok=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['status']=='ok']
if not ok: raise SystemExit('no Stage-J candidates')
if len({r['residue'] for r in ok})!=1: raise SystemExit('Stage-J residue mismatch')
b=min(ok,key=lambda r:float(r['wall_s']))
print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"; [[ -x "$BEST_BIN" ]] || exit 4
BEST_SHA="$(sha256sum "$BEST_BIN"|awk '{print $1}')"; SHA12="${BEST_SHA:0:12}"; BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"
CHECKPOINT="$BEST_WORK/checkpoint.json"; [[ -s "$CHECKPOINT" ]] || exit 4; RESULT_SHA="$(sha256sum "$FINAL_RESULT"|awk '{print $1}')"
RUNTIME=forced; THREADS=0; RUN_TARGET="$FORCED_TARGET_MIB"
if [[ "$BEST_BIN" == "$J_BIN" ]]; then THREADS="$J_THREADS"
elif [[ "$BEST_BIN" == "$J_CTL" ]]; then THREADS="$J_CTL_THREADS"
elif [[ -n "$EXTRA_BIN" && "$BEST_BIN" == "$EXTRA_BIN" ]]; then THREADS="$EXTRA_THREADS"
elif [[ "$BEST" == warp_tuned ]]; then RUNTIME=warp; RUN_TARGET="$BUCKET_TARGET_MIB"
elif [[ "$BEST" == orbit_tuned ]]; then RUNTIME=orbit; RUN_TARGET="$BUCKET_TARGET_MIB"
else exit 4; fi
PROMOTED=0; [[ "$BEST_BIN" == "$J_BIN" ]] && PROMOTED=1

python3 - "$CHECKPOINT" "$BEST_SHA" "$BASE_PROFILE_SHA" "$BASE_SMOKE_PRIME" "$BEST_RES" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); b,p,prime,res=sys.argv[2:]
assert int(d['n'])==27
assert d['solver_fingerprint']=={'schema':3,'binary_sha256':b,'profile_sha256':p}
assert int(d['residues'][str(int(prime))]['residue'])==int(res)
PY

{
  printf 'schema=2\nstagej_validated=1\n'
  printf 'base_selected_env=%s\nstage_f_env=%s\nofficial_stagei_prepare_env=%s\n' "$BASE_SELECTED_ENV" "$STAGE_F_ENV" "$BASE_STAGEI_PREPARE_ENV"
  printf 'stagej_prepare_env=%s\nstagej_manifest=%s\nstagej_build_dir=%s\n' "$STAGEJ_PREPARE_ENV" "$J_MANIFEST" "$J_BUILD_DIR"
  printf 'stagej_self_evict=%s\nstagej_self_width=%s\nstagej_self_distance=%s\nstagej_mate_width=%s\nstagej_mate_distance=%s\nstagej_mate_evict=%s\nstagej_speedup=%s\n' "$SELF_EVICT" "$J_SW" "$J_SD" "$J_MW" "$J_MD" "$MATE_EVICT" "$J_SPEED"
  printf 'selected_backend=%s\nselected_profile=%s\nselected_binary=%s\nselected_binary_sha256=%s\nselected_runtime_kind=%s\nselected_work_dir=%s\nrace_result=%s\nrace_result_sha256=%s\n' "$BEST" "$BEST_PROFILE" "$BEST_BIN" "$BEST_SHA" "$RUNTIME" "$BEST_WORK" "$FINAL_RESULT" "$RESULT_SHA"
} >"$META"

{
  printf 'B300_GRAND_SELECTED_SCHEMA=1\nB300_GRAND_SELECTED_VALIDATED=1\nB300_GRAND_SELECTED_N=27\n'
  printf 'B300_GRAND_SELECTED_HEAD_SHA=%q\nB300_GRAND_SELECTED_HEAD_DIRTY=%q\n' "$BASE_HEAD_SHA" "$BASE_HEAD_DIRTY"
  printf 'B300_GRAND_SELECTED_PROFILE_FILE=%q\nB300_GRAND_SELECTED_PROFILE_SHA256=%q\n' "$BASE_PROFILE_FILE" "$BASE_PROFILE_SHA"
  printf 'B300_GRAND_SELECTED_BACKEND=%q\nB300_GRAND_SELECTED_PROFILE=%q\nB300_GRAND_SELECTED_BINARY=%q\nB300_GRAND_SELECTED_BINARY_SHA256=%q\n' "$BEST" "$BEST_PROFILE" "$BEST_BIN" "$BEST_SHA"
  printf 'B300_GRAND_SELECTED_RESIDUE=%q\nB300_GRAND_SELECTED_WALL_S=%q\nB300_GRAND_SELECTED_SMOKE_PRIME=%q\n' "$BEST_RES" "$BEST_WALL" "$BASE_SMOKE_PRIME"
  printf 'B300_GRAND_SELECTED_RUNTIME_KIND=%q\nB300_GRAND_SELECTED_THREADS=%q\nB300_GRAND_SELECTED_TARGET_MIB=%q\nB300_GRAND_SELECTED_MAX_WINDOW=%q\n' "$RUNTIME" "$THREADS" "$RUN_TARGET" "$BASE_MAX_WINDOW"
  printf 'B300_GRAND_SELECTED_WORK_DIR=%q\nB300_GRAND_SELECTED_CHECKPOINT=%q\nB300_GRAND_SELECTED_RACE_PREFIX=%q\nB300_GRAND_SELECTED_RACE_RESULT=%q\nB300_GRAND_SELECTED_RACE_RESULT_SHA256=%q\nB300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$BEST_WORK" "$CHECKPOINT" "$FINAL_RACE_PREFIX" "$FINAL_RESULT" "$RESULT_SHA" "$META"
  printf 'B300_GRAND_STAGEJ_SELECTED_SCHEMA=2\nB300_GRAND_STAGEJ_SELECTED_VALIDATED=1\nB300_GRAND_STAGEJ_APPLICABLE=1\nB300_GRAND_STAGEJ_PROMOTED=%q\n' "$PROMOTED"
  printf 'B300_GRAND_STAGEJ_SELF_EVICT=%q\nB300_GRAND_STAGEJ_SELF_WIDTH=%q\nB300_GRAND_STAGEJ_SELF_DISTANCE=%q\nB300_GRAND_STAGEJ_MATE_WIDTH=%q\nB300_GRAND_STAGEJ_MATE_DISTANCE=%q\nB300_GRAND_STAGEJ_MATE_EVICT=%q\nB300_GRAND_STAGEJ_SPEEDUP=%q\n' "$SELF_EVICT" "$J_SW" "$J_SD" "$J_MW" "$J_MD" "$MATE_EVICT" "$J_SPEED"
  printf 'B300_GRAND_STAGEJ_PREPARE_ENV=%q\nB300_GRAND_STAGEJ_MANIFEST=%q\nB300_GRAND_STAGEJ_BUILD_DIR=%q\n' "$STAGEJ_PREPARE_ENV" "$J_MANIFEST" "$J_BUILD_DIR"
} >"$SELECTED_ENV"
cat "$SELECTED_ENV"
echo "b300x8-grand-stagej-v2-firstpass OK backend=$BEST promoted=$PROMOTED self=w${J_SW}d${J_SD}/$SELF_EVICT mate=w${J_MW}d${J_MD}/$MATE_EVICT selected_env=$SELECTED_ENV namespace_isolated=1" >&2
