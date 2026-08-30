#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand Stage-K first-pass targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -s "$PROFILE_FILE" ]] || exit 2
ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"; SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
STAGEJ_MIN_SPEEDUP="${STAGEJ_MIN_SPEEDUP:-1.002}"; STAGEK_MIN_SPEEDUP="${STAGEK_MIN_SPEEDUP:-1.002}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"; MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"; MATE_EVICT_BASE="${MATE_EVICT_BASE:-default}"; MATE_EVICT_LIST="${MATE_EVICT_LIST:-default normal last}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_stagek_firstpass_n27}"
BASE_PREFIX="${BASE_PREFIX:-${PREFIX}.base}"; BASE_SELECTED_ENV="${BASE_SELECTED_ENV:-${BASE_PREFIX}.selected.env}"; STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"; BASE_STAGEI_PREPARE_ENV="${BASE_STAGEI_PREPARE_ENV:-${BASE_PREFIX}.stagei-evict.prepared.env}"
STAGEJ_PREFIX="${STAGEJ_PREFIX:-${PREFIX}.stagej}"; STAGEJ_WINNER_ENV="${STAGEJ_WINNER_ENV:-${STAGEJ_PREFIX}_winner.env}"; STAGEJ_PREPARE_ENV="${STAGEJ_PREPARE_ENV:-${PREFIX}.stagej.prepared.env}"; STAGEJ_RACE_PREFIX="${STAGEJ_RACE_PREFIX:-${PREFIX}.stagej.promote}"
STAGEK_PREFIX="${STAGEK_PREFIX:-${PREFIX}.stagek}"; STAGEK_WINNER_ENV="${STAGEK_WINNER_ENV:-${STAGEK_PREFIX}_winner.env}"; STAGEK_PREPARE_ENV="${STAGEK_PREPARE_ENV:-${PREFIX}.stagek.prepared.env}"; STAGEK_RACE_PREFIX="${STAGEK_RACE_PREFIX:-${PREFIX}.stagek.promote}"
FINAL_RACE_PREFIX="${FINAL_RACE_PREFIX:-${PREFIX}.race}"; FINAL_RESULT="${FINAL_RESULT:-${FINAL_RACE_PREFIX}.tsv}"; FINAL_LOG="${FINAL_LOG:-${PREFIX}.log}"; SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"; META="${META:-${PREFIX}.meta}"
mkdir -p "$(dirname "$SELECTED_ENV")" "$(dirname "$FINAL_LOG")" "$WORK_ROOT"
case "$MATE_EVICT_BASE" in default|normal|last) ;; *) exit 2;; esac
python3 - "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" <<'PY'
import sys
for n,v in zip(('STAGEJ_MIN_SPEEDUP','STAGEK_MIN_SPEEDUP'),map(float,sys.argv[1:])):
    if v < 1.0: raise SystemExit(f'{n} must be >=1')
PY

# 1) Existing grand selector is the incumbent. It already exact-races all older
# families and seeds a schema-3 smoke-prime checkpoint for its winner.
echo '=== Stage K grand: incumbent grand selection ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" REBUILD_BUCKETS="$REBUILD_BUCKETS" WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh" 27
[[ -s "$BASE_SELECTED_ENV" ]] || { echo 'base selected env missing' >&2; exit 3; }
# shellcheck disable=SC1090
source "$BASE_SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]] || exit 3
BASE_BIN="$B300_GRAND_SELECTED_BINARY"; BASE_BACKEND="$B300_GRAND_SELECTED_BACKEND"; BASE_PROFILE="$B300_GRAND_SELECTED_PROFILE"; BASE_RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; BASE_THREADS="$B300_GRAND_SELECTED_THREADS"
BASE_HEAD="$B300_GRAND_SELECTED_HEAD_SHA"; BASE_DIRTY="$B300_GRAND_SELECTED_HEAD_DIRTY"; BASE_PROFILE_FILE="$B300_GRAND_SELECTED_PROFILE_FILE"; BASE_PROFILE_SHA="$B300_GRAND_SELECTED_PROFILE_SHA256"; BASE_PRIME="$B300_GRAND_SELECTED_SMOKE_PRIME"; BASE_MAXW="$B300_GRAND_SELECTED_MAX_WINDOW"
[[ -x "$BASE_BIN" ]] || exit 3
if [[ ! -s "$STAGE_F_ENV" ]]; then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEK_APPLICABLE=0\nB300_GRAND_STAGEK_REASON=no_stage_f_env\n' >>"$SELECTED_ENV"
  exit 0
fi

# Preserve the official Stage-I self eviction hint when present. Support both
# the namespaced adapter and the historical raw prepare contract.
SELF_EVICT=default
if [[ -s "$BASE_STAGEI_PREPARE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$BASE_STAGEI_PREPARE_ENV"
  if [[ "${B300_EVICT_PREPARED:-0}" == 1 ]]; then
    SELF_EVICT="$B300_EVICT_HINT"
  elif [[ "${B300_STAGEI_PREPARED:-0}" == 1 ]]; then
    SELF_EVICT="$B300_STAGEI_PREPARED_HINT"
  fi
fi
case "$SELF_EVICT" in default|normal|last) ;; *) echo "bad inherited self eviction=$SELF_EVICT" >&2; exit 3;; esac

# 2) Stage J: independent mate geometry. PREPARE_ONLY means no complete-prime
# race yet; the winner is promoted only if its staged exact/spill gates pass.
echo "=== Stage K grand: Stage J staged mate geometry self_evict=$SELF_EVICT ===" >&2
set +e
PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$BASE_MAXW" MOD="$BASE_PRIME" INPUT_ENV="$STAGE_F_ENV" \
  RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEJ_MIN_SPEEDUP" MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT_BASE" \
  STAGED_PREFIX="$STAGEJ_PREFIX" WINNER_ENV="$STAGEJ_WINNER_ENV" RACE_PREFIX="$STAGEJ_RACE_PREFIX" PREPARE_ENV="$STAGEJ_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh" 27
J_RC=$?
set -e
if ((J_RC==4)); then
  cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"
  printf 'B300_GRAND_STAGEK_APPLICABLE=1\nB300_GRAND_STAGEJ_PROMOTED=0\nB300_GRAND_STAGEK_PROMOTED=0\nB300_GRAND_STAGEK_REASON=stagej_rejected\n' >>"$SELECTED_ENV"
  echo "Stage J rejected; retained incumbent $BASE_BACKEND" >&2
  exit 0
fi
((J_RC==0)) || exit "$J_RC"
[[ -s "$STAGEJ_PREPARE_ENV" && -s "$STAGEJ_WINNER_ENV" ]] || exit 3
# shellcheck disable=SC1090
source "$STAGEJ_PREPARE_ENV"
for k in B300_STAGEJ_PREPARED B300_STAGEJ_PREPARED_BIN B300_STAGEJ_PREPARED_LABEL B300_STAGEJ_PREPARED_THREADS B300_STAGEJ_PREPARED_CONTROL_BIN B300_STAGEJ_PREPARED_CONTROL_LABEL B300_STAGEJ_PREPARED_CONTROL_THREADS B300_STAGEJ_PREPARED_SELF_WIDTH B300_STAGEJ_PREPARED_SELF_DISTANCE B300_STAGEJ_PREPARED_SELF_EVICT B300_STAGEJ_PREPARED_MATE_WIDTH B300_STAGEJ_PREPARED_MATE_DISTANCE B300_STAGEJ_PREPARED_MATE_EVICT B300_STAGEJ_PREPARED_STAGED_SPEEDUP; do [[ -n "${!k+x}" ]] || { echo "Stage-J prepare missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEJ_PREPARED" == 1 && -x "$B300_STAGEJ_PREPARED_BIN" && -x "$B300_STAGEJ_PREPARED_CONTROL_BIN" ]] || exit 3
J_BIN="$B300_STAGEJ_PREPARED_BIN"; J_LABEL="$B300_STAGEJ_PREPARED_LABEL"; J_THREADS="$B300_STAGEJ_PREPARED_THREADS"; J_CONTROL_BIN="$B300_STAGEJ_PREPARED_CONTROL_BIN"; J_CONTROL_LABEL="$B300_STAGEJ_PREPARED_CONTROL_LABEL"; J_CONTROL_THREADS="$B300_STAGEJ_PREPARED_CONTROL_THREADS"
J_SW="$B300_STAGEJ_PREPARED_SELF_WIDTH"; J_SD="$B300_STAGEJ_PREPARED_SELF_DISTANCE"; J_SE="$B300_STAGEJ_PREPARED_SELF_EVICT"; J_MW="$B300_STAGEJ_PREPARED_MATE_WIDTH"; J_MD="$B300_STAGEJ_PREPARED_MATE_DISTANCE"; J_ME="$B300_STAGEJ_PREPARED_MATE_EVICT"; J_SPEED="$B300_STAGEJ_PREPARED_STAGED_SPEEDUP"

# 3) Stage K: only the mate eviction priority moves. Failure here does not drop
# Stage J; it simply means Stage J goes to the final exact race unchanged.
K_OK=0
echo "=== Stage K grand: staged mate eviction geometry=self:w${J_SW}d${J_SD},mate:w${J_MW}d${J_MD} ===" >&2
set +e
PROFILE_FILE="$BASE_PROFILE_FILE" STAGE_F_ENV="$STAGE_F_ENV" STAGEJ_WINNER_ENV="$STAGEJ_WINNER_ENV" STAGEJ_PREPARE_ENV="$STAGEJ_PREPARE_ENV" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$BASE_MAXW" MOD="$BASE_PRIME" \
  RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEK_MIN_SPEEDUP" EVICT_LIST="$MATE_EVICT_LIST" STAGED_PREFIX="$STAGEK_PREFIX" WINNER_ENV="$STAGEK_WINNER_ENV" RACE_PREFIX="$STAGEK_RACE_PREFIX" PREPARE_ENV="$STAGEK_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh" 27
K_RC=$?
set -e
if ((K_RC==0)); then K_OK=1
elif ((K_RC==4)); then echo 'Stage K mate-eviction refinement rejected; retaining Stage J candidate' >&2
else exit "$K_RC"
fi
if ((K_OK)); then
  [[ -s "$STAGEK_PREPARE_ENV" ]] || exit 3
  # shellcheck disable=SC1090
  source "$STAGEK_PREPARE_ENV"
  [[ "${B300_STAGEK_PREPARED:-0}" == 1 && -x "$B300_STAGEK_PREPARED_BIN" && -x "$B300_STAGEK_PREPARED_CONTROL_BIN" ]] || exit 3
fi

# 4) One complete-prime race only. If K survived: K vs J vs self-control vs
# incumbent forced winner. If K did not: J vs self-control vs incumbent.
P_BIN="$J_BIN"; P_LABEL="$J_LABEL"; P_THREADS="$J_THREADS"
B_BIN="$J_CONTROL_BIN"; B_LABEL="$J_CONTROL_LABEL"; B_THREADS="$J_CONTROL_THREADS"
E1_BIN=""; E1_LABEL=""; E1_THREADS=256
if ((K_OK)); then
  P_BIN="$B300_STAGEK_PREPARED_BIN"; P_LABEL="$B300_STAGEK_PREPARED_LABEL"; P_THREADS="$B300_STAGEK_PREPARED_THREADS"
  B_BIN="$B300_STAGEK_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEK_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEK_PREPARED_CONTROL_THREADS"
  if [[ "$J_CONTROL_BIN" != "$P_BIN" && "$J_CONTROL_BIN" != "$B_BIN" ]]; then E1_BIN="$J_CONTROL_BIN"; E1_LABEL="$J_CONTROL_LABEL"; E1_THREADS="$J_CONTROL_THREADS"; fi
fi
E2_BIN=""; E2_LABEL=""; E2_THREADS=256
if [[ "$BASE_RUNTIME" == forced && "$BASE_BIN" != "$P_BIN" && "$BASE_BIN" != "$B_BIN" && "$BASE_BIN" != "$E1_BIN" ]]; then E2_BIN="$BASE_BIN"; E2_LABEL="grand_previous_${BASE_BACKEND}_${BASE_PROFILE}"; E2_THREADS="$BASE_THREADS"; fi

echo "=== Stage K final complete-prime race J=1 K=$K_OK incumbent=$BASE_BACKEND ===" >&2
set +e
env PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$BASE_PRIME" MAX_WINDOW="$BASE_MAXW" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" \
  FORCED_OVERRIDE_BIN="$P_BIN" FORCED_OVERRIDE_LABEL="$P_LABEL" FORCED_OVERRIDE_THREADS="$P_THREADS" FORCED_BASE_BIN="$B_BIN" FORCED_BASE_LABEL="$B_LABEL" FORCED_BASE_THREADS="$B_THREADS" \
  FORCED_EXTRA_BIN="$E1_BIN" FORCED_EXTRA_LABEL="$E1_LABEL" FORCED_EXTRA_THREADS="$E1_THREADS" FORCED_EXTRA2_BIN="$E2_BIN" FORCED_EXTRA2_LABEL="$E2_LABEL" FORCED_EXTRA2_THREADS="$E2_THREADS" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$FINAL_RACE_PREFIX" RESULT="$FINAL_RESULT" "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@" 2>&1 | tee "$FINAL_LOG"
RC=${PIPESTATUS[0]}; set -e; ((RC==0)) || exit "$RC"; [[ -s "$FINAL_RESULT" ]] || exit 4
WIN="$(python3 - "$FINAL_RESULT" <<'PY'
import csv,sys
ok=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['status']=='ok']
if not ok or len({r['residue'] for r in ok}) != 1: raise SystemExit('Stage-K final exact gate failed')
b=min(ok,key=lambda r:float(r['wall_s'])); print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"; [[ -x "$BEST_BIN" ]] || exit 4
BEST_SHA="$(sha256sum "$BEST_BIN" | awk '{print $1}')"; SHA12="${BEST_SHA:0:12}"; BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; CHECKPOINT="$BEST_WORK/checkpoint.json"; [[ -s "$CHECKPOINT" ]] || exit 4; RESULT_SHA="$(sha256sum "$FINAL_RESULT" | awk '{print $1}')"
RUNTIME=forced; THREADS=0; RUN_TARGET="$FORCED_TARGET_MIB"
if [[ "$BEST_BIN" == "$P_BIN" ]]; then THREADS="$P_THREADS"
elif [[ "$BEST_BIN" == "$B_BIN" ]]; then THREADS="$B_THREADS"
elif [[ -n "$E1_BIN" && "$BEST_BIN" == "$E1_BIN" ]]; then THREADS="$E1_THREADS"
elif [[ -n "$E2_BIN" && "$BEST_BIN" == "$E2_BIN" ]]; then THREADS="$E2_THREADS"
elif [[ "$BEST" == warp_tuned ]]; then RUNTIME=warp; RUN_TARGET="$BUCKET_TARGET_MIB"
elif [[ "$BEST" == orbit_tuned ]]; then RUNTIME=orbit; RUN_TARGET="$BUCKET_TARGET_MIB"
else echo "cannot map Stage-K winner backend=$BEST profile=$BEST_PROFILE" >&2; exit 4; fi
python3 - "$CHECKPOINT" "$BEST_SHA" "$BASE_PROFILE_SHA" "$BASE_PRIME" "$BEST_RES" <<'PY'
import json,sys
cp,bsha,psha,p,r=sys.argv[1:]; d=json.load(open(cp))
if int(d.get('n',-1)) != 27 or d.get('solver_fingerprint') != {'schema':3,'binary_sha256':bsha,'profile_sha256':psha}: raise SystemExit('Stage-K checkpoint fingerprint mismatch')
z=d.get('residues',{}).get(str(int(p)))
if not z or int(z.get('residue',-1)) != int(r): raise SystemExit('Stage-K checkpoint smoke residue mismatch')
PY
K_PROMOTED=0; ((K_OK)) && [[ "$BEST_BIN" == "$B300_STAGEK_PREPARED_BIN" ]] && K_PROMOTED=1
J_PROMOTED=0; if [[ "$BEST_BIN" == "$J_BIN" ]] || ((K_PROMOTED)); then J_PROMOTED=1; fi
FINAL_MATE_EVICT="$J_ME"; if ((K_PROMOTED)); then FINAL_MATE_EVICT="$B300_STAGEK_PREPARED_MATE_EVICT"; fi
{
  printf 'schema=1\n'; printf 'stagej_staged=1\n'; printf 'stagek_staged=%s\n' "$K_OK"; printf 'stagej_speedup=%s\n' "$J_SPEED"; printf 'stagek_speedup=%s\n' "${B300_STAGEK_PREPARED_STAGED_SPEEDUP:-1.000000000}"; printf 'self_geometry=w%sd%s\n' "$J_SW" "$J_SD"; printf 'self_evict=%s\n' "$J_SE"; printf 'mate_geometry=w%sd%s\n' "$J_MW" "$J_MD"; printf 'mate_evict=%s\n' "$FINAL_MATE_EVICT"; printf 'selected_backend=%s\n' "$BEST"; printf 'selected_profile=%s\n' "$BEST_PROFILE"; printf 'selected_binary_sha256=%s\n' "$BEST_SHA"; printf 'race_result_sha256=%s\n' "$RESULT_SHA";
} >"$META"
{
  printf 'B300_GRAND_SELECTED_SCHEMA=1\n'; printf 'B300_GRAND_SELECTED_VALIDATED=1\n'; printf 'B300_GRAND_SELECTED_N=27\n'; printf 'B300_GRAND_SELECTED_HEAD_SHA=%q\n' "$BASE_HEAD"; printf 'B300_GRAND_SELECTED_HEAD_DIRTY=%q\n' "$BASE_DIRTY"; printf 'B300_GRAND_SELECTED_PROFILE_FILE=%q\n' "$BASE_PROFILE_FILE"; printf 'B300_GRAND_SELECTED_PROFILE_SHA256=%q\n' "$BASE_PROFILE_SHA"; printf 'B300_GRAND_SELECTED_BACKEND=%q\n' "$BEST"; printf 'B300_GRAND_SELECTED_PROFILE=%q\n' "$BEST_PROFILE"; printf 'B300_GRAND_SELECTED_BINARY=%q\n' "$BEST_BIN"; printf 'B300_GRAND_SELECTED_BINARY_SHA256=%q\n' "$BEST_SHA"; printf 'B300_GRAND_SELECTED_RESIDUE=%q\n' "$BEST_RES"; printf 'B300_GRAND_SELECTED_WALL_S=%q\n' "$BEST_WALL"; printf 'B300_GRAND_SELECTED_SMOKE_PRIME=%q\n' "$BASE_PRIME"; printf 'B300_GRAND_SELECTED_RUNTIME_KIND=%q\n' "$RUNTIME"; printf 'B300_GRAND_SELECTED_THREADS=%q\n' "$THREADS"; printf 'B300_GRAND_SELECTED_TARGET_MIB=%q\n' "$RUN_TARGET"; printf 'B300_GRAND_SELECTED_MAX_WINDOW=%q\n' "$BASE_MAXW"; printf 'B300_GRAND_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"; printf 'B300_GRAND_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"; printf 'B300_GRAND_SELECTED_RACE_PREFIX=%q\n' "$FINAL_RACE_PREFIX"; printf 'B300_GRAND_SELECTED_RACE_RESULT=%q\n' "$FINAL_RESULT"; printf 'B300_GRAND_SELECTED_RACE_RESULT_SHA256=%q\n' "$RESULT_SHA"; printf 'B300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$META";
  printf 'B300_GRAND_STAGEJ_STAGED=1\n'; printf 'B300_GRAND_STAGEJ_PROMOTED=%q\n' "$J_PROMOTED"; printf 'B300_GRAND_STAGEJ_SELF_WIDTH=%q\n' "$J_SW"; printf 'B300_GRAND_STAGEJ_SELF_DISTANCE=%q\n' "$J_SD"; printf 'B300_GRAND_STAGEJ_SELF_EVICT=%q\n' "$J_SE"; printf 'B300_GRAND_STAGEJ_MATE_WIDTH=%q\n' "$J_MW"; printf 'B300_GRAND_STAGEJ_MATE_DISTANCE=%q\n' "$J_MD"; printf 'B300_GRAND_STAGEJ_MATE_EVICT=%q\n' "$J_ME"; printf 'B300_GRAND_STAGEJ_PREPARE_ENV=%q\n' "$STAGEJ_PREPARE_ENV"; printf 'B300_GRAND_STAGEJ_WINNER_ENV=%q\n' "$STAGEJ_WINNER_ENV";
  printf 'B300_GRAND_STAGEK_APPLICABLE=1\n'; printf 'B300_GRAND_STAGEK_STAGED=%q\n' "$K_OK"; printf 'B300_GRAND_STAGEK_PROMOTED=%q\n' "$K_PROMOTED"; printf 'B300_GRAND_STAGEK_MATE_EVICT=%q\n' "$FINAL_MATE_EVICT"; printf 'B300_GRAND_STAGEK_PREPARE_ENV=%q\n' "${STAGEK_PREPARE_ENV:-}"; printf 'B300_GRAND_STAGEK_WINNER_ENV=%q\n' "${STAGEK_WINNER_ENV:-}";
} >"$SELECTED_ENV"
cat "$SELECTED_ENV"
echo "b300x8-grand-stagek-firstpass OK J=$J_PROMOTED K=$K_PROMOTED backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL self=w${J_SW}d${J_SD}/$J_SE mate=w${J_MW}d${J_MD}/$FINAL_MATE_EVICT selected_env=$SELECTED_ENV" >&2
