#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi; [[ "$N" == 27 ]] || exit 2
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -f "$PROFILE_FILE" ]] || exit 2
ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"; SMOKE_PRIME="${SMOKE_PRIME:-4294967291}"; FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
STAGEH_MIN_SPEEDUP="${STAGEH_MIN_SPEEDUP:-1.002}"; STAGEI_MIN_SPEEDUP="${STAGEI_MIN_SPEEDUP:-1.002}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_refine_firstpass_n27}"; BASE_PREFIX="${BASE_PREFIX:-${PREFIX}.base}"; BASE_SELECTED_ENV="${BASE_SELECTED_ENV:-${BASE_PREFIX}.selected.env}"; STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"
H_PREFIX="${H_PREFIX:-${PREFIX}.stageh}"; H_ENV="${H_ENV:-${H_PREFIX}_winner.env}"; H_PREP="${H_PREP:-${PREFIX}.stageh.prepared.env}"; H_RACE="${H_RACE:-${PREFIX}.stageh.promote}"
I_PREFIX="${I_PREFIX:-${PREFIX}.stagei}"; I_ENV="${I_ENV:-${I_PREFIX}_winner.env}"; I_PREP="${I_PREP:-${PREFIX}.stagei.prepared.env}"; I_RACE="${I_RACE:-${PREFIX}.stagei.promote}"
FINAL_PREFIX="${FINAL_PREFIX:-${PREFIX}.race}"; FINAL_RESULT="${FINAL_RESULT:-${FINAL_PREFIX}.tsv}"; FINAL_LOG="${FINAL_LOG:-${PREFIX}.log}"; SELECTED_ENV="${SELECTED_ENV:-${PREFIX}.selected.env}"; META="${META:-${PREFIX}.meta}"
mkdir -p "$(dirname "$SELECTED_ENV")" "$WORK_ROOT"

echo '=== refinement grand: base grand selection ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" SMOKE_PRIME="$SMOKE_PRIME" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" REBUILD_BUCKETS="$REBUILD_BUCKETS" WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh" 27
[[ -s "$BASE_SELECTED_ENV" ]] || exit 3; source "$BASE_SELECTED_ENV"; [[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]] || exit 3
BASE_BIN="$B300_GRAND_SELECTED_BINARY"; BASE_BACKEND="$B300_GRAND_SELECTED_BACKEND"; BASE_RUNTIME="$B300_GRAND_SELECTED_RUNTIME_KIND"; BASE_THREADS="$B300_GRAND_SELECTED_THREADS"; BASE_PROFILE_FILE="$B300_GRAND_SELECTED_PROFILE_FILE"; BASE_PROFILE_SHA="$B300_GRAND_SELECTED_PROFILE_SHA256"; BASE_PRIME="$B300_GRAND_SELECTED_SMOKE_PRIME"; BASE_MAXW="$B300_GRAND_SELECTED_MAX_WINDOW"; BASE_HEAD="$B300_GRAND_SELECTED_HEAD_SHA"; BASE_DIRTY="$B300_GRAND_SELECTED_HEAD_DIRTY"
if [[ ! -s "$STAGE_F_ENV" ]]; then cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"; printf 'B300_GRAND_REFINE_APPLICABLE=0\nB300_GRAND_REFINE_REASON=no_stage_f_env\n' >>"$SELECTED_ENV"; exit 0; fi

prepare_one(){
  local kind="$1" script="$2" min="$3" prefix="$4" envf="$5" race="$6" prep="$7"
  set +e
  PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$BASE_MAXW" MOD="$BASE_PRIME" INPUT_ENV="$STAGE_F_ENV" RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$min" STAGED_PREFIX="$prefix" WINNER_ENV="$envf" RACE_PREFIX="$race" PREPARE_ENV="$prep" bash "$script" 27
  local rc=$?
  set -e
  if ((rc==0)); then echo "$kind=1"; elif ((rc==4)); then echo "$kind=0"; else return "$rc"; fi
}
H_OK=0; I_OK=0
x="$(prepare_one H "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh" "$STAGEH_MIN_SPEEDUP" "$H_PREFIX" "$H_ENV" "$H_RACE" "$H_PREP")" || exit $?; [[ "$x" == H=1 ]] && H_OK=1
x="$(prepare_one I "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-evict-staged-fullprime-race.sh" "$STAGEI_MIN_SPEEDUP" "$I_PREFIX" "$I_ENV" "$I_RACE" "$I_PREP")" || exit $?; [[ "$x" == I=1 ]] && I_OK=1
if ((H_OK)); then source "$H_PREP"; [[ "${B300_STAGEH_PREPARED:-0}" == 1 ]] || exit 3; fi
if ((I_OK)); then source "$I_PREP"; [[ "${B300_STAGEI_PREPARED:-0}" == 1 ]] || exit 3; fi
if (( !H_OK && !I_OK )); then cp "$BASE_SELECTED_ENV" "$SELECTED_ENV"; printf 'B300_GRAND_REFINE_APPLICABLE=1\nB300_GRAND_REFINE_H_OK=0\nB300_GRAND_REFINE_I_OK=0\nB300_GRAND_REFINE_REASON=both_refinements_rejected\n' >>"$SELECTED_ENV"; exit 0; fi

P_BIN=""; P_LABEL=""; P_THREADS=256; B_BIN=""; B_LABEL=""; B_THREADS=256; E1_BIN=""; E1_LABEL=""; E1_THREADS=256; E2_BIN=""; E2_LABEL=""; E2_THREADS=256
if ((H_OK)); then
  P_BIN="$B300_STAGEH_PREPARED_BIN"; P_LABEL="$B300_STAGEH_PREPARED_LABEL"; P_THREADS="$B300_STAGEH_PREPARED_THREADS"; B_BIN="$B300_STAGEH_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEH_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEH_PREPARED_CONTROL_THREADS"
  if ((I_OK)); then E1_BIN="$B300_STAGEI_PREPARED_BIN"; E1_LABEL="$B300_STAGEI_PREPARED_LABEL"; E1_THREADS="$B300_STAGEI_PREPARED_THREADS"; fi
else
  P_BIN="$B300_STAGEI_PREPARED_BIN"; P_LABEL="$B300_STAGEI_PREPARED_LABEL"; P_THREADS="$B300_STAGEI_PREPARED_THREADS"; B_BIN="$B300_STAGEI_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEI_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEI_PREPARED_CONTROL_THREADS"
fi
if [[ "$BASE_RUNTIME" == forced && "$BASE_BIN" != "$P_BIN" && "$BASE_BIN" != "$B_BIN" && "$BASE_BIN" != "$E1_BIN" ]]; then E2_BIN="$BASE_BIN"; E2_LABEL="grand_previous_${BASE_BACKEND}"; E2_THREADS="$BASE_THREADS"; fi

echo "=== refinement full-prime race H=$H_OK I=$I_OK previous=$BASE_BACKEND ===" >&2
set +e
env PROFILE_FILE="$BASE_PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$BASE_PRIME" MAX_WINDOW="$BASE_MAXW" FORCED_TARGET_MIB="$FORCED_TARGET_MIB" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" WORK_ROOT="$WORK_ROOT" \
 FORCED_OVERRIDE_BIN="$P_BIN" FORCED_OVERRIDE_LABEL="$P_LABEL" FORCED_OVERRIDE_THREADS="$P_THREADS" FORCED_BASE_BIN="$B_BIN" FORCED_BASE_LABEL="$B_LABEL" FORCED_BASE_THREADS="$B_THREADS" FORCED_EXTRA_BIN="$E1_BIN" FORCED_EXTRA_LABEL="$E1_LABEL" FORCED_EXTRA_THREADS="$E1_THREADS" FORCED_EXTRA2_BIN="$E2_BIN" FORCED_EXTRA2_LABEL="$E2_LABEL" FORCED_EXTRA2_THREADS="$E2_THREADS" SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$FINAL_PREFIX" RESULT="$FINAL_RESULT" "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@" 2>&1 | tee "$FINAL_LOG"
rc=${PIPESTATUS[0]}; set -e; ((rc==0)) || exit "$rc"
WIN="$(python3 - "$FINAL_RESULT" <<'PY'
import csv,sys
ok=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['status']=='ok']
if not ok or len({r['residue'] for r in ok})!=1: raise SystemExit('refinement final exact gate failed')
b=min(ok,key=lambda r:float(r['wall_s'])); print('\t'.join((b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])))
PY
)"; IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"; [[ -x "$BEST_BIN" ]] || exit 4
BEST_SHA="$(sha256sum "$BEST_BIN" | awk '{print $1}')"; SHA12="${BEST_SHA:0:12}"; BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; CHECKPOINT="$BEST_WORK/checkpoint.json"; [[ -s "$CHECKPOINT" ]] || exit 4; RESULT_SHA="$(sha256sum "$FINAL_RESULT" | awk '{print $1}')"
RUNTIME=forced; THREADS=0; RUN_TARGET="$FORCED_TARGET_MIB"
if [[ "$BEST_BIN" == "$P_BIN" ]]; then THREADS="$P_THREADS"; elif [[ "$BEST_BIN" == "$B_BIN" ]]; then THREADS="$B_THREADS"; elif [[ -n "$E1_BIN" && "$BEST_BIN" == "$E1_BIN" ]]; then THREADS="$E1_THREADS"; elif [[ -n "$E2_BIN" && "$BEST_BIN" == "$E2_BIN" ]]; then THREADS="$E2_THREADS"; elif [[ "$BEST" == warp_tuned ]]; then RUNTIME=warp; RUN_TARGET="$BUCKET_TARGET_MIB"; else RUNTIME=orbit; RUN_TARGET="$BUCKET_TARGET_MIB"; fi
python3 - "$CHECKPOINT" "$BEST_SHA" "$BASE_PROFILE_SHA" "$BASE_PRIME" "$BEST_RES" <<'PY'
import json,sys
cp,bsha,psha,p,r=sys.argv[1:];d=json.load(open(cp));
if d.get('solver_fingerprint')!={'schema':3,'binary_sha256':bsha,'profile_sha256':psha}: raise SystemExit('refine checkpoint fingerprint mismatch')
z=d.get('residues',{}).get(str(int(p))); 
if not z or int(z.get('residue',-1))!=int(r): raise SystemExit('refine checkpoint residue mismatch')
PY
{
 printf 'schema=1\n'; printf 'h_ok=%s\n' "$H_OK"; printf 'i_ok=%s\n' "$I_OK"; printf 'base_selected_env=%s\n' "$BASE_SELECTED_ENV"; printf 'selected_backend=%s\n' "$BEST"; printf 'selected_profile=%s\n' "$BEST_PROFILE"; printf 'selected_binary_sha256=%s\n' "$BEST_SHA"; printf 'selected_wall_s=%s\n' "$BEST_WALL"; printf 'race_result_sha256=%s\n' "$RESULT_SHA";
} >"$META"
{
 printf 'B300_GRAND_SELECTED_SCHEMA=1\n'; printf 'B300_GRAND_SELECTED_VALIDATED=1\n'; printf 'B300_GRAND_SELECTED_N=27\n'; printf 'B300_GRAND_SELECTED_HEAD_SHA=%q\n' "$BASE_HEAD"; printf 'B300_GRAND_SELECTED_HEAD_DIRTY=%q\n' "$BASE_DIRTY"; printf 'B300_GRAND_SELECTED_PROFILE_FILE=%q\n' "$BASE_PROFILE_FILE"; printf 'B300_GRAND_SELECTED_PROFILE_SHA256=%q\n' "$BASE_PROFILE_SHA"; printf 'B300_GRAND_SELECTED_BACKEND=%q\n' "$BEST"; printf 'B300_GRAND_SELECTED_PROFILE=%q\n' "$BEST_PROFILE"; printf 'B300_GRAND_SELECTED_BINARY=%q\n' "$BEST_BIN"; printf 'B300_GRAND_SELECTED_BINARY_SHA256=%q\n' "$BEST_SHA"; printf 'B300_GRAND_SELECTED_RESIDUE=%q\n' "$BEST_RES"; printf 'B300_GRAND_SELECTED_WALL_S=%q\n' "$BEST_WALL"; printf 'B300_GRAND_SELECTED_SMOKE_PRIME=%q\n' "$BASE_PRIME"; printf 'B300_GRAND_SELECTED_RUNTIME_KIND=%q\n' "$RUNTIME"; printf 'B300_GRAND_SELECTED_THREADS=%q\n' "$THREADS"; printf 'B300_GRAND_SELECTED_TARGET_MIB=%q\n' "$RUN_TARGET"; printf 'B300_GRAND_SELECTED_MAX_WINDOW=%q\n' "$BASE_MAXW"; printf 'B300_GRAND_SELECTED_WORK_DIR=%q\n' "$BEST_WORK"; printf 'B300_GRAND_SELECTED_CHECKPOINT=%q\n' "$CHECKPOINT"; printf 'B300_GRAND_SELECTED_RACE_PREFIX=%q\n' "$FINAL_PREFIX"; printf 'B300_GRAND_SELECTED_RACE_RESULT=%q\n' "$FINAL_RESULT"; printf 'B300_GRAND_SELECTED_RACE_RESULT_SHA256=%q\n' "$RESULT_SHA"; printf 'B300_GRAND_SELECTED_FIRSTPASS_META=%q\n' "$META"; printf 'B300_GRAND_REFINE_APPLICABLE=1\n'; printf 'B300_GRAND_REFINE_H_OK=%q\n' "$H_OK"; printf 'B300_GRAND_REFINE_I_OK=%q\n' "$I_OK"; printf 'B300_GRAND_REFINE_H_PREP=%q\n' "$H_PREP"; printf 'B300_GRAND_REFINE_I_PREP=%q\n' "$I_PREP";
} >"$SELECTED_ENV"
cat "$SELECTED_ENV"; echo "b300x8-grand-refine-firstpass OK H=$H_OK I=$I_OK backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL selected_env=$SELECTED_ENV" >&2
