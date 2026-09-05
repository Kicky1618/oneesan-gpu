#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'single-pass joint calibrated selector targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; PRIME="${SMOKE_PRIME:-4294967291}"; TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_joint_calibrated_once_n27}"; CAL_PREFIX="${CAL_PREFIX:-${PREFIX}.calibration}"; CAL_LOG="${CAL_LOG:-${PREFIX}.calibration.log}"
RECALIBRATE="${RECALIBRATE:-1}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
for x in RECALIBRATE SELECT_ONLY REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile $PROFILE_FILE" >&2; exit 2; }
mkdir -p "$(dirname "$PREFIX")"
getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }

if [[ "$RECALIBRATE" == 1 || ! -s "$CAL_LOG" ]]; then
  echo '=== single-pass joint selector: forced joint calibration ===' >&2
  ARCH="$ARCH" MOD="$PRIME" ROWS="${CAL_ROWS:-1}" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="${THREADS_LIST:-128 256 512 1024}" BATCH_LIST="${BATCH_LIST:-2 4}" HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}" REPEATS="${REPEATS:-1}" TRANSFORM_MIN_SPEEDUP="${TRANSFORM_MIN_SPEEDUP:-1.01}" PREFIX="$CAL_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-forced-joint-calibrate.sh" | tee "$CAL_LOG"
fi
[[ "$(getv b300_forced_joint_exact_gates "$CAL_LOG")" == 1 ]] || { echo 'joint calibration exact gate missing' >&2; exit 3; }
HIGH="$(getv b300_forced_joint_final_high_drop_chunk "$CAL_LOG")"; DUAL="$(getv b300_forced_joint_final_dualmask "$CAL_LOG")"; BATCH="$(getv b300_forced_joint_final_closure_batch "$CAL_LOG")"; THREADS="$(getv b300_forced_joint_final_threads "$CAL_LOG")"; MODE="$(getv b300_forced_joint_final_mode "$CAL_LOG")"
BASE_HIGH="$(getv b300_forced_joint_global_base_high_drop "$CAL_LOG")"; BASE_THREADS="$(getv b300_forced_joint_global_base_threads "$CAL_LOG")"
[[ "$HIGH" == 0 || "$HIGH" == 1 ]] || exit 3; [[ "$DUAL" == 0 || "$DUAL" == 1 ]] || exit 3; case "$BATCH" in 0|2|4);;*)exit 3;;esac
[[ "$THREADS" =~ ^[0-9]+$ && "$BASE_THREADS" =~ ^[0-9]+$ ]] || exit 3; [[ "$BASE_HIGH" == 0 || "$BASE_HIGH" == 1 ]] || exit 3

LABEL="forced_joint_${MODE}_hd${HIGH}_dual${DUAL}_cb${BATCH}_t${THREADS}"; BIN="$ONEESAN_BUILD_DIR/b300_${LABEL}_n27"; BUILD_OUT="${PREFIX}.build.out"; BUILD_ERR="${PREFIX}.build.err"
echo "=== single-pass joint selector: build $LABEL ===" >&2
N=27 ARCH="$ARCH" OUT="$BIN" HIGH_DROP_CHUNK="$HIGH" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" BUILD_ERR="$BUILD_ERR" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-mlp-calibrated.sh" >"$BUILD_OUT" 2>"${PREFIX}.build.driver.err"
[[ -x "$BIN" ]] || { echo 'joint primary binary missing' >&2; exit 4; }
grep -Fq "mlp_calibrated_forced=1 high_drop_chunk=$HIGH dualmask=$DUAL closure_batch=$BATCH" "$BUILD_OUT"

unset FORCED_BASE_BIN FORCED_BASE_LABEL FORCED_BASE_THREADS
if [[ "$DUAL" != 0 || "$BATCH" != 0 || "$HIGH" != "$BASE_HIGH" || "$THREADS" != "$BASE_THREADS" ]]; then
  BASE_LABEL="forced_joint_base_hd${BASE_HIGH}_dual0_cb0_t${BASE_THREADS}"; BASE_BIN="$ONEESAN_BUILD_DIR/b300_${BASE_LABEL}_n27"; BASE_OUT="${PREFIX}.base.build.out"; BASE_ERR="${PREFIX}.base.build.err"
  echo "=== single-pass joint selector: build baseline $BASE_LABEL ===" >&2
  N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$BASE_HIGH" DUALMASK=0 CLOSURE_BATCH=0 BUILD_ERR="$BASE_ERR" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-mlp-calibrated.sh" >"$BASE_OUT" 2>"${PREFIX}.base.build.driver.err"
  [[ -x "$BASE_BIN" ]] || { echo 'joint baseline binary missing' >&2; exit 4; }
  export FORCED_BASE_BIN="$BASE_BIN" FORCED_BASE_LABEL="$BASE_LABEL" FORCED_BASE_THREADS="$BASE_THREADS"
fi

export FORCED_OVERRIDE_BIN="$BIN" FORCED_OVERRIDE_LABEL="$LABEL" FORCED_OVERRIDE_THREADS="$THREADS"
export PROFILE_FILE ARCH SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW REBUILD_BUCKETS SELECT_ONLY
export PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
echo "SINGLE PASS JOINT primary=$LABEL base=${FORCED_BASE_LABEL:-none} select_only=$SELECT_ONLY" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
