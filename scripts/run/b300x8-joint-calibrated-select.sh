#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}";if(($#>0));then shift;fi
[[ "$N" == 27 ]]||{ echo 'joint calibrated selector targets n=27' >&2;exit 2; }
ARCH="${ARCH:-native}";PRIME="${SMOKE_PRIME:-4294967291}";TARGET_MIB="${FORCED_TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_joint_calibrated_select_n27}";CAL_PREFIX="${CAL_PREFIX:-${PREFIX}.calibration}";CAL_LOG="${CAL_LOG:-${PREFIX}.calibration.log}"
RECALIBRATE="${RECALIBRATE:-1}";SELECT_ONLY="${SELECT_ONLY:-1}"
[[ "$RECALIBRATE" == 0 || "$RECALIBRATE" == 1 ]]||{ echo 'RECALIBRATE must be 0 or 1' >&2;exit 2; }
[[ "$SELECT_ONLY" == 0 || "$SELECT_ONLY" == 1 ]]||{ echo 'SELECT_ONLY must be 0 or 1' >&2;exit 2; }
mkdir -p "$(dirname "$PREFIX")"
getv(){ local k="$1" f="$2";sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f"|tail -n1; }

if [[ "$RECALIBRATE" == 1 || ! -s "$CAL_LOG" ]];then
  echo '=== joint calibrated selector: exhaustive forced row calibration ===' >&2
  ARCH="$ARCH" MOD="$PRIME" ROWS="${CAL_ROWS:-1}" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  THREADS_LIST="${THREADS_LIST:-128 256 512}" BATCH_LIST="${BATCH_LIST:-2 4}" HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}" REPEATS="${REPEATS:-1}" TRANSFORM_MIN_SPEEDUP="${TRANSFORM_MIN_SPEEDUP:-1.01}" PREFIX="$CAL_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-forced-joint-calibrate.sh" | tee "$CAL_LOG"
fi
[[ "$(getv b300_forced_joint_exact_gates "$CAL_LOG")" == 1 ]]||{ echo 'joint calibration exact gate missing' >&2;exit 3; }
HIGH="$(getv b300_forced_joint_final_high_drop_chunk "$CAL_LOG")";DUAL="$(getv b300_forced_joint_final_dualmask "$CAL_LOG")";BATCH="$(getv b300_forced_joint_final_closure_batch "$CAL_LOG")";THREADS="$(getv b300_forced_joint_final_threads "$CAL_LOG")";MODE="$(getv b300_forced_joint_final_mode "$CAL_LOG")"
[[ "$HIGH" == 0 || "$HIGH" == 1 ]]||{ echo "bad highdrop=$HIGH" >&2;exit 3; };[[ "$DUAL" == 0 || "$DUAL" == 1 ]]||{ echo "bad dualmask=$DUAL" >&2;exit 3; };case "$BATCH" in 0|2|4);;*)echo "bad closure_batch=$BATCH" >&2;exit 3;;esac;[[ "$THREADS" =~ ^[0-9]+$ ]]||{ echo "bad threads=$THREADS" >&2;exit 3; }
LABEL="forced_joint_${MODE}_hd${HIGH}_dual${DUAL}_cb${BATCH}_t${THREADS}";BIN="$ONEESAN_BUILD_DIR/b300_${LABEL}_n27";BUILD_OUT="${PREFIX}.build.out";BUILD_ERR="${PREFIX}.build.err"
echo "=== joint calibrated selector: build $LABEL ===" >&2
N=27 ARCH="$ARCH" OUT="$BIN" HIGH_DROP_CHUNK="$HIGH" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" BUILD_ERR="$BUILD_ERR" PTXAS_VERBOSE=1 \
 bash "$ONEESAN_ROOT/scripts/build/b300-forced-mlp-calibrated.sh" >"$BUILD_OUT" 2>"${PREFIX}.build.driver.err"
[[ -x "$BIN" ]]||{ echo 'joint calibrated binary missing' >&2;exit 4; };grep -Fq "mlp_calibrated_forced=1 high_drop_chunk=$HIGH dualmask=$DUAL closure_batch=$BATCH" "$BUILD_OUT"
echo "JOINT CALIBRATED FORCED label=$LABEL binary=$BIN select_only=$SELECT_ONLY" >&2

export SELECT_ONLY FORCED_OVERRIDE_BIN="$BIN" FORCED_OVERRIDE_LABEL="$LABEL" FORCED_OVERRIDE_THREADS="$THREADS" PROFILE_FILE SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW
export PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled.sh" 27 "$@"
