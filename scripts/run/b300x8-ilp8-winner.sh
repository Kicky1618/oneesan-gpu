#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

WINNER_ENV="${WINNER_ENV:-$ONEESAN_ROOT/work/b300x8_saturate_ilp8_ab_n27/winner.env}"
[[ -f "$WINNER_ENV" ]] || { echo "winner env not found: $WINNER_ENV" >&2; echo 'run scripts/bench/b300x8-saturate-ilp8-ab.sh first' >&2; exit 2; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
: "${B300_ILP8_WINNER_BIN:?}" "${B300_ILP8_WINNER_THREADS:?}" "${B300_ILP8_WINNER_RESIDUE:?}"
[[ -x "$B300_ILP8_WINNER_BIN" ]] || { echo "winner binary missing: $B300_ILP8_WINNER_BIN" >&2; exit 2; }

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-28}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
[[ "$N" == 27 && "$NGPU" == 8 ]] || { echo 'calibrated winner is specialized for n=27, ngpu=8' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo "B300 x8 calibrated winner profile=${B300_ILP8_WINNER_PROFILE:-unknown} calibration_wall=${B300_ILP8_WINNER_WALL_S:-unknown}s calibration_mc=${B300_ILP8_WINNER_MC_AVG_PCT:-unknown}% calibration_regs=${B300_ILP8_WINNER_REGS_MAIN:-unknown} prefetch_l2=${B300_ILP8_WINNER_PREFETCH_L2:-0} cpasync=${B300_ILP8_WINNER_CPASYNC:-0} maxrregcount=${B300_ILP8_WINNER_MAXRREGCOUNT:-0}" >&2
echo "N=$N rows=$ROWS mod=$MOD threads=$B300_ILP8_WINNER_THREADS target=${TARGET_MIB}MiB plan=${PLAN_MIB}MiB window=$MAX_WINDOW GPUs=$NGPU" >&2
echo "BIN=$B300_ILP8_WINNER_BIN" >&2
export B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$B300_ILP8_WINNER_THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB"
exec "$B300_ILP8_WINNER_BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU"
