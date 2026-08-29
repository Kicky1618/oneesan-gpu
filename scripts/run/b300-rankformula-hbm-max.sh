#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";NGPU="${NGPU:-8}";MOD="${MOD:-4294967291}"
THREADS="${BUCKET_THREADS:-256}";GX="${BUCKET_GRID_X:-32}";GY="${BUCKET_GRID_Y:-8}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_max_n${N}}"
[[ "$NGPU" == 8 ]]||{ echo 'HBM max launcher currently requires NGPU=8' >&2;exit 2; }
if [[ "${BUILD:-1}" == 1 ]];then N="$N" OUT="$OUT" COL_ILP="${COL_ILP:-2}" DEPTHMAJOR="${DEPTHMAJOR:-1}" PREFETCH_NEXT="${PREFETCH_NEXT:-1}" FORCE7="${FORCE7:-0}" MLP_WINDOW4="${MLP_WINDOW4:-0}" MAXRREGCOUNT="${MAXRREGCOUNT:-0}" bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-max.sh";fi
[[ -x "$OUT" ]]||{ echo "missing binary $OUT" >&2;exit 3; }
echo "run HBM-max n=$N threads=$THREADS gx=$GX gy=$GY col_ilp=${COL_ILP:-2} depthmajor=${DEPTHMAJOR:-1} prefetch=${PREFETCH_NEXT:-1} force7=${FORCE7:-0}" >&2
BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" "$OUT" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD"
