#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# B300/HGX preset for the HIGH closure when HBM controller utilization is low.
# Trade a few MiB of read-only descriptor traffic for much lower integer/control
# pressure: direct rank->group descriptors + direct CROSS gather descriptors,
# then issue all source reads independently before one balanced reduction.
N="${N:-21}"
ARCH="${ARCH:-native}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
RANKFORMULA_DIRECTGATHER_FORCE7="${RANKFORMULA_DIRECTGATHER_FORCE7:-0}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_fast_n${N}}"

N="$N" ARCH="$ARCH" OUT="$OUT" \
RANKFORMULA_NOMETA_BLOCK=16 \
RANKFORMULA_NOMETA_WARPSHARE=1 \
RANKFORMULA_NOMETA_COOPGROUP=1 \
RANKFORMULA_NOMETA_COOP_UNROLL=0 \
RANKFORMULA_NOMETA_GROUP56=0 \
RANKFORMULA_NOMETA_GROUP61=1 \
RANKFORMULA_NOMETA_DIRECTMAP=1 \
RANKFORMULA_DIRECTGATHER=1 \
RANKFORMULA_DIRECTGATHER_FORCE7="$RANKFORMULA_DIRECTGATHER_FORCE7" \
RANKFORMULA_ABSTRACT_SELECT8=1 \
RANKFORMULA_ABSTRACT_DEPTH4=1 \
RANKFORMULA_ABSTRACT_SRCPACK10=1 \
RANKFORMULA_GATHER_MLP=1 \
MAXRREGCOUNT="$MAXRREGCOUNT" \
PM_ACCUM=1 \
TERNARY_KEY4=1 \
DEPTHCODE_DECODE_LOAD=ldg \
RANKSTREAM_LUT_LOAD=ldg \
TRANSPOSE_MODE="$TRANSPOSE_MODE" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh"

printf 'b300-rankformula-hbm-fast OK out=%s n=%s block=16 group61=1 directmap=1 directgather=1 force7=%s depth4=1 srcpack10=1 gather_mlp=1 maxrregcount=%s pm_accum=1\n' "$OUT" "$N" "$RANKFORMULA_DIRECTGATHER_FORCE7" "$MAXRREGCOUNT" >&2
