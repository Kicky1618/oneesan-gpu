#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; ARCH="${ARCH:-sm_103}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; MODE="${MODE:-pipe2}"
THREADS="${THREADS:-256}"; HIGH_GX="${HIGH_GX:-32}"; HIGH_GY="${HIGH_GY:-8}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"; SORTED="${SORTED:-1}"; SPARSE64="${SPARSE64:-1}"
case "$MODE" in cross|local|overlap|pipe2) ;; *) echo 'MODE must be cross/local/overlap/pipe2' >&2; exit 2;; esac
for x in SORTED SPARSE64; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$MODE" in
 cross) local_cpa=0; overlap=0; pipe2=0 ;;
 local) local_cpa=1; overlap=0; pipe2=0 ;;
 overlap) local_cpa=0; overlap=1; pipe2=0 ;;
 pipe2) local_cpa=0; overlap=1; pipe2=1 ;;
esac
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_high_plan_profile_${MODE}_n${N}}"; BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_high_plan_profile_${MODE}_n${N}}"
mkdir -p "$(dirname "$PREFIX")"
N="$N" ARCH="$ARCH" OUT="$BIN" COL_ILP=2 PM_ACCUM=1 DEPTHMAJOR=1 PAIR_MLP=1 QUAD_MLP=0 MLP_WINDOW4=1 \
 CPASYNC_PAIR=1 CPASYNC_LOCAL_PAIR="$local_cpa" CPASYNC_OVERLAP_LOCAL_PAIR="$overlap" CPASYNC_OVERLAP_LOCAL_PIPE2="$pipe2" OVERLAP_LOCAL_CG=0 \
 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" SORTED="$SORTED" HIGH_PLAN_PROFILE=1 \
 FORCE7=0 PREFETCH_NEXT=0 PRECTX_FORWARD=0 PRECTX_REVERSE=0 PTXAS_VERBOSE=1 \
 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"

BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
 BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_HIGH_GRID_X="$HIGH_GX" BUCKET_HIGH_GRID_Y="$HIGH_GY" \
 "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"${PREFIX}.out" 2>"${PREFIX}.err"

grep '^rankformula_plan_profile_summary ' "${PREFIX}.err" | tee "${PREFIX}.summary"
python3 - "${PREFIX}.summary" <<'PY'
import re,sys
rows=[]
for line in open(sys.argv[1]):
    d=dict(re.findall(r'(\w+)=([^\s]+)',line));
    if d: rows.append(d)
for phase in ('forward','reverse'):
    g=[x for x in rows if x.get('phase')==phase]
    if not g: continue
    cols=sum(int(x['columns']) for x in g)
    reads=sum(int(x['local_source_reads']) for x in g)
    # Each GPU profile reports its own owner traffic. Aggregate by work columns.
    cross_cols=sum(float(x['cross_column_fraction'])*int(x['columns']) for x in g)
    print(f'AGG phase={phase} columns={cols} local_source_reads={reads} avg_local_sources_per_column={reads/cols if cols else 0:.6f} cross_column_fraction={cross_cols/cols if cols else 0:.6f}')
PY

echo "b300-high-plan-profile OK mode=$MODE summary=${PREFIX}.summary" >&2
