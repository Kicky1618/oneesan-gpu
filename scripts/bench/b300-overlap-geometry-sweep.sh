#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; ARCH="${ARCH:-sm_103}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; REPEATS="${REPEATS:-1}"
MODE="${MODE:-pipe2}"; SORTED="${SORTED:-1}"; SPARSE64="${SPARSE64:-1}"; OVERLAP_LOCAL_CG="${OVERLAP_LOCAL_CG:-0}"
CASES="${CASES:-128:32:8 128:64:8 256:32:8 256:64:8 256:128:4 512:32:4}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.10}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo 'EXPECT required' >&2; exit 2; }; fi
case "$MODE" in cross|local|overlap|pipe2) ;; *) echo 'MODE must be cross/local/overlap/pipe2' >&2; exit 2;; esac
for x in SORTED SPARSE64 OVERLAP_LOCAL_CG; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$MODE" == overlap || "$MODE" == pipe2 || "$OVERLAP_LOCAL_CG" == 0 ]] || { echo 'OVERLAP_LOCAL_CG requires overlap or pipe2 mode' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }; command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

case "$MODE" in
 cross) local_cpa=0; overlap=0; pipe2=0 ;;
 local) local_cpa=1; overlap=0; pipe2=0 ;;
 overlap) local_cpa=0; overlap=1; pipe2=0 ;;
 pipe2) local_cpa=0; overlap=1; pipe2=1 ;;
esac
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_overlap_geometry_${MODE}_n${N}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"; bin="$ONEESAN_BUILD_DIR/b300_overlap_geometry_${MODE}_n${N}"
N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=2 PM_ACCUM=1 DEPTHMAJOR=1 PAIR_MLP=1 QUAD_MLP=0 MLP_WINDOW4=1 \
 CPASYNC_PAIR=1 CPASYNC_LOCAL_PAIR="$local_cpa" CPASYNC_OVERLAP_LOCAL_PAIR="$overlap" CPASYNC_OVERLAP_LOCAL_PIPE2="$pipe2" OVERLAP_LOCAL_CG="$OVERLAP_LOCAL_CG" \
 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" SORTED="$SORTED" FORCE7=0 PREFETCH_NEXT=0 PRECTX_FORWARD=0 PRECTX_REVERSE=0 PTXAS_VERBOSE=1 \
 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
sample_process(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'threads\thigh_gx\thigh_gy\trepeat\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_warp_occupancy_pct\treverse_warp_occupancy_pct\n' >"$RESULT"
for spec in $CASES; do
 IFS=: read -r threads gx gy <<<"$spec"; [[ "$threads" =~ ^[0-9]+$ && "$gx" =~ ^[0-9]+$ && "$gy" =~ ^[0-9]+$ ]] || { echo "bad case $spec" >&2; exit 2; }
 ((threads>=32 && threads<=1024 && threads%32==0 && gx>0 && gy>0)) || { echo "bad case $spec" >&2; exit 2; }
 for ((r=1;r<=REPEATS;++r)); do
  tag="t${threads}_x${gx}_y${gy}_r${r}"; so="$LOGDIR/$tag.out"; se="$LOGDIR/$tag.err"; util="$LOGDIR/$tag.util"
  BUCKET_THREADS="$threads" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_HIGH_GRID_X="$gx" BUCKET_HIGH_GRID_Y="$gy" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  pid=$!; sample_process "$pid" "$util" & sampler=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sampler" || true; ((rc==0)) || exit "$rc"
  line="$(grep '^residue=' "$so" | tail -n1)"; [[ "$(field residue "$line")" == "$EXPECT" ]] || { echo "$tag residue mismatch" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1)"; occ="$(grep '^rankformula_high_occupancy ' "$se" | head -n1 || true)"
  wall="$(field wall_s "$line")"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$threads" "$gx" "$gy" "$r" "$wall" "$fh" "$rh" "$high" "$ag" "$am" "$mm" "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_warp_occupancy_pct "$occ")" "$(field reverse_warp_occupancy_pct "$occ")" >>"$RESULT"
 done
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); g={}
for x in r:g.setdefault((x['threads'],x['high_gx'],x['high_gy']),[]).append(x)
out=[]
for k,v in g.items():
 z={q:statistics.median(float(x[q]) for x in v if x[q] not in ('','NA')) for q in ('wall_s','high_s','avg_memctrl_util_pct')}; out.append((z['high_s'],k,z))
for _,k,z in sorted(out):print('SUMMARY',f'threads={k[0]} gx={k[1]} gy={k[2]}',*(f'{q}={z[q]}' for q in z))
print('BEST_HIGH',sorted(out)[0][1],sorted(out)[0][2])
PY

echo "b300-overlap-geometry-sweep OK mode=$MODE result=$RESULT" >&2
