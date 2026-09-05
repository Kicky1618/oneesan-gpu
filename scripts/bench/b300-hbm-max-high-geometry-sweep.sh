#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}";MOD="${MOD:-4294967291}";NGPU="${NGPU:-8}";REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}";LOW_GX="${BUCKET_LOW_GRID_X:-16}";LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
GEOMETRIES="${GEOMETRIES:-16:8 32:8 64:8 128:8 32:16 64:16}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
if [[ -z "${EXPECT+x}" ]];then [[ "$N" == 21 && "$MOD" == 4294967291 ]]&&EXPECT=998035516||{ echo EXPECT required >&2;exit 2;};fi
command -v nvcc >/dev/null&&command -v nvidia-smi >/dev/null||{ echo 'nvcc+nvidia-smi required' >&2;exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l) >= NGPU ))||exit 2
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_highgeom_n${N}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";TSV="${TSV:-${PREFIX}.tsv}";mkdir -p "$LOGDIR"
BIN="$ONEESAN_BUILD_DIR/b300_hbm_highgeom_n${N}"
N="$N" ARCH="$ARCH" OUT="$BIN" COL_ILP="${COL_ILP:-2}" PAIR_MLP="${PAIR_MLP:-1}" MLP_WINDOW4="${MLP_WINDOW4:-1}" DEPTHMAJOR=1 PREFETCH_NEXT="${PREFETCH_NEXT:-0}" PM_ACCUM="${PM_ACCUM:-1}" MAXRREGCOUNT="${MAXRREGCOUNT:-0}" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-max.sh">"$LOGDIR/build.out" 2>"$LOGDIR/build.err"
field(){ local k="$1" s="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$s"|tail -n1;}
sample(){ local pid="$1" out="$2";:>"$out";while kill -0 "$pid" 2>/dev/null;do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null|awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}'>>"$out"||true;sleep "$SAMPLE_INTERVAL";done;}
printf 'high_gx\thigh_gy\trepeat\tresidue\twall_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\n'>"$TSV"
for geom in $GEOMETRIES;do IFS=: read -r hgx hgy<<<"$geom";for((r=1;r<=REPEATS;++r));do tag="x${hgx}_y${hgy}_r${r}";so="$LOGDIR/$tag.out";se="$LOGDIR/$tag.err";u="$LOGDIR/$tag.util";BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_HIGH_GRID_X="$hgx" BUCKET_HIGH_GRID_Y="$hgy" "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD">"$so" 2>"$se"&pid=$!;sample "$pid" "$u"&sp=$!;set +e;wait "$pid";rc=$?;set -e;wait "$sp"||true;((rc==0))||exit "$rc";line="$(grep '^residue=' "$so"|tail -n1)";detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";res="$(field residue "$line")";[[ "$res" == "$EXPECT" ]]||exit 4;wall="$(field wall_s "$line")";fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";high="$(python3 -c 'import sys;print(float(sys.argv[1])+float(sys.argv[2]))' "$fh" "$rh")";read -r gu mu gmax mmax < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$u");printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$hgx" "$hgy" "$r" "$res" "$wall" "$high" "$gu" "$mu" "$gmax" "$mmax">>"$TSV";done;done
cat "$TSV"
python3 - "$TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));ks=sorted({(x['high_gx'],x['high_gy']) for x in r},key=lambda q:tuple(map(int,q)))
def med(k,f):
 xs=[float(x[f]) for x in r if (x['high_gx'],x['high_gy'])==k and x[f]!='NA'];return statistics.median(xs)
for k in ks:print(f'high_geometry={k[0]}x{k[1]} wall={med(k,"wall_s"):.6f} high={med(k,"high_s"):.6f} memavg={med(k,"avg_memctrl_util_pct"):.2f}% memmax={med(k,"max_memctrl_util_pct"):.2f}%')
b=min(ks,key=lambda k:(med(k,'high_s'),med(k,'wall_s')));print(f'BEST high_gx={b[0]} high_gy={b[1]} high_s={med(b,"high_s"):.6f} wall_s={med(b,"wall_s"):.6f} memavg={med(b,"avg_memctrl_util_pct"):.2f}%')
PY
