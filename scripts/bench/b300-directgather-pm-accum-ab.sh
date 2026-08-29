#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-21}";MOD="${MOD:-4294967291}";NGPU="${NGPU:-8}";REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";ARCH="${ARCH:-native}";COL_ILP="${COL_ILP:-2}"
THREADS="${BUCKET_THREADS:-256}";GX="${BUCKET_GRID_X:-32}";GY="${BUCKET_GRID_Y:-8}";SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}";PREFETCH_NEXT="${PREFETCH_NEXT:-1}"
if [[ -z "${EXPECT+x}" ]];then [[ "$N" == 21 && "$MOD" == 4294967291 ]]&&EXPECT=998035516||{ echo EXPECT required >&2;exit 2;};fi
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pmaccum_n${N}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";TSV="${TSV:-${PREFIX}.tsv}";mkdir -p "$LOGDIR"
field(){ local k="$1" s="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$s"|tail -n1;}
sample(){ local pid="$1" out="$2";:>"$out";while kill -0 "$pid" 2>/dev/null;do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null|awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm}'>>"$out"||true;sleep "$SAMPLE_INTERVAL";done;}
printf 'pm_accum\trepeat\tresidue\twall_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n'>"$TSV"
for pm in 0 1;do bin="$ONEESAN_BUILD_DIR/b300_pmaccum${pm}_n${N}";N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" DEPTHMAJOR=1 PREFETCH_NEXT="$PREFETCH_NEXT" PM_ACCUM="$pm" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh">"$LOGDIR/pm${pm}.build.out" 2>"$LOGDIR/pm${pm}.build.err";for((r=1;r<=REPEATS;++r));do so="$LOGDIR/pm${pm}_r${r}.out";se="$LOGDIR/pm${pm}_r${r}.err";u="$LOGDIR/pm${pm}_r${r}.util";BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD">"$so" 2>"$se"&pid=$!;sample "$pid" "$u"&sp=$!;set +e;wait "$pid";rc=$?;set -e;wait "$sp"||true;((rc==0))||exit "$rc";line="$(grep '^residue=' "$so"|tail -n1)";detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";res="$(field residue "$line")";[[ "$res" == "$EXPECT" ]]||exit 4;wall="$(field wall_s "$line")";fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";high="$(python3 -c 'import sys;print(float(sys.argv[1])+float(sys.argv[2]))' "$fh" "$rh")";read -r gu mu mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$u");printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pm" "$r" "$res" "$wall" "$high" "$gu" "$mu" "$mm">>"$TSV";done;done
cat "$TSV"
python3 - "$TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(m,k):
 x=[float(z[k]) for z in r if z['pm_accum']==m and z[k]!='NA'];return statistics.median(x)
for m in ('0','1'):print(f'pm={m} wall={med(m,"wall_s"):.6f} high={med(m,"high_s"):.6f} memavg={med(m,"avg_memctrl_util_pct"):.2f}% memmax={med(m,"max_memctrl_util_pct"):.2f}%')
print(f'pm0_high_speedup_vs_pm1={med("1","high_s")/med("0","high_s"):.6f}x')
PY
