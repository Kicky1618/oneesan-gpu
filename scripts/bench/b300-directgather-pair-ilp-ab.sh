#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-21}";MOD="${MOD:-4294967291}";NGPU="${NGPU:-8}";REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}";GX="${BUCKET_HIGH_GRID_X:-32}";GY="${BUCKET_HIGH_GRID_Y:-8}";SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
if [[ -z "${EXPECT+x}" ]];then [[ "$N" == 21 && "$MOD" == 4294967291 ]]&&EXPECT=998035516||{ echo EXPECT required >&2;exit 2;};fi
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pair_ilp_n${N}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";TSV="${TSV:-${PREFIX}.tsv}";mkdir -p "$LOGDIR"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
field(){ local k="$1" s="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$s"|tail -n1;}
sample(){ local pid="$1" out="$2";:>"$out";while kill -0 "$pid" 2>/dev/null;do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null|awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}'>>"$out"||true;sleep "$SAMPLE_INTERVAL";done;}
printf 'col_ilp\trepeat\tresidue\twall_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tmax_regs\tspill_store_bytes\tspill_load_bytes\n'>"$TSV"
for ilp in 2 4;do bin="$ONEESAN_BUILD_DIR/b300_pair_ilp${ilp}_n${N}";be="$LOGDIR/ilp${ilp}.build.err";N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$ilp" DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 PREFETCH_NEXT="${PREFETCH_NEXT:-0}" FORCE7=0 PM_ACCUM="${PM_ACCUM:-1}" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh">"$LOGDIR/ilp${ilp}.build.out" 2>"$be";read -r regs ss sl < <(python3 - "$be" <<'PY'
import re,sys
s=open(sys.argv[1],errors='ignore').read();r=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)];p=[tuple(map(int,x)) for x in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)];print(max(r) if r else -1,sum(a for a,b in p),sum(b for a,b in p))
PY
);for((r=1;r<=REPEATS;++r));do so="$LOGDIR/ilp${ilp}_r${r}.out";se="$LOGDIR/ilp${ilp}_r${r}.err";u="$LOGDIR/ilp${ilp}_r${r}.util";BUCKET_THREADS="$THREADS" BUCKET_GRID_X=16 BUCKET_GRID_Y=8 BUCKET_HIGH_GRID_X="$GX" BUCKET_HIGH_GRID_Y="$GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD">"$so" 2>"$se"&pid=$!;sample "$pid" "$u"&sp=$!;set +e;wait "$pid";rc=$?;set -e;wait "$sp"||true;((rc==0))||exit "$rc";line="$(grep '^residue=' "$so"|tail -n1)";detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";res="$(field residue "$line")";[[ "$res" == "$EXPECT" ]]||exit 4;wall="$(field wall_s "$line")";fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";high="$(python3 -c 'import sys;print(float(sys.argv[1])+float(sys.argv[2]))' "$fh" "$rh")";read -r gu mu _ mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$u");printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ilp" "$r" "$res" "$wall" "$high" "$gu" "$mu" "$mm" "$regs" "$ss" "$sl">>"$TSV";done;done
cat "$TSV"
python3 - "$TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(i,k):return statistics.median(float(x[k]) for x in r if x['col_ilp']==i and x[k]!='NA')
for i in ('2','4'):print(f'ilp={i} high={med(i,"high_s"):.6f} wall={med(i,"wall_s"):.6f} memavg={med(i,"avg_memctrl_util_pct"):.2f}% memmax={med(i,"max_memctrl_util_pct"):.2f}%')
print(f'ilp4_high_speedup={med("2","high_s")/med("4","high_s"):.6f}x')
PY
