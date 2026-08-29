#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Fast B300x8 response path for low memory-controller utilization.
# Tune on the exact n=21 residue, rank candidates by HIGH time, then sweep only
# the winning binary's HIGH launch geometry. MC utilization and occupancy are
# recorded as diagnostics, never as the optimization objective.
N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
GEOMETRIES="${GEOMETRIES:-16:8 32:8 64:8 128:8 32:16 64:16}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_emergency}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
CAND_TSV="${CAND_TSV:-${PREFIX}_candidates.tsv}"; GEOM_TSV="${GEOM_TSV:-${PREFIX}_geometry.tsv}"
mkdir -p "$LOGDIR"
[[ "$N" == 21 && "$MOD" == 4294967291 ]] || { echo "emergency tuner defaults require N=21/MOD=4294967291; override EXPECT knowingly" >&2; }
command -v nvcc >/dev/null && command -v nvidia-smi >/dev/null || { echo "nvcc+nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

field(){ local k="$1" s="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$s" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

# label col_ilp pair window4 prefetch pm force7 maxrregcount
CANDIDATES=(
  'single2 2 0 1 0 1 0 0'
  'pair2 2 1 1 0 1 0 0'
  'pair4 4 1 1 0 1 0 0'
  'pair2_pm32 2 1 1 0 0 0 0'
  'pair2_prefetch 2 1 1 1 1 0 0'
  'full2 2 0 0 0 1 0 0'
  'force7 2 0 0 0 1 1 0'
)

declare -A BIN SPEC
printf 'mode\trepeat\tresidue\twall_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tmin_forward_occ_pct\tmin_reverse_occ_pct\tmax_regs\tspill_store_bytes\tspill_load_bytes\n' >"$CAND_TSV"
for spec in "${CANDIDATES[@]}"; do
  read -r mode ilp pair win pf pm force cap <<<"$spec"; SPEC[$mode]="$ilp $pair $win $pf $pm $force $cap"
  bin="$ONEESAN_BUILD_DIR/b300_emergency_${mode}_n${N}"; BIN[$mode]="$bin"; bo="$LOGDIR/$mode.build.out"; be="$LOGDIR/$mode.build.err"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$ilp" DEPTHMAJOR=1 PAIR_MLP="$pair" MLP_WINDOW4="$win" PREFETCH_NEXT="$pf" PM_ACCUM="$pm" FORCE7="$force" MAXRREGCOUNT="$cap" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$bo" 2>"$be"
  read -r regs ss sl < <(python3 - "$be" <<'PY'
import re,sys
s=open(sys.argv[1],errors='ignore').read();r=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)];p=[tuple(map(int,x)) for x in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)];print(max(r) if r else -1,sum(a for a,b in p),sum(b for a,b in p))
PY
)
  for ((rep=1;rep<=REPEATS;++rep)); do
    so="$LOGDIR/${mode}_r${rep}.out"; se="$LOGDIR/${mode}_r${rep}.err"; util="$LOGDIR/${mode}_r${rep}.util"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_HIGH_GRID_X=32 BUCKET_HIGH_GRID_Y=8 "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!
    sample "$pid" "$util" & sampler=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sampler" || true; ((rc==0)) || exit "$rc"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; [[ -n "$line" && -n "$detail" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
    wall="$(field wall_s "$line")"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; high="$(python3 -c 'import sys;print(float(sys.argv[1])+float(sys.argv[2]))' "$fh" "$rh")"
    read -r gu mu _ mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    read -r focc rocc < <(python3 - "$se" <<'PY'
import re,sys
s=open(sys.argv[1],errors='ignore').read();f=[float(x) for x in re.findall(r'forward_warp_occupancy_pct=([0-9.]+)',s)];r=[float(x) for x in re.findall(r'reverse_warp_occupancy_pct=([0-9.]+)',s)];print(min(f) if f else -1,min(r) if r else -1)
PY
)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$residue" "$wall" "$high" "$gu" "$mu" "$mm" "$focc" "$rocc" "$regs" "$ss" "$sl" >>"$CAND_TSV"
  done
done
cat "$CAND_TSV"
BEST="$(python3 - "$CAND_TSV" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));m=sorted({x['mode'] for x in r})
def med(q,k):return statistics.median(float(x[k]) for x in r if x['mode']==q and x[k]!='NA')
for q in m:print(f'candidate={q} high={med(q,"high_s"):.6f} wall={med(q,"wall_s"):.6f} memavg={med(q,"avg_memctrl_util_pct"):.2f}% occF={med(q,"min_forward_occ_pct"):.1f}% occR={med(q,"min_reverse_occ_pct"):.1f}% regs={med(q,"max_regs"):.0f}',file=sys.stderr)
print(min(m,key=lambda q:(med(q,'high_s'),med(q,'wall_s'))))
PY
)"
echo "emergency_best_candidate=$BEST spec=${SPEC[$BEST]}" >&2
BEST_BIN="${BIN[$BEST]}"

printf 'high_gx\thigh_gy\trepeat\tresidue\twall_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$GEOM_TSV"
for geom in $GEOMETRIES; do IFS=: read -r hgx hgy <<<"$geom"; for ((rep=1;rep<=REPEATS;++rep)); do tag="best_${BEST}_x${hgx}_y${hgy}_r${rep}";so="$LOGDIR/$tag.out";se="$LOGDIR/$tag.err";util="$LOGDIR/$tag.util";BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_HIGH_GRID_X="$hgx" BUCKET_HIGH_GRID_Y="$hgy" "$BEST_BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!;sample "$pid" "$util"&sampler=$!;set +e;wait "$pid";rc=$?;set -e;wait "$sampler"||true;((rc==0))||exit "$rc";line="$(grep '^residue=' "$so"|tail -n1)";detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1)";residue="$(field residue "$line")";[[ "$residue" == "$EXPECT" ]]||exit 4;wall="$(field wall_s "$line")";fh="$(field forward_high_s "$detail")";rh="$(field reverse_high_s "$detail")";high="$(python3 -c 'import sys;print(float(sys.argv[1])+float(sys.argv[2]))' "$fh" "$rh")";read -r gu mu _ mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util");printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$hgx" "$hgy" "$rep" "$residue" "$wall" "$high" "$gu" "$mu" "$mm" >>"$GEOM_TSV";done;done
cat "$GEOM_TSV"
python3 - "$GEOM_TSV" "$BEST" "${SPEC[$BEST]}" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));mode=sys.argv[2];spec=sys.argv[3];ks=sorted({(x['high_gx'],x['high_gy']) for x in r},key=lambda q:tuple(map(int,q)))
def med(k,f):return statistics.median(float(x[f]) for x in r if (x['high_gx'],x['high_gy'])==k and x[f]!='NA')
b=min(ks,key=lambda k:(med(k,'high_s'),med(k,'wall_s')))
for k in ks:print(f'geometry={k[0]}x{k[1]} high={med(k,"high_s"):.6f} wall={med(k,"wall_s"):.6f} memavg={med(k,"avg_memctrl_util_pct"):.2f}%')
print(f'EMERGENCY_BEST mode={mode} spec="{spec}" high_gx={b[0]} high_gy={b[1]} high_s={med(b,"high_s"):.6f} wall_s={med(b,"wall_s"):.6f} memavg={med(b,"avg_memctrl_util_pct"):.2f}%')
PY
