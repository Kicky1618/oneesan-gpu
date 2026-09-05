#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-native}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"; REPEATS="${REPEATS:-2}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.10}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2; }
fi
case "$COL_ILP" in 2|4) ;; *) echo "COL_ILP must be 2 or 4 for pair A/B" >&2; exit 2;; esac
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather64_pair_n${N}_ilp${COL_ILP}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

bash "$ONEESAN_ROOT/scripts/bench/rankformula-directgather64-proof.sh" >"$LOGDIR/dg64.proof.out" 2>"$LOGDIR/dg64.proof.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
sample_process(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\ttotal_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tsamples\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for pair in 0 1; do
  mode=$([[ "$pair" == 1 ]] && echo pair64 || echo scalar64)
  bin="$ONEESAN_BUILD_DIR/b300_directgather64_${mode}_n${N}_ilp${COL_ILP}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" \
    DEPTHMAJOR=1 PAIR_MLP="$pair" MLP_WINDOW4=1 CPASYNC_PAIR=0 \
    FORCE7=0 PREFETCH_NEXT=0 DIRECTGATHER64=1 MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true

  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; util="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    pid=$!; sample_process "$pid" "$util" & sampler=$!
    set +e; wait "$pid"; rc=$?; set -e
    wait "$sampler" || true
    (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    wall="$(field wall_s "$line")"; detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    high="$(python3 - "${fh:-0}" "${rh:-0}" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    read -r ag am mg mm ns < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d %d\n",sg/n,sm/n,mg,mm,n;else print "NA NA NA NA 0"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$r" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "$high" "$ag" "$am" "$mg" "$mm" "$ns" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('scalar64','pair64'):
    g=[r for r in rows if r['mode']==mode]
    z={'mode':mode,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
    out.append(z)
keys=('mode','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z[k]) for k in keys)+'\n')
q={z['mode']:z for z in out}
print(f'pair64_wall_speedup={q["scalar64"]["wall_s"]/q["pair64"]["wall_s"]:.6f}x')
print(f'pair64_total_high_speedup={q["scalar64"]["total_high_s"]/q["pair64"]["total_high_s"]:.6f}x')
if q['scalar64']['avg_memctrl_util_pct'] is not None and q['pair64']['avg_memctrl_util_pct'] is not None:
    print(f'pair64_memctrl_delta={q["pair64"]["avg_memctrl_util_pct"]-q["scalar64"]["avg_memctrl_util_pct"]:.6f}pp')
print('descriptor=directgather64')
print('pair_issue=index_primary_rare_then_source_windows')
print(f'summary={dst}')
PY

echo "b300-directgather64-pair-ab OK n=$N ilp=$COL_ILP repeats=$REPEATS result=$RESULT resource=$RESOURCE" >&2
