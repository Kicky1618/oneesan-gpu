#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-native}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
PM_ACCUM="${PM_ACCUM:-1}"; SPARSE64="${SPARSE64:-1}"; REPEATS="${REPEATS:-2}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; ORBIT_GRID_Y="${ORBIT_GRID_Y:-128}"
LOW_GRID_X="${LOW_GRID_X:-16}"; LOW_GRID_Y="${LOW_GRID_Y:-8}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.10}"
MODES="${MODES:-ilp1 ilp2 ilp2pair ilp4pair}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2; }
fi
case "$SPARSE64" in 0|1) ;; *) echo "SPARSE64 must be 0 or 1" >&2; exit 2;; esac
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbit64_ilp_pair_n${N}_sparse${SPARSE64}}"
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

mode_params(){
  case "$1" in
    ilp1) echo '1 0' ;;
    ilp2) echo '2 0' ;;
    ilp2pair) echo '2 1' ;;
    ilp4pair) echo '4 1' ;;
    *) echo "unknown mode $1" >&2; return 2 ;;
  esac
}

printf 'mode\tilp\tpair\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\ttotal_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tsamples\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for mode in $MODES; do
  read -r ilp pair <<<"$(mode_params "$mode")"
  bin="$ONEESAN_BUILD_DIR/b300_orbit64_${mode}_sparse${SPARSE64}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" PM_ACCUM="$PM_ACCUM" \
    DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" \
    RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP="$pair" ORBITCTA_COL_ILP="$ilp" \
    RANKFORMULA_DIRECTGATHER_FORCE7=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true

  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; util="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GRID_Y" \
      BUCKET_LOW_GRID_X="$LOW_GRID_X" BUCKET_LOW_GRID_Y="$LOW_GRID_Y" \
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$ilp" "$pair" "$r" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "$high" "$ag" "$am" "$mg" "$mm" "$ns" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in sorted({r['mode'] for r in rows}):
    g=[r for r in rows if r['mode']==mode]; z={'mode':mode,'ilp':g[0]['ilp'],'pair':g[0]['pair'],'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
    out.append(z)
out.sort(key=lambda z:z['total_high_s'])
keys=('mode','ilp','pair','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z[k]) for k in keys)+'\n')
base=next((z for z in out if z['mode']=='ilp1'),None)
for z in out:
    speed=base['total_high_s']/z['total_high_s'] if base else 1.0
    print(z['mode'],f'total_high_s={z["total_high_s"]:.6f}',f'wall_s={z["wall_s"]:.6f}',f'memctrl={z["avg_memctrl_util_pct"]}',f'high_speedup_vs_ilp1={speed:.6f}x')
if out: print('BEST',out[0]['mode'])
print(f'summary={dst}')
PY

echo "b300-orbit64-ilp-pair-ab OK n=$N sparse64=$SPARSE64 modes='$MODES' result=$RESULT resource=$RESOURCE" >&2
