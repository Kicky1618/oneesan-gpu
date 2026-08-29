#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
REPEATS="${REPEATS:-1}"; COL_ILP="${COL_ILP:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_depthmajor_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
bash "$ONEESAN_ROOT/scripts/bench/directgather-depthmajor-proof.sh" >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

sample_process(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

printf 'layout\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for dm in 0 1; do
  label=$([[ "$dm" == 1 ]] && echo depthmajor || echo rankmajor)
  bin="$ONEESAN_BUILD_DIR/b300_directgather_${label}_ab_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" DEPTHMAJOR="$dm" \
    MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
  python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${label}_r${r}.out"; se="$LOGDIR/${label}_r${r}.err"; util="$LOGDIR/${label}_r${r}.util"
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!
    sample_process "$pid" "$util" & sampler=$!
    set +e; wait "$pid"; rc=$?; set -e; wait "$sampler" || true; ((rc==0)) || exit "$rc"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$r" "$residue" "$(field wall_s "$line")" "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" "$ag" "$am" "$mm" >>"$RESULT"
  done
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
q={}
for mode in ('rankmajor','depthmajor'):
 g=[r for r in rows if r['layout']==mode]
 z={}
 for k in ('wall_s','forward_high_s','reverse_high_s','avg_memctrl_util_pct','max_memctrl_util_pct'):
  xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
 z['high_s']=z['forward_high_s']+z['reverse_high_s']; q[mode]=z
 print(mode,z)
print(f"depthmajor_wall_speedup={q['rankmajor']['wall_s']/q['depthmajor']['wall_s']:.6f}x")
print(f"depthmajor_high_speedup={q['rankmajor']['high_s']/q['depthmajor']['high_s']:.6f}x")
PY
