#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"; SORTED="${SORTED:-0}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2;
  }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather64_quad_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
sample_process(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'mode\tsparse64\tquad\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for sparse in 0 1; do
  for quad in 0 1; do
    mode=$([[ "$sparse" == 1 ]] && echo sparse || echo dense)
    mode="${mode}_$([[ "$quad" == 1 ]] && echo quad4 || echo pair4)"
    bin="$ONEESAN_BUILD_DIR/b300_dg64_${mode}_n${N}"
    N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=4 PM_ACCUM="$PM_ACCUM" \
      DEPTHMAJOR=1 PAIR_MLP=1 QUAD_MLP="$quad" MLP_WINDOW4=1 \
      DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" SORTED="$SORTED" \
      CPASYNC_PAIR=0 CPASYNC_LOCAL_PAIR=0 FORCE7=0 PREFETCH_NEXT=0 \
      PRECTX_FORWARD=0 PRECTX_REVERSE=0 PTXAS_VERBOSE=1 \
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
      detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
      wall="$(field wall_s "$line")"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
      high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f"{float(sys.argv[1])+float(sys.argv[2]):.9f}")
PY
)"
      read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$sparse" "$quad" "$r" "$residue" "$wall" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$RESULT"
    done
  done
done

cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['mode'],[]).append(r)
res={m:{x['residue'] for x in g} for m,g in by.items()}
if any(len(v)!=1 for v in res.values()) or len({next(iter(v)) for v in res.values()})!=1:
    raise SystemExit(f'RESIDUE MISMATCH {res}')
print('residue_match=OK',next(iter(next(iter(res.values())))))
q={}
for m,g in by.items():
    high=statistics.median(float(x['high_s']) for x in g)
    wall=statistics.median(float(x['wall_s']) for x in g)
    mc=statistics.median(float(x['avg_memctrl_util_pct']) for x in g if x['avg_memctrl_util_pct']!='NA')
    q[m]=(high,wall,mc)
    print(m,f'high_s={high:.6f}',f'wall_s={wall:.6f}',f'mc_avg_pct={mc:.3f}')
for family in ('dense','sparse'):
    a=q[f'{family}_pair4']; b=q[f'{family}_quad4']
    print(f'{family}_quad_high_speedup={a[0]/b[0]:.6f}x',f'wall_speedup={a[1]/b[1]:.6f}x',f'mc_delta_pct={b[2]-a[2]:.3f}')
print('BEST_HIGH',min(q,key=lambda k:q[k][0]))
PY

echo "b300-directgather64-quad-ab OK result=$RESULT resources=$RESOURCE" >&2
