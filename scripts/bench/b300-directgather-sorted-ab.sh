#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_sorted_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{sg+=$1+0;sm+=$2+0;if(($2+0)>mm)mm=$2+0;n++} END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

printf 'mode\trepeat\tresidue\twall_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
for sorted in 0 1; do
  mode=$([[ "$sorted" == 1 ]] && echo sorted || echo original)
  bin="$ONEESAN_BUILD_DIR/b300_directgather_${mode}_ab_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" \
    DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 SORTED="$sorted" CPASYNC_PAIR=0 \
    FORCE7=0 PREFETCH_NEXT=0 DIRECTGATHER64=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; u="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!
    sample "$pid" "$u" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true; ((rc==0)) || exit "$rc"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f"{float(sys.argv[1])+float(sys.argv[2]):.9f}")
PY
)"
    read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++} END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$u")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "$residue" "$(field wall_s "$line")" "$high" "$ag" "$am" "$mm" >>"$RESULT"
  done
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
q={}
for m in ('original','sorted'):
 g=[x for x in r if x['mode']==m]
 q[m]={k:statistics.median(float(x[k]) for x in g) for k in ('wall_s','high_s','avg_memctrl_util_pct','max_memctrl_util_pct')}
 print(m,q[m])
print(f"sorted_high_speedup={q['original']['high_s']/q['sorted']['high_s']:.6f}x")
print(f"sorted_wall_speedup={q['original']['wall_s']/q['sorted']['wall_s']:.6f}x")
PY
