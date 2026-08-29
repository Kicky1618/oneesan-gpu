#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"; BATCHES="${BATCHES:-1 2 4 8 16}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; COL_ILP="${ORBITCTA_COL_ILP:-2}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"; PRECTX_COMPACT="${PRECTX_COMPACT:-0}"
PM_ACCUM="${PM_ACCUM:-1}"; PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_dynamic_batch_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2; }
for x in SPARSE64 CPASYNC_PAIR PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT PM_ACCUM; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$COL_ILP" in 2|4) ;; *) echo 'dynamic batch sweep expects ORBITCTA_COL_ILP=2 or 4' >&2; exit 2;; esac
for b in $BATCHES; do case "$b" in 1|2|4|8|16) ;; *) echo "bad dynamic batch=$b" >&2; exit 2;; esac; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }
if [[ "$PRECTX_COMPACT" == 1 ]]; then ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/prectx.out" 2>"$LOGDIR/prectx.err"; fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync.out" 2>"$LOGDIR/cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync.out" || { echo 'cp.async peer gate failed' >&2; exit 5; }
fi

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_COL_ILP="$COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" PRECTX_FLAT_BID=0 PRECTX_FLAT_BID_FUSED=0 PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
STATIC_BIN="$ONEESAN_BUILD_DIR/b300_flat_dynamic_batch_static_n${N}"
env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC=0 ORBITCTA_FLAT_DYNAMIC_BATCH=1 OUT="$STATIC_BIN" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/static.build.out" 2>"$LOGDIR/static.build.err"
python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/static.build.err" --label static >>"$RESOURCE" || true
for b in $BATCHES; do
  bin="$ONEESAN_BUILD_DIR/b300_flat_dynamic_batch${b}_n${N}"
  env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$b" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/b${b}.build.out" 2>"$LOGDIR/b${b}.build.err"
  grep -q "flat_dynamic=1 flat_dynamic_batch=$b" "$LOGDIR/b${b}.build.err" || { echo "dynamic batch build marker mismatch b=$b" >&2; exit 6; }
  python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/b${b}.build.err" --label "dynamic_b${b}" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'mode\tdynamic\tbatch\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
run_one(){
  local mode="$1" dynamic="$2" batch="$3" bin="$4" rep="$5" so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err" util="$LOGDIR/${mode}_r${rep}.util"
  env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
  local line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
  local detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)" fh rh high ag am mm
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; [[ -n "$fh" && -n "$rh" ]] || { echo "$mode missing HIGH timing" >&2; exit 5; }
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$dynamic" "$batch" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_one static 0 1 "$STATIC_BIN" "$r"; done
for b in $BATCHES; do for ((r=1;r<=REPEATS;++r)); do run_one "dynamic_b${b}" 1 "$b" "$ONEESAN_BUILD_DIR/b300_flat_dynamic_batch${b}_n${N}" "$r"; done; done

cat "$RESULT"
python3 - "$RESULT" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,winner=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in dict.fromkeys(r['mode'] for r in rows):
 g=[r for r in rows if r['mode']==mode]
 wall=statistics.median(float(r['wall_s']) for r in g); high=statistics.median(float(r['high_s']) for r in g)
 mc=[float(r['avg_memctrl_util_pct']) for r in g if r['avg_memctrl_util_pct']!='NA']; m=statistics.median(mc) if mc else float('nan')
 out.append((wall,high,mode,int(g[0]['dynamic']),int(g[0]['batch']),m))
for w,h,n,d,b,m in sorted(out): print('DYNAMIC_BATCH',n,f'wall_s={w:.6f}',f'high_s={h:.6f}',f'mc_avg_pct={m:.3f}')
best=min(out)
with open(winner,'w') as f:
 f.write('ORBITCTA_FLAT=1\nORBITCTA_FLAT_CHUNK=1\n')
 f.write(f'ORBITCTA_FLAT_DYNAMIC={best[3]}\nORBITCTA_FLAT_DYNAMIC_BATCH={best[4]}\n')
 f.write('ORBIT_QUAD_MLP=0\nORBIT_QUAD_OVERLAP_LOCAL=0\nORBIT_QUAD_LOCAL_DIRECT_MAX=0\nORBIT_QUAD_SPARSE_DESC_MLP=0\nORBIT_QUAD_OVERLAP_BYPASS_LOCAL0=0\nORBIT_QUAD_CPASYNC_GROUP_COLS=1\nORBIT_QUAD_CPASYNC_PREFETCH_BYTES=0\nORBIT_PRECTX_FLAT_BID=0\nORBIT_PRECTX_FLAT_BID_FUSED=0\nORBIT_PRECTX_WARPCOOP=0\n')
 f.write(f'ORBIT_DYNAMIC_WALL_S={best[0]:.9f}\nORBIT_DYNAMIC_HIGH_S={best[1]:.9f}\nORBIT_DYNAMIC_PROFILE={best[2]}\n')
print('BEST_DYNAMIC_BATCH',f'mode={best[2]}',f'dynamic={best[3]}',f'batch={best[4]}',f'wall_s={best[0]:.6f}',f'high_s={best[1]:.6f}',f'winner_env={winner}')
PY

echo "dynamic flat batch sweep OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
