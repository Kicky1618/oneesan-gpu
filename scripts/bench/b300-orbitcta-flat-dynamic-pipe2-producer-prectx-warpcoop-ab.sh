#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"; ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; PM_ACCUM="${PM_ACCUM:-1}"; CPASYNC_PAIR="${CPASYNC_PAIR:-1}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-0}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pipe2_producer_prectx_warpcoop_ab_n${N}_t${THREADS}_b${BATCH}_bid${PRECTX_FLAT_BID}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || {
  echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2;
}
(( THREADS >= 64 && THREADS % 32 == 0 )) || { echo 'producer prectx A/B requires BUCKET_THREADS>=64 and multiple of 32' >&2; exit 2; }
case "$BATCH" in 1|2|4|8|16) ;; *) echo 'dynamic batch must be 1,2,4,8,16' >&2; exit 2;; esac
case "$ADAPTIVE_WAVES" in 0|1|2|4) ;; *) echo 'adaptive waves must be 0,1,2,4' >&2; exit 2;; esac
(( ADAPTIVE_WAVES == 0 || BATCH > 1 )) || { echo 'adaptive waves require batch>1' >&2; exit 2; }
for x in SPARSE64 PM_ACCUM CPASYNC_PAIR PRECTX_FLAT_BID PRECTX_FLAT_BID_FUSED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$CPASYNC_PAIR" == 1 ]] || { echo 'producer prectx A/B fixes CPASYNC_PAIR=1' >&2; exit 2; }
[[ "$PRECTX_FLAT_BID_FUSED" == 0 ]] || { echo 'producer prectx warpcoop currently requires PRECTX_FLAT_BID_FUSED=0' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh" >"$LOGDIR/coverage.out"
grep -q 'b300_pipe2_producer_warp_coverage=OK' "$LOGDIR/coverage.out" || exit 5
ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" \
  >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
if [[ "$PRECTX_FLAT_BID" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" \
    >"$LOGDIR/compact-flat-bid.out" 2>"$LOGDIR/compact-flat-bid.err"
  grep -q 'bucket-compact-flat-bid-selftest OK' "$LOGDIR/compact-flat-bid.out" || { echo 'compact flat-bid gate failed' >&2; exit 5; }
fi
PIPE2_PRODUCER_PATCH_ONLY=1 PRODUCER_PRECTX_WARPCOOP=1 QUAD_MLP=1 ORBITCTA_COL_ILP=4 \
  PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED=0 \
  ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_FLAT_CHUNK=1 \
  ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh" >"$LOGDIR/producer-prectx-patch.out"
grep -q 'producer_prectx_warpcoop=1' "$LOGDIR/producer-prectx-patch.out" || { echo 'producer prectx patch marker missing' >&2; exit 5; }
grep -q "prectx_flat_bid=$PRECTX_FLAT_BID prectx_flat_bid_fused=0" "$LOGDIR/producer-prectx-patch.out" || { echo 'producer prectx cached-bid patch marker mismatch' >&2; exit 5; }

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0
  RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1
  ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ADAPTIVE_WAVES"
  ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_COL_ILP=4
  QUAD_MLP=1 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0
  PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED=0 PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
build_one(){
  local coop="$1" name="$2" bin="$ONEESAN_BUILD_DIR/b300_pipe2_producer_prectx_${name}_t${THREADS}_b${BATCH}_bid${PRECTX_FLAT_BID}_n${N}"
  env "${COMMON[@]}" PRODUCER_PRECTX_WARPCOOP="$coop" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh" \
    >"$LOGDIR/${name}.build.out" 2>"$LOGDIR/${name}.build.err"
  [[ -x "$bin" ]] || { echo "$name binary missing" >&2; exit 6; }
  grep -q 'pipe2_producer_warp=1' "$LOGDIR/${name}.build.err" || { echo "$name producer marker missing" >&2; exit 6; }
  grep -q "pipe2_producer_prectx_warpcoop=$coop" "$LOGDIR/${name}.build.err" || { echo "$name producer prectx marker mismatch" >&2; exit 6; }
  grep -q "prectx_flat_bid=$PRECTX_FLAT_BID" "$LOGDIR/${name}.build.err" || { echo "$name flat-bid marker mismatch" >&2; exit 6; }
  grep -q 'quad_mlp=1' "$LOGDIR/${name}.build.err" || { echo "$name quad marker missing" >&2; exit 6; }
  python3 "$PARSER" "$LOGDIR/${name}.build.err" --label "$name" >>"$RESOURCE" || true
  printf '%s' "$bin"
}

serial_bin="$(build_one 0 serial)"
coop_bin="$(build_one 1 warpcoop)"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null |
  awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' >>"$out" || true
  sleep "$SAMPLE_INTERVAL"
done; }

printf 'variant\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_gpu_pct\tpeak_memctrl_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\n' >"$RESULT"
run_one(){
  local name="$1" bin="$2" rep="$3" so="$LOGDIR/${name}_r${rep}.out" se="$LOGDIR/${name}_r${rep}.err" util="$LOGDIR/${name}_r${rep}.util"
  BUCKET_THREADS="$THREADS" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!
  set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  (( rc == 0 )) || { echo "$name repeat=$rep failed rc=$rc" >&2; exit "$rc"; }
  local line residue detail occ fh rh high ag am pg pm
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$name missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$name residue=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se" | head -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  [[ -n "$fh" && -n "$rh" ]] || { echo "$name missing HIGH timing" >&2; exit 7; }
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am pg pm < <(awk '{sg+=$1;sm+=$2;if($3>pg)pg=$3;if($4>pm)pm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,pg,pm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$pg" "$pm" \
    "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_one serial "$serial_bin" "$r"; done
for ((r=1;r<=REPEATS;++r)); do run_one warpcoop "$coop_bin" "$r"; done

cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL residue mismatch '+repr(sorted(res)))
def m(v,k): return statistics.median(float(x[k]) for x in rows if x['variant']==v and x[k]!='NA')
print(f'producer_prectx_wall_speedup={m("serial","wall_s")/m("warpcoop","wall_s"):.6f}x')
print(f'producer_prectx_high_speedup={m("serial","high_s")/m("warpcoop","high_s"):.6f}x')
print(f'producer_prectx_memctrl_delta={m("warpcoop","avg_memctrl_pct")-m("serial","avg_memctrl_pct"):.6f}pp')
PY

echo "pipe2 producer-prectx warpcoop A/B OK result=$RESULT resources=$RESOURCE cached_bid=$PRECTX_FLAT_BID" >&2
