#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"; ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; COL_ILP="${ORBITCTA_COL_ILP:-2}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; PM_ACCUM="${PM_ACCUM:-1}"
QUAD_MLP="${QUAD_MLP:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"; PRECTX_COMPACT="${PRECTX_COMPACT:-0}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-0}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pipe2_producer_selected_ab_n${N}_t${THREADS}_b${BATCH}_q${QUAD_MLP}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2; }
(( THREADS >= 64 && THREADS % 32 == 0 )) || { echo 'producer-warp A/B requires BUCKET_THREADS>=64 and multiple of 32' >&2; exit 2; }
case "$BATCH" in 1|2|4|8|16) ;; *) echo 'bad dynamic batch' >&2; exit 2;; esac
case "$ADAPTIVE_WAVES" in 0|1|2|4) ;; *) echo 'bad adaptive waves' >&2; exit 2;; esac
case "$COL_ILP" in 1|2|4) ;; *) echo 'bad ORBITCTA_COL_ILP' >&2; exit 2;; esac
(( ADAPTIVE_WAVES == 0 || BATCH > 1 )) || { echo 'adaptive waves require batch>1' >&2; exit 2; }
for x in SPARSE64 CPASYNC_PAIR PM_ACCUM QUAD_MLP PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT PRECTX_FLAT_BID PRECTX_FLAT_BID_FUSED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$QUAD_MLP" == 0 || "$COL_ILP" == 4 ]] || { echo 'QUAD_MLP=1 requires COL_ILP=4' >&2; exit 2; }
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }; command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh" >"$LOGDIR/coverage.out"
grep -q 'b300_pipe2_producer_warp_coverage=OK' "$LOGDIR/coverage.out" || exit 5
PIPE2_PRODUCER_PATCH_ONLY=1 QUAD_MLP="$QUAD_MLP" ORBITCTA_COL_ILP="$COL_ILP" \
  PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" PRECTX_FLAT_BID="$PRECTX_FLAT_BID" \
  ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh" >"$LOGDIR/producer-patch.out"
grep -q 'b300_pipe2_producer_warp_patch=OK' "$LOGDIR/producer-patch.out" || exit 5

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0
 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1
 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ADAPTIVE_WAVES"
 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_COL_ILP="$COL_ILP"
 QUAD_MLP="$QUAD_MLP" QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1
 PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT"
 PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
std_bin="$ONEESAN_BUILD_DIR/b300_pipe2_selected_standard_t${THREADS}_b${BATCH}_q${QUAD_MLP}_n${N}"
env "${COMMON[@]}" OUT="$std_bin" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/standard.build.out" 2>"$LOGDIR/standard.build.err"
grep -q "flat_dynamic_pipe2=1" "$LOGDIR/standard.build.err" || { echo 'standard pipe2 marker missing' >&2; exit 6; }
python3 "$PARSER" "$LOGDIR/standard.build.err" --label standard >>"$RESOURCE" || true
prod_bin="$ONEESAN_BUILD_DIR/b300_pipe2_selected_producer_t${THREADS}_b${BATCH}_q${QUAD_MLP}_n${N}"
env "${COMMON[@]}" PRODUCER_PRECTX_WARPCOOP=0 OUT="$prod_bin" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh" >"$LOGDIR/producer.build.out" 2>"$LOGDIR/producer.build.err"
grep -q 'pipe2_producer_warp=1' "$LOGDIR/producer.build.err" || { echo 'producer build marker missing' >&2; exit 6; }
grep -q "quad_mlp=$QUAD_MLP" "$LOGDIR/producer.build.err" || { echo 'producer quad marker mismatch' >&2; exit 6; }
python3 "$PARSER" "$LOGDIR/producer.build.err" --label producer >>"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do
 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null |
 awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' >>"$out" || true
 sleep "$SAMPLE_INTERVAL"; done; }
printf 'variant\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_gpu_pct\tpeak_memctrl_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\n' >"$RESULT"
run_one(){
 local name="$1" bin="$2" rep="$3" so="$LOGDIR/${name}_r${rep}.out" se="$LOGDIR/${name}_r${rep}.err" util="$LOGDIR/${name}_r${rep}.util"
 runenv=(BUCKET_THREADS="$THREADS" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
 [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
 [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
 env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
 local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true; ((rc==0)) || { echo "$name failed rc=$rc" >&2; exit "$rc"; }
 local line residue detail occ fh rh high ag am pg pm
 line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || { echo "$name missing residue" >&2; exit 3; }
 residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$name residue=$residue expected=$EXPECT" >&2; exit 4; }
 detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)"; occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
 fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; [[ -n "$fh" && -n "$rh" ]] || { echo "$name missing HIGH timing" >&2; exit 5; }
 high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
 read -r ag am pg pm < <(awk '{sg+=$1;sm+=$2;if($3>pg)pg=$3;if($4>pm)pm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,pg,pm;else print "NA NA NA NA"}' "$util")
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$pg" "$pm" "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_one standard "$std_bin" "$r"; done
for ((r=1;r<=REPEATS;++r)); do run_one producer "$prod_bin" "$r"; done
cat "$RESULT"
python3 - "$RESULT" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,out=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t'))
def m(v,k): return statistics.median(float(x[k]) for x in rows if x['variant']==v and x[k]!='NA')
res={x['residue'] for x in rows}
if len(res)!=1: raise SystemExit('FATAL residue mismatch '+repr(sorted(res)))
w='producer' if m('producer','wall_s') < m('standard','wall_s') else 'standard'
print(f'producer_wall_speedup={m("standard","wall_s")/m("producer","wall_s"):.6f}x winner={w}')
print(f'producer_high_speedup={m("standard","high_s")/m("producer","high_s"):.6f}x')
with open(out,'w') as f:
 f.write(f'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP={1 if w=="producer" else 0}\n')
 f.write('ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP=0\n')
 f.write(f'DYNAMIC_PRODUCER_WALL_S={m(w,"wall_s"):.9f}\nDYNAMIC_PRODUCER_HIGH_S={m(w,"high_s"):.9f}\n')
PY
echo "selected pipe2 producer-warp A/B OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
