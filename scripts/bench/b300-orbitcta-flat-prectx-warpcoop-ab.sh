#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
CHUNK="${ORBITCTA_FLAT_CHUNK:-4}"; COL_ILP="${ORBITCTA_COL_ILP:-4}"; REPEATS="${REPEATS:-1}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; PAIR_MLP="${PAIR_MLP:-1}"; CPASYNC_PAIR="${CPASYNC_PAIR:-1}"
QUAD_MLP="${QUAD_MLP:-1}"; QUAD_OVERLAP_LOCAL="${QUAD_OVERLAP_LOCAL:-1}"
QUAD_LOCAL_DIRECT_MAX="${QUAD_LOCAL_DIRECT_MAX:-0}"; QUAD_SPARSE_DESC_MLP="${QUAD_SPARSE_DESC_MLP:-0}"
WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_prectx_warpcoop_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || {
  echo 'correctness gate is fixed to n=21 mod 4294967291 residue 998035516' >&2; exit 2;
}
for x in SPARSE64 PAIR_MLP CPASYNC_PAIR QUAD_MLP QUAD_OVERLAP_LOCAL QUAD_SPARSE_DESC_MLP WINDOW4 PM_ACCUM PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
case "$CHUNK" in 2|4|8|16|32) ;; *) echo 'ORBITCTA_FLAT_CHUNK must be 2,4,8,16,32' >&2; exit 2;; esac
[[ "$COL_ILP" == 4 ]] || { echo 'warpcoop quad A/B requires ORBITCTA_COL_ILP=4' >&2; exit 2; }
[[ "$PAIR_MLP" == 1 && "$WINDOW4" == 1 ]] || { echo 'warpcoop A/B requires PAIR_MLP=1 WINDOW4=1' >&2; exit 2; }
[[ "$QUAD_MLP" == 1 ]] || { echo 'warpcoop A/B currently fixes QUAD_MLP=1' >&2; exit 2; }
[[ "$CPASYNC_PAIR" == 1 ]] || { echo 'warpcoop A/B currently fixes CPASYNC_PAIR=1' >&2; exit 2; }
[[ "$QUAD_OVERLAP_LOCAL" == 1 ]] || { echo 'warpcoop A/B currently fixes QUAD_OVERLAP_LOCAL=1' >&2; exit 2; }
(( QUAD_LOCAL_DIRECT_MAX == 0 )) || { echo 'QUAD_OVERLAP_LOCAL requires QUAD_LOCAL_DIRECT_MAX=0' >&2; exit 2; }
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
  >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async remote-peer preflight failed' >&2; exit 5; }
ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" \
  >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"

COMMON=(N="$N" ARCH="$ARCH" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK="$CHUNK" ORBITCTA_COL_ILP="$COL_ILP"
  DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0
  RANKFORMULA_MLP_WINDOW4="$WINDOW4" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR"
  QUAD_MLP="$QUAD_MLP" QUAD_OVERLAP_LOCAL="$QUAD_OVERLAP_LOCAL" QUAD_LOCAL_DIRECT_MAX="$QUAD_LOCAL_DIRECT_MAX"
  QUAD_SPARSE_DESC_MLP="$QUAD_SPARSE_DESC_MLP" PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1
  PRECTX_FLAT_BID=0 PRECTX_FLAT_BID_FUSED=0 PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE="$PTXAS_VERBOSE")

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true
  sleep "$SAMPLE_INTERVAL"
done; }

printf 'variant\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

build_one(){
  local coop="$1" name="$2" bin="$ONEESAN_BUILD_DIR/b300_orbitcta_flat_prectx_${name}_n${N}"
  env "${COMMON[@]}" PRECTX_WARPCOOP="$coop" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/${name}.build.out" 2>"$LOGDIR/${name}.build.err"
  grep -q "prectx_warpcoop=$coop" "$LOGDIR/${name}.build.err" || { echo "$name build marker mismatch" >&2; exit 6; }
  python3 "$PARSER" "$LOGDIR/${name}.build.err" --label "$name" >>"$RESOURCE" || true
  printf '%s' "$bin"
}
run_one(){
  local name="$1" bin="$2" rep="$3" so="$LOGDIR/${name}_r${rep}.out" se="$LOGDIR/${name}_r${rep}.err" util="$LOGDIR/${name}_r${rep}.util"
  runenv=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
  [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
  [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
  env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
  (( rc == 0 )) || { echo "$name repeat=$rep failed rc=$rc" >&2; exit "$rc"; }
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$name missing residue" >&2; exit 3; }
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
  read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mg" "$mm" \
    "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" >>"$RESULT"
}

serial_bin="$(build_one 0 serial)"; coop_bin="$(build_one 1 warpcoop)"
for ((r=1;r<=REPEATS;++r)); do run_one serial "$serial_bin" "$r"; done
for ((r=1;r<=REPEATS;++r)); do run_one warpcoop "$coop_bin" "$r"; done

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL residue mismatch '+repr(sorted(res)))
keys=('wall_s','forward_high_s','reverse_high_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
out=[]
for variant in ('serial','warpcoop'):
 g=[r for r in rows if r['variant']==variant]
 z={'variant':variant,'repeats':len(g),'residue':g[0]['residue']}
 for k in keys:z[k]=statistics.median(float(r[k]) for r in g if r[k]!='NA')
 for k in ('forward_regs','reverse_regs','forward_blocks_per_sm','reverse_blocks_per_sm'):
  z[k]=int(statistics.median(int(r[k]) for r in g if r[k]))
 out.append(z)
cols=('variant','repeats','residue')+keys+('forward_regs','reverse_regs','forward_blocks_per_sm','reverse_blocks_per_sm')
with open(sys.argv[2],'w') as f:
 f.write('\t'.join(cols)+'\n')
 for z in out:f.write('\t'.join(str(z[k]) for k in cols)+'\n')
q={z['variant']:z for z in out}
print(f'warpcoop_wall_speedup={q["serial"]["wall_s"]/q["warpcoop"]["wall_s"]:.6f}x')
print(f'warpcoop_high_speedup={q["serial"]["high_s"]/q["warpcoop"]["high_s"]:.6f}x')
print(f'warpcoop_memctrl_delta={q["warpcoop"]["avg_memctrl_util_pct"]-q["serial"]["avg_memctrl_util_pct"]:.6f}pp')
print(f'warpcoop_regs={q["warpcoop"]["forward_regs"]}/{q["warpcoop"]["reverse_regs"]} serial_regs={q["serial"]["forward_regs"]}/{q["serial"]["reverse_regs"]}')
PY
cat "$RESULT"
echo "warpcoop prectx A/B OK result=$RESULT summary=$SUMMARY chunk=$CHUNK col_ilp=$COL_ILP" >&2
