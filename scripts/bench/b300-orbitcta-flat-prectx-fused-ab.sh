#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SPARSE64="${DIRECTGATHER_SPARSE64:-1}"
COL_ILP="${ORBITCTA_COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_flat_prectx_fused_ab_n${N}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n21/mod4294967291' >&2; exit 2; }
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
case "$COL_ILP" in 1|2|4) ;; *) exit 2;; esac
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2

COMMON=(N="$N" ARCH="$ARCH" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0 ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR=0 QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_FLAT_BID=1 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for fused in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_flat_prectx_fused${fused}_n${N}"
  env "${COMMON[@]}" PRECTX_FLAT_BID_FUSED="$fused" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/fused${fused}.build.out" 2>"$LOGDIR/fused${fused}.build.err"
  python3 "$PARSER" "$LOGDIR/fused${fused}.build.err" --label "fused${fused}" >>"$RESOURCE" || true
done
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'fused\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tforward_flat_blocks\treverse_flat_blocks\n' >"$RESULT"
for fused in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_flat_prectx_fused${fused}_n${N}"
  for ((rep=1;rep<=REPEATS;++rep)); do
    so="$LOGDIR/fused${fused}_r${rep}.out";se="$LOGDIR/fused${fused}_r${rep}.err"
    runenv=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
    [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
    [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
    env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so"|tail -n1||true)";[[ -n "$line" ]]||exit 3;residue="$(field residue "$line")";[[ "$residue" == "$EXPECT" ]]||exit 4
    d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)";g="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se"|head -n1||true)";o="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
    fh="$(field forward_high_s "$d")";rh="$(field reverse_high_s "$d")";high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(float(sys.argv[1])+float(sys.argv[2]))
PY
)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$fused" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$(field forward_regs "$o")" "$(field reverse_regs "$o")" "$(field forward_blocks_per_sm "$o")" "$(field reverse_blocks_per_sm "$o")" "$(field forward_flat_blocks "$g")" "$(field reverse_flat_blocks "$g")" >>"$RESULT"
  done
done
python3 - "$RESULT" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));z={}
for f in ('0','1'):
 g=[r for r in rows if r['fused']==f];z[f]={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','high_s','forward_high_s','reverse_high_s')};z[f]['fr']=int(g[0]['forward_regs'] or 0);z[f]['rr']=int(g[0]['reverse_regs'] or 0);z[f]['fb']=int(g[0]['forward_blocks_per_sm'] or 0);z[f]['rb']=int(g[0]['reverse_blocks_per_sm'] or 0)
w='1' if z['1']['wall_s']<z['0']['wall_s'] else '0'
print(f"PRECTX_FUSED wall_speedup={z['0']['wall_s']/z['1']['wall_s']:.6f} high_speedup={z['0']['high_s']/z['1']['high_s']:.6f} regs0={z['0']['fr']}/{z['0']['rr']} regs1={z['1']['fr']}/{z['1']['rr']} active0={z['0']['fb']}/{z['0']['rb']} active1={z['1']['fb']}/{z['1']['rb']} winner={w}")
with open(sys.argv[2],'w') as f:f.write('ORBITCTA_FLAT=1\nORBITCTA_FLAT_CHUNK=1\nPRECTX_FORWARD=1\nPRECTX_REVERSE=1\nPRECTX_COMPACT=1\nPRECTX_FLAT_BID=1\nPRECTX_FLAT_BID_FUSED='+w+'\n')
PY
cat "$RESULT"
echo "flat prectx fused A/B OK result=$RESULT ptxas=$RESOURCE winner_env=$WINNER_ENV" >&2
