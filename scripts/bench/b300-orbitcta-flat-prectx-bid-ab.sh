#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SPARSE64="${DIRECTGATHER_SPARSE64:-1}"
COL_ILP="${ORBITCTA_COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"
PM_ACCUM="${PM_ACCUM:-1}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_prectx_bid_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'default correctness gate is n=21/mod4294967291/residue998035516' >&2; exit 2; }
for x in SPARSE64 PAIR_MLP WINDOW4 PM_ACCUM CPASYNC_PAIR PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$COL_ILP" in 1|2|4) ;; *) echo 'ORBITCTA_COL_ILP must be 1,2,4' >&2; exit 2;; esac
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
if [[ "$PAIR_MLP" == 1 ]]; then [[ "$WINDOW4" == 1 ]] || exit 2; [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || exit 2; fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || exit 5
fi

COMMON=(N="$N" ARCH="$ARCH" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0 ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PTXAS_VERBOSE="$PTXAS_VERBOSE")
BASE_BIN="$ONEESAN_BUILD_DIR/b300_flat_prectx_bid0_n${N}"; CAND_BIN="$ONEESAN_BUILD_DIR/b300_flat_prectx_bid1_n${N}"
env "${COMMON[@]}" PRECTX_FLAT_BID=0 OUT="$BASE_BIN" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
env "${COMMON[@]}" PRECTX_FLAT_BID=1 OUT="$CAND_BIN" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/cand.build.out" 2>"$LOGDIR/cand.build.err"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$LOGDIR/base.build.err" --label base >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/cand.build.err" --label cand >>"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'mode\tprectx_flat_bid\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tforward_flat_blocks\treverse_flat_blocks\tflat_bid_mode\n' >"$RESULT"
run_one(){
  local mode="$1" flag="$2" bin="$3" rep="$4" so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err"
  runenv=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
  [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
  [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
  env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; grid="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se" | head -n1 || true)"; occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se" | head -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  mode_seen="$(field flat_bid_mode "$grid")"
  if [[ "$flag" == 0 ]]; then [[ "$mode_seen" == binary_search ]] || { echo "base flat_bid_mode=$mode_seen" >&2; exit 6; }; else [[ "$mode_seen" == compact_prectx ]] || { echo "cand flat_bid_mode=$mode_seen" >&2; exit 6; }; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$flag" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" "$(field forward_flat_blocks "$grid")" "$(field reverse_flat_blocks "$grid")" "$mode_seen" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_one base 0 "$BASE_BIN" "$r"; run_one cand 1 "$CAND_BIN" "$r"; done

python3 - "$RESULT" "$SUMMARY" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,summary,winner=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out={}
for mode in ('base','cand'):
 g=[r for r in rows if r['mode']==mode]
 out[mode]={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','forward_high_s','reverse_high_s','high_s')}
with open(summary,'w') as f:
 f.write('metric\tbase\tcand\tspeedup_base_over_cand\n')
 for k in ('wall_s','forward_high_s','reverse_high_s','high_s'): f.write(f'{k}\t{out["base"][k]}\t{out["cand"][k]}\t{out["base"][k]/out["cand"][k]}\n')
winner_flag=1 if out['cand']['wall_s'] < out['base']['wall_s'] else 0
with open(winner,'w') as f:
 f.write('ORBITCTA_FLAT=1\nORBITCTA_FLAT_CHUNK=1\nPRECTX_FORWARD=1\nPRECTX_REVERSE=1\nPRECTX_COMPACT=1\n')
 f.write(f'PRECTX_FLAT_BID={winner_flag}\n')
print(f"PRECTX_FLAT_BID wall_speedup={out['base']['wall_s']/out['cand']['wall_s']:.6f} high_speedup={out['base']['high_s']/out['cand']['high_s']:.6f} fh_speedup={out['base']['forward_high_s']/out['cand']['forward_high_s']:.6f} rh_speedup={out['base']['reverse_high_s']/out['cand']['reverse_high_s']:.6f} winner={winner_flag}")
PY
cat "$RESULT"
echo "flat prectx-bid A/B OK result=$RESULT summary=$SUMMARY winner_env=$WINNER_ENV" >&2
