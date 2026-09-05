#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-3}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"; RUN_PTXAS="${RUN_PTXAS:-1}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
for x in PM_ACCUM TERNARY_KEY4 RUN_SELFTEST RUN_PTXAS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) exit 2;; esac
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankformula_nometa_coop_block_ab_n${N}_${TRANSPOSE_MODE}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-blocks-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-coopgroup-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lazy-load-proof.sh"
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){ local block="$1" bin="$2" label="block${block}"; N="$N" ARCH="$ARCH" OUT="$bin" RANKFORMULA_NOMETA_BLOCK="$block" RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"; }
run_one(){ local block="$1" bin="$2" rep="$3" label="block${block}" so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err"; BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; local line detail residue wall fh rh; line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3; residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }; wall="$(field wall_s "$line")"; detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" >>"$RESULT"; }

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
[[ "$RUN_PTXAS" == 1 ]] && printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for block in 8 16; do
  label="block${block}"
  if [[ "$RUN_SELFTEST" == 1 ]]; then RANKFORMULA_NOMETA_BLOCK="$block" RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-block-selftest.sh" >"$LOGDIR/$label.selftest.out" 2>"$LOGDIR/$label.selftest.err"; fi
  bin="$ONEESAN_BUILD_DIR/ab_rankformula_nometa_coop_${label}_n${N}"; build_one "$block" "$bin"
  [[ "$RUN_PTXAS" == 1 ]] && python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"
  for ((r=1;r<=REPEATS;++r)); do run_one "$block" "$bin" "$r"; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" <<'PY'
import csv,statistics,sys
src,dst,resource,run_ptxas=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('block8','block16'):
 g=[r for r in rows if r['mode']==mode]; z={'mode':mode,'repeats':len(g)}
 for k in ('wall_s','forward_high_s','reverse_high_s'):
  xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
 z['total_high_s']=(z['forward_high_s']+z['reverse_high_s']) if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None; out.append(z)
with open(dst,'w') as f:
 f.write('mode\trepeats\twall_s\tforward_high_s\treverse_high_s\ttotal_high_s\n')
 for z in out:f.write('\t'.join(str(z[k]) for k in ('mode','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s'))+'\n')
q={z['mode']:z for z in out}
for k in ('wall_s','forward_high_s','reverse_high_s','total_high_s'):
 if q['block8'][k] and q['block16'][k]:print(f'block16_{k}_speedup={q["block8"][k]/q["block16"][k]:.6f}x')
if run_ptxas=='1':
 rr=list(csv.DictReader(open(resource),delimiter='\t'))
 for mode in ('block8','block16'):
  g=[r for r in rr if r['backend']==mode and 'high' in r['kernel'].lower()]; regs=[int(r['registers']) for r in g if r['registers']!='NA']; ss=[int(r['spill_store_bytes']) for r in g if r['spill_store_bytes']!='NA']; sl=[int(r['spill_load_bytes']) for r in g if r['spill_load_bytes']!='NA']; print(f'{mode}_high_max_registers={max(regs) if regs else "NA"}'); print(f'{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}'); print(f'{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
print('block8_aux_bytes_model=857642'); print('block16_aux_bytes_model=707406'); print('block8_table_loads_model=361431'); print('block16_table_loads_model=215509'); print('block8_avg_early_ballots=1.921043'); print('block16_avg_early_ballots=2.251382'); print('coopgroup=1 warpshare=1 group64_selfindexed=1')
PY
echo "b300-depthcode-rankformula-nometa-coop-block-ab OK n=$N repeats=$REPEATS result=$RESULT" >&2
