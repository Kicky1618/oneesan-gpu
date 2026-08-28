#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Bounded comparison of the three structurally different sparse HIGH paths.
# Keep the matrix intentionally small: 3 candidates x 3 repeats by default.
N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-3}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"; RUN_PTXAS="${RUN_PTXAS:-1}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-1}"
RANKDELTA8_FUSED13="${RANKDELTA8_FUSED13:-1}"

if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516
  else echo "EXPECT must be set when N/MOD differ" >&2; exit 2; fi
fi
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
for x in PM_ACCUM TERNARY_KEY4 RUN_SELFTEST RUN_PTXAS RANKCHUNK32_FUSED16 RANKDELTA8_FUSED13; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then
  echo "invalid launch/sweep parameters" >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2; exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_sparse_high_candidates_n${N}_${TRANSPOSE_MODE}_${RANKSTREAM_LUT_LOAD}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-align32-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankdelta8-plan.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-plan.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-sparse-base-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-base-delta-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-inline-cross-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-slotmeta-proof.sh"

if [[ "$RUN_SELFTEST" == 1 ]]; then
  ARCH="$ARCH" W=10 DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKCHUNK32_ONESHFL=1 RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" RANKCHUNK32_BYTEPACK=0 RANKCHUNK32_ALIGN32=1 RANKCHUNK32_BLOCK64=0 \
    PM_ACCUM="$PM_ACCUM" RUN_LAYOUT_PROOF=0 \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh" \
    >"$LOGDIR/rankchunk32.selftest.out" 2>"$LOGDIR/rankchunk32.selftest.err"

  ARCH="$ARCH" W=10 DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKDELTA8_ALIGN32=1 RANKDELTA8_FUSED13="$RANKDELTA8_FUSED13" PM_ACCUM="$PM_ACCUM" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankdelta8-cross5-selftest.sh" \
    >"$LOGDIR/rankdelta8.selftest.out" 2>"$LOGDIR/rankdelta8.selftest.err"

  ARCH="$ARCH" W=10 DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKDELTA8_FUSED13="$RANKDELTA8_FUSED13" PM_ACCUM="$PM_ACCUM" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-slotmeta-selftest.sh" \
    >"$LOGDIR/rankformula.selftest.out" 2>"$LOGDIR/rankformula.selftest.err"
fi

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local mode="$1" bin="$2" highctx
  case "$mode" in
    rankchunk32) highctx=warpstriped_delta_direct_affine_rankchunk32_cross5 ;;
    rankdelta8) highctx=warpstriped_delta_direct_affine_rankdelta8_cross5 ;;
    rankformula) highctx=warpstriped_delta_direct_affine_rankformula_cross5 ;;
    *) exit 2 ;;
  esac
  N="$N" ARCH="$ARCH" OUT="$bin" HIGH_CTX="$highctx" \
    DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKCHUNK32_ONESHFL=1 RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" RANKCHUNK32_BYTEPACK=0 RANKCHUNK32_ALIGN32=1 RANKCHUNK32_BLOCK64=0 \
    RANKDELTA8_ALIGN32=1 RANKDELTA8_FUSED13="$RANKDELTA8_FUSED13" \
    RANKFORMULA_SLOTMETA=1 \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
}

run_one(){
  local mode="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
if [[ "$RUN_PTXAS" == 1 ]]; then
  printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
fi
for mode in rankchunk32 rankdelta8 rankformula; do
  bin="$ONEESAN_BUILD_DIR/sparse_high_candidate_${mode}_n${N}"
  echo "=== build $mode ===" >&2
  build_one "$mode" "$bin"
  [[ "$RUN_PTXAS" == 1 ]] && python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE"
  for ((r=1;r<=REPEATS;++r)); do
    echo "=== run $mode $r/$REPEATS ===" >&2
    run_one "$mode" "$bin" "$r"
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" <<'PY'
import csv,statistics,sys
src,dst,resource,run_ptxas=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
out=[]
for mode in ('rankchunk32','rankdelta8','rankformula'):
    g=[r for r in rows if r['mode']==mode]
    z={'mode':mode,'repeats':len(g)}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!='NA']
        z[m]=statistics.median(xs) if xs else None
    z['total_high_s']=(z['forward_high_s']+z['reverse_high_s']) if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None
    out.append(z)
with open(dst,'w') as f:
    f.write('mode\trepeats\twall_s\tforward_high_s\treverse_high_s\ttotal_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n')
    for z in out:
        f.write('\t'.join(str(z[k]) for k in ('mode','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','forward_low_s','reverse_low_s','transpose_s'))+'\n')
valid=[z for z in out if z['total_high_s'] is not None]
if valid:
    best=min(valid,key=lambda z:z['total_high_s'])
    print(f'best_total_high_backend={best["mode"]}')
    print(f'best_total_high_s={best["total_high_s"]:.9f}')
    for z in valid:
        print(f'{z["mode"]}_vs_best_total_high={z["total_high_s"]/best["total_high_s"]:.6f}x')
valid_wall=[z for z in out if z['wall_s'] is not None]
if valid_wall:
    best=min(valid_wall,key=lambda z:z['wall_s'])
    print(f'best_wall_backend={best["mode"]}')
    print(f'best_wall_s={best["wall_s"]:.9f}')
if run_ptxas=='1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for mode in ('rankchunk32','rankdelta8','rankformula'):
        g=[r for r in rr if r['backend']==mode and 'high' in r['kernel'].lower()]
        regs=[int(r['registers']) for r in g if r['registers']!='NA']
        ss=[int(r['spill_store_bytes']) for r in g if r['spill_store_bytes']!='NA']
        sl=[int(r['spill_load_bytes']) for r in g if r['spill_load_bytes']!='NA']
        print(f'{mode}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
print('rankchunk32_candidate=compact23_align32_oneshfl_fused16')
print('rankdelta8_candidate=align32_fused13')
print('rankformula_candidate=sparse_int16_delta_slot_lmask_inline_cross')
PY

echo "b300-depthcode-sparse-high-candidates OK n=$N repeats=$REPEATS result=$RESULT" >&2
