#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"; REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 28 || W % 2 != 0 || NGPU < 2 || NGPU > 16 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "invalid W/NGPU/BLOCKS/REPEATS/WARMUP" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_bitcompact_ab_w${W}_g${NGPU}_b${BLOCKS}}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-bitcompact-proof.sh" >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"
build_one(){ local fastbits="$1" bin="$2"; MODE=two-row-runtime-multigpu RUNTIME_CACHE_EDGES=1 RUNTIME_FAST_P32M5_MOD=1 RUNTIME_POLL_GLOBAL_ERROR=0 RUNTIME_PACK_SHARED_KEYS=1 RUNTIME_FAST_DIV64=1 RUNTIME_PRIMITIVE_RANK_SETBITS=1 RUNTIME_BROADWORD_SUPPORT=1 RUNTIME_OWNER_FROM_BOUNDARIES=1 RUNTIME_SUPPORT_RANK_SETBITS=1 RUNTIME_SECTOR_OFFSET_TABLE=1 RUNTIME_OUTER_GROUP_TABLE=1 RUNTIME_FAST_OUTSIDE_COMPACT="$fastbits" RUNTIME_FAST_ERASE_TWO_BITS="$fastbits" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/fastbits${fastbits}.build.out" 2>"$LOGDIR/fastbits${fastbits}.build.err"; }
BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_bitcompact0_w${W}_g${NGPU}_b${BLOCKS}"; BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_bitcompact1_w${W}_g${NGPU}_b${BLOCKS}"; build_one 0 "$BIN0"; build_one 1 "$BIN1"
run_one(){ local fastbits="$1" bin="$2" rep="$3" phase="$4"; local out="$LOGDIR/fastbits${fastbits}_${phase}${rep}.out" err="$LOGDIR/fastbits${fastbits}_${phase}${rep}.err"; "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"; local line; line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }; grep -Fq ' exact=OK' <<<"$line" || { echo "$line" >&2; exit 4; }; if [[ "$phase" == run ]]; then local wall; wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$wall" ]] || exit 5; printf '%s\t%s\t%s\n' "$fastbits" "$rep" "$wall" >>"$RESULT"; fi; }
for ((r=1;r<=WARMUP;++r)); do run_one 0 "$BIN0" "$r" warmup; run_one 1 "$BIN1" "$r" warmup; done
printf 'bitcompact\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do if ((r&1)); then order=(0 1); else order=(1 0); fi; for fastbits in "${order[@]}"; do [[ "$fastbits" == 0 ]] && bin="$BIN0" || bin="$BIN1"; echo "=== runtime bitcompact=$fastbits run $r/$REPEATS ===" >&2; run_one "$fastbits" "$bin" "$r" run; done; done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('0','1'):
 xs=[float(r['wall_ms']) for r in rows if r['bitcompact']==mode]
 if not xs: raise SystemExit(f'missing bitcompact={mode}')
 out.append({'bitcompact':mode,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=('bitcompact','repeats','wall_ms_median','wall_ms_min','wall_ms_max'),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['bitcompact']:r for r in out}; old=float(q['0']['wall_ms_median']); new=float(q['1']['wall_ms_median'])
print(f'runtime_bitcompact_wall_speedup={old/new:.6f}x'); print(f'runtime_bitcompact_wall_delta_pct={(new/old-1)*100:.4f}%'); print('runtime_bitcompact_outside_old_scan=W'); print('runtime_bitcompact_outside_new=mask_shift'); print('runtime_bitcompact_erase_old_scan=L'); print('runtime_bitcompact_erase_new=three_segment_mask_shift'); print('runtime_bitcompact_extra_shared_bytes=0'); print('runtime_bitcompact_extra_constant_bytes=0'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then echo '--- ptxas bit loops ---' >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fastbits0.build.err" >&2 || true; echo '--- ptxas bit compact ---' >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fastbits1.build.err" >&2 || true; fi
echo "gridfp-reduced-runtime-bitcompact-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
