#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"; REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "forward last main step A/B supports even W=8..10" >&2; exit 2; fi
if [[ "$MOD" != 4294967291 ]]; then echo "forward last main step A/B requires MOD=4294967291" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_forward_last_main_step_ab_w${W}_g${NGPU}_b${BLOCKS}}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-turn-direct-high-expand-step-proof.sh" >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"

build_one(){ local fast="$1" bin="$2"; MODE=two-row-runtime-multigpu RUNTIME_CACHE_EDGES=1 RUNTIME_FAST_P32M5_MOD=1 RUNTIME_POLL_GLOBAL_ERROR=0 RUNTIME_PACK_SHARED_KEYS=1 RUNTIME_FAST_DIV64=1 RUNTIME_PRIMITIVE_RANK_SETBITS=1 RUNTIME_MATERIALIZE_PRIMITIVE_SETBITS=1 RUNTIME_SUPPORT_UNRANK_EARLY_EXIT=1 RUNTIME_BROADWORD_SUPPORT=1 RUNTIME_OWNER_FROM_BOUNDARIES=1 RUNTIME_OWNER_RECIPROCAL=1 RUNTIME_OWNER_FIXED54=0 RUNTIME_OWNER_FIXED52=1 RUNTIME_OWNER_U32LIMB=0 RUNTIME_OWNER_W28_NGPU8_DIRECT=0 RUNTIME_OWNER_SUPPORT_BITPACK=1 RUNTIME_OWNER_PREFIX_BINARY=1 RUNTIME_OWNER_LOCAL_SECTOR_TABLE=1 RUNTIME_OWNER_LOCAL_SECTOR_PARITY=1 RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE=0 RUNTIME_TURN_LOCAL_SECTOR_TABLE=1 RUNTIME_TURN_LOCAL_SECTOR_W28_TREE=0 RUNTIME_TURN_DISCOVERY_NONN_SCAN=1 RUNTIME_TURN_DIRECT_COMPRESS_INVERSE=1 RUNTIME_TURN_DIRECT_HIGH_COMPRESS_STEP=1 RUNTIME_TURN_DIRECT_LOW_COMPRESS_STEP=1 RUNTIME_TURN_DIRECT_HIGH_COMPRESS_INVERSE=1 RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE=1 RUNTIME_SUPPORT_RANK_SETBITS=1 RUNTIME_FUSE_PRIMITIVE_SUPPORT_RANK=1 RUNTIME_DIRECT_BLOCKED_RANK=1 RUNTIME_SECTOR_OFFSET_TABLE=1 RUNTIME_CACHE_SECTOR_ROW_BASE=1 RUNTIME_OUTER_GROUP_TABLE=1 RUNTIME_FAST_OUTSIDE_COMPACT=1 RUNTIME_FAST_ERASE_TWO_BITS=1 RUNTIME_FAST_DISCOVERY_VALIDITY=1 RUNTIME_DISCOVERY_ENDPOINT_SCAN=1 RUNTIME_FAST_CLOSURE_NONN_SCAN=1 RUNTIME_FIND_RECENT_FIRST=1 RUNTIME_FIND_SIGNATURE_FILTER=0 RUNTIME_FIND_INDEX_CACHE=1 RUNTIME_FIND_INDEX_BUCKETS=64 RUNTIME_FIND_INDEX_WAYS=2 RUNTIME_FAST_MIRROR_MATE=1 RUNTIME_FAST_INCLUDE_HORIZONTAL_REVERSE=1 RUNTIME_FAST_BLOCKED_EXCLUDE_REVERSE=1 RUNTIME_DIRECT_REVERSE_SMALL_STEP=1 RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP=1 RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP="$fast" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/fast${fast}.build.out" 2>"$LOGDIR/fast${fast}.build.err"; }
BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_forward_last_generic_w${W}_g${NGPU}_b${BLOCKS}"; BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_forward_last_direct_w${W}_g${NGPU}_b${BLOCKS}"; build_one 0 "$BIN0"; build_one 1 "$BIN1"
run_one(){ local fast="$1" bin="$2" rep="$3" phase="$4" out="$LOGDIR/fast${fast}_${phase}${rep}.out" err="$LOGDIR/fast${fast}_${phase}${rep}.err"; "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"; local line; line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }; grep -Fq ' exact=OK' <<<"$line" || { echo "$line" >&2; exit 4; }; if [[ "$phase" == run ]]; then local wall; wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; printf '%s\t%s\t%s\n' "$fast" "$rep" "$wall" >>"$RESULT"; fi; }
for ((r=1;r<=WARMUP;++r)); do run_one 0 "$BIN0" "$r" warmup; run_one 1 "$BIN1" "$r" warmup; done
printf 'forward_last_main_step\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do if ((r&1)); then order=(0 1); else order=(1 0); fi; for fast in "${order[@]}"; do [[ "$fast" == 0 ]] && bin="$BIN0" || bin="$BIN1"; run_one "$fast" "$bin" "$r" run; done; done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for m in ('0','1'):
 xs=[float(r['wall_ms']) for r in rows if r['forward_last_main_step']==m]; out.append({'forward_last_main_step':m,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['forward_last_main_step']:float(r['wall_ms_median']) for r in out}; print(f'runtime_forward_last_main_step_wall_speedup={q["0"]/q["1"]:.6f}x'); print('runtime_forward_last_main_step_include_result_dispatch=0'); print('runtime_forward_last_main_step_generic_project_forward_calls=0'); print('runtime_forward_last_main_step_projection=direct_Wm2'); print('runtime_forward_last_main_step_extra_shared_bytes=0'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then for fast in 0 1; do echo "--- ptxas forward_last_main=$fast ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fast${fast}.build.err" >&2 || true; done; fi
echo "gridfp-reduced-runtime-forward-last-main-step-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
