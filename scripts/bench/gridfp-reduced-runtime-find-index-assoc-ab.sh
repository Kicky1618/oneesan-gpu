#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"
MOD="${MOD:-4294967291}"; REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"
ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"; STORAGE="${STORAGE:-64}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "index-assoc full-runtime A/B supports even W=8..10" >&2; exit 2
fi
if [[ "$STORAGE" != 64 ]]; then echo "index-assoc A/B currently fixes STORAGE=64 for equal 4096 B/block shared use" >&2; exit 2; fi
if [[ "$MOD" != 4294967291 ]]; then echo "index-assoc A/B requires MOD=4294967291" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_find_index_assoc_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-find-index-cache-proof.sh" >"$LOGDIR/index-proof.out" 2>"$LOGDIR/index-proof.err"
MAX_W=10 bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-find-index-cache-model.sh" >"$LOGDIR/index-model.out" 2>"$LOGDIR/index-model.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-shared-budget-proof.sh" >"$LOGDIR/shared-budget-proof.out" 2>"$LOGDIR/shared-budget-proof.err"

build_one() {
  local ways="$1" bin="$2"
  MODE=two-row-runtime-multigpu \
  RUNTIME_CACHE_EDGES=1 RUNTIME_FAST_P32M5_MOD=1 RUNTIME_POLL_GLOBAL_ERROR=0 \
  RUNTIME_PACK_SHARED_KEYS=1 RUNTIME_FAST_DIV64=1 \
  RUNTIME_PRIMITIVE_RANK_SETBITS=1 RUNTIME_MATERIALIZE_PRIMITIVE_SETBITS=1 \
  RUNTIME_SUPPORT_UNRANK_EARLY_EXIT=1 RUNTIME_BROADWORD_SUPPORT=1 \
  RUNTIME_OWNER_FROM_BOUNDARIES=1 RUNTIME_OWNER_RECIPROCAL=1 \
  RUNTIME_OWNER_FIXED54=0 RUNTIME_OWNER_FIXED52=1 RUNTIME_OWNER_U32LIMB=0 \
  RUNTIME_OWNER_W28_NGPU8_DIRECT=0 RUNTIME_OWNER_SUPPORT_BITPACK=1 \
  RUNTIME_OWNER_PREFIX_BINARY=1 RUNTIME_OWNER_LOCAL_SECTOR_TABLE=1 \
  RUNTIME_OWNER_LOCAL_SECTOR_PARITY=1 RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE=0 \
  RUNTIME_TURN_LOCAL_SECTOR_TABLE=1 RUNTIME_TURN_LOCAL_SECTOR_W28_TREE=0 \
  RUNTIME_TURN_DISCOVERY_NONN_SCAN=0 RUNTIME_SUPPORT_RANK_SETBITS=1 \
  RUNTIME_FUSE_PRIMITIVE_SUPPORT_RANK=1 RUNTIME_DIRECT_BLOCKED_RANK=1 \
  RUNTIME_SECTOR_OFFSET_TABLE=1 RUNTIME_CACHE_SECTOR_ROW_BASE=1 \
  RUNTIME_OUTER_GROUP_TABLE=1 RUNTIME_FAST_OUTSIDE_COMPACT=1 \
  RUNTIME_FAST_ERASE_TWO_BITS=1 RUNTIME_FAST_DISCOVERY_VALIDITY=1 \
  RUNTIME_DISCOVERY_ENDPOINT_SCAN=1 RUNTIME_FAST_CLOSURE_NONN_SCAN=1 \
  RUNTIME_FIND_RECENT_FIRST=1 RUNTIME_FIND_SIGNATURE_FILTER=0 \
  RUNTIME_FIND_INDEX_CACHE=1 RUNTIME_FIND_INDEX_BUCKETS="$STORAGE" \
  RUNTIME_FIND_INDEX_WAYS="$ways" RUNTIME_FAST_MIRROR_MATE=1 \
  RUNTIME_FAST_INCLUDE_HORIZONTAL_REVERSE=1 RUNTIME_FAST_BLOCKED_EXCLUDE_REVERSE=1 \
  RUNTIME_DIRECT_REVERSE_SMALL_STEP=1 ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
  >"$LOGDIR/w${ways}.build.out" 2>"$LOGDIR/w${ways}.build.err"
}
for ways in 1 2 4; do
  eval "BIN${ways}=\"$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_find_index_assoc_w${ways}_w${W}_g${NGPU}_b${BLOCKS}\""
  eval "bin=\$BIN${ways}"; build_one "$ways" "$bin"
done

run_one() {
  local ways="$1" bin="$2" rep="$3" phase="$4"
  local out="$LOGDIR/w${ways}_${phase}${rep}.out" err="$LOGDIR/w${ways}_${phase}${rep}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line; line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "ways=$ways $phase$rep missing runtime result" >&2; cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' exact=OK' <<<"$line" || { echo "ways=$ways $phase$rep failed exactness" >&2; echo "$line" >&2; exit 4; }
  if [[ "$phase" == run ]]; then
    local wall; wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$wall" ]] || exit 5
    printf '%s\t%s\t%s\n' "$ways" "$rep" "$wall" >>"$RESULT"
  fi
}
for ((r=1;r<=WARMUP;++r)); do for ways in 1 2 4; do eval "bin=\$BIN${ways}"; run_one "$ways" "$bin" "$r" warmup; done; done
printf 'index_ways\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do
  case $((r%3)) in 1) order=(1 2 4);; 2) order=(4 1 2);; 0) order=(2 4 1);; esac
  for ways in "${order[@]}"; do eval "bin=\$BIN${ways}"; echo "=== runtime index-ways=$ways run $r/$REPEATS ===" >&2; run_one "$ways" "$bin" "$r" run; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('1','2','4'):
    xs=[float(r['wall_ms']) for r in rows if r['index_ways']==mode]
    if not xs: raise SystemExit(f'missing index_ways={mode}')
    out.append({'index_ways':mode,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('index_ways','repeats','wall_ms_median','wall_ms_min','wall_ms_max'),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['index_ways']:float(r['wall_ms_median']) for r in out}; best=min(q,key=q.get)
print(f'runtime_find_index_assoc_best_ways={best}')
print(f'runtime_find_index_assoc_2way_vs_1way_speedup={q["1"]/q["2"]:.6f}x')
print(f'runtime_find_index_assoc_4way_vs_2way_speedup={q["2"]/q["4"]:.6f}x')
print('runtime_find_index_assoc_storage_bytes_per_set=64')
print('runtime_find_index_assoc_shared_bytes_per_block=4096')
print('runtime_find_index_assoc_total_known_shared_bytes=20480')
print('runtime_find_index_assoc_net_vs_unpacked_baseline=-3584')
print('runtime_find_index_assoc_1way_hash_buckets=64')
print('runtime_find_index_assoc_2way_hash_buckets=32')
print('runtime_find_index_assoc_4way_hash_buckets=16')
print('runtime_find_index_assoc_overflow_state=packed_high_bit')
print('runtime_find_index_assoc_extra_overflow_registers=0')
print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  for ways in 1 2 4; do echo "--- ptxas ways=$ways ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/w${ways}.build.err" >&2 || true; done
fi
echo "gridfp-reduced-runtime-find-index-assoc-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
