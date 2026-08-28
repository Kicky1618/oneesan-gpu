#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"
NGPU="${NGPU:-2}"
BLOCKS="${BLOCKS:-256}"
MOD="${MOD:-4294967291}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-1}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 ||
      BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "compact primitive runtime A/B supports even W=8..10" >&2
  exit 2
fi
if [[ "$MOD" != 4294967291 ]]; then
  echo "compact primitive runtime A/B requires MOD=4294967291" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_primitive_sym_u32_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PREINCLUDE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_primitive_sym_u32_preinclude.cuh"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
[[ -f "$PREINCLUDE" ]] || { echo "missing preinclude: $PREINCLUDE" >&2; exit 2; }

ARCH="$ARCH" \
LOG="$LOGDIR/nvcc-prepend-smoke.log" \
OUT="$ONEESAN_BUILD_DIR/gridfp_primitive_sym_nvcc_prepend_smoke_should_not_exist" \
bash "$ONEESAN_ROOT/scripts/bench/gridfp-build-nvcc-prepend-smoke.sh" \
  >"$LOGDIR/nvcc-prepend-smoke.out" 2>"$LOGDIR/nvcc-prepend-smoke.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-primitive-sym-u32-table-proof.sh" \
  >"$LOGDIR/primitive-proof.out" 2>"$LOGDIR/primitive-proof.err"

build_one() {
  local fast="$1" bin="$2"
  local inject="-include $PREINCLUDE -DRP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE=$fast -DRP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT=0 -DRP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_LAST_R=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED=0"
  NVCC_PREPEND_FLAGS="$inject" \
  MODE=two-row-runtime-multigpu \
  RUNTIME_CACHE_EDGES=1 \
  RUNTIME_FAST_P32M5_MOD=1 \
  RUNTIME_POLL_GLOBAL_ERROR=0 \
  RUNTIME_PACK_SHARED_KEYS=1 \
  RUNTIME_FAST_DIV64=1 \
  RUNTIME_PRIMITIVE_RANK_SETBITS=1 \
  RUNTIME_MATERIALIZE_PRIMITIVE_SETBITS=1 \
  RUNTIME_SUPPORT_UNRANK_EARLY_EXIT=1 \
  RUNTIME_BROADWORD_SUPPORT=1 \
  RUNTIME_OWNER_FROM_BOUNDARIES=1 \
  RUNTIME_OWNER_RECIPROCAL=1 \
  RUNTIME_OWNER_FIXED54=0 \
  RUNTIME_OWNER_FIXED52=1 \
  RUNTIME_OWNER_U32LIMB=0 \
  RUNTIME_OWNER_W28_NGPU8_DIRECT=0 \
  RUNTIME_OWNER_SUPPORT_BITPACK=1 \
  RUNTIME_OWNER_PREFIX_BINARY=1 \
  RUNTIME_OWNER_LOCAL_SECTOR_TABLE=1 \
  RUNTIME_OWNER_LOCAL_SECTOR_PARITY=1 \
  RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE=0 \
  RUNTIME_TURN_LOCAL_SECTOR_TABLE=1 \
  RUNTIME_TURN_LOCAL_SECTOR_W28_TREE=0 \
  RUNTIME_TURN_DISCOVERY_NONN_SCAN=0 \
  RUNTIME_SUPPORT_RANK_SETBITS=1 \
  RUNTIME_FUSE_PRIMITIVE_SUPPORT_RANK=1 \
  RUNTIME_DIRECT_BLOCKED_RANK=1 \
  RUNTIME_SECTOR_OFFSET_TABLE=1 \
  RUNTIME_CACHE_SECTOR_ROW_BASE=1 \
  RUNTIME_OUTER_GROUP_TABLE=1 \
  RUNTIME_FAST_OUTSIDE_COMPACT=1 \
  RUNTIME_FAST_ERASE_TWO_BITS=1 \
  RUNTIME_FAST_DISCOVERY_VALIDITY=1 \
  RUNTIME_DISCOVERY_ENDPOINT_SCAN=1 \
  RUNTIME_FAST_CLOSURE_NONN_SCAN=1 \
  RUNTIME_FIND_RECENT_FIRST=0 \
  RUNTIME_FIND_SIGNATURE_FILTER=0 \
  RUNTIME_FIND_INDEX_CACHE=0 \
  RUNTIME_FIND_INDEX_BUCKETS=64 \
  RUNTIME_FIND_INDEX_WAYS=1 \
  RUNTIME_FAST_MIRROR_MATE=1 \
  RUNTIME_FAST_INCLUDE_HORIZONTAL_REVERSE=1 \
  RUNTIME_FAST_BLOCKED_EXCLUDE_REVERSE=1 \
  RUNTIME_DIRECT_REVERSE_SMALL_STEP=1 \
  ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
  bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
    >"$LOGDIR/fast${fast}.build.out" 2>"$LOGDIR/fast${fast}.build.err"
  grep -Fq "nvcc_prepend=$inject)" "$LOGDIR/fast${fast}.build.out" || {
    echo "primitive_sym_u32=$fast build did not report expected prepend flags" >&2
    cat "$LOGDIR/fast${fast}.build.out" >&2 || true
    exit 6
  }
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_primitive_sym_u32_0_w${W}_g${NGPU}_b${BLOCKS}"
BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_primitive_sym_u32_1_w${W}_g${NGPU}_b${BLOCKS}"
build_one 0 "$BIN0"
build_one 1 "$BIN1"

run_one() {
  local fast="$1" bin="$2" rep="$3" phase="$4"
  local out="$LOGDIR/fast${fast}_${phase}${rep}.out"
  local err="$LOGDIR/fast${fast}_${phase}${rep}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || {
    echo "primitive_sym_u32=$fast $phase$rep missing runtime result" >&2
    cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3
  }
  grep -Fq ' exact=OK' <<<"$line" || {
    echo "primitive_sym_u32=$fast $phase$rep failed exactness" >&2
    echo "$line" >&2; exit 4
  }
  if [[ "$phase" == run ]]; then
    local wall
    wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || exit 5
    printf '%s\t%s\t%s\n' "$fast" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r=1; r<=WARMUP; ++r)); do
  run_one 0 "$BIN0" "$r" warmup
  run_one 1 "$BIN1" "$r" warmup
done
printf 'primitive_sym_u32\trepeat\twall_ms\n' >"$RESULT"
for ((r=1; r<=REPEATS; ++r)); do
  if ((r & 1)); then order=(0 1); else order=(1 0); fi
  for fast in "${order[@]}"; do
    [[ "$fast" == 0 ]] && bin="$BIN0" || bin="$BIN1"
    echo "=== runtime primitive-sym-u32=$fast run $r/$REPEATS ===" >&2
    run_one "$fast" "$bin" "$r" run
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
out=[]
for mode in ('0','1'):
    xs=[float(r['wall_ms']) for r in rows if r['primitive_sym_u32']==mode]
    if not xs: raise SystemExit(f'missing primitive_sym_u32={mode}')
    out.append({'primitive_sym_u32':mode,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['primitive_sym_u32']:r for r in out}
old=float(q['0']['wall_ms_median']); new=float(q['1']['wall_ms_median'])
print(f'runtime_primitive_sym_u32_wall_speedup={old/new:.6f}x')
print(f'runtime_primitive_sym_u32_wall_delta_pct={(new/old-1)*100:.4f}%')
print('runtime_primitive_sym_u32_old_table_bytes=6960')
print('runtime_primitive_sym_u32_new_table_bytes=900')
print('runtime_primitive_sym_u32_saved_bytes=6060')
print('runtime_primitive_sym_u32_nvcc_prepend_smoke=1')
print('runtime_primitive_sym_u32_signature_filter=0')
print('runtime_primitive_sym_u32_index_cache=0')
print('runtime_primitive_sym_u32_exact=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas primitive baseline ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fast0.build.err" >&2 || true
  echo '--- ptxas primitive compact u32 proxy ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fast1.build.err" >&2 || true
fi

echo "gridfp-reduced-runtime-primitive-sym-u32-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
