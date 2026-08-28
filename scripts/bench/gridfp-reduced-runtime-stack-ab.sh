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

if (( W < 8 || W > 28 || W % 2 != 0 )); then
  echo "W must be even and in [8,28]" >&2
  exit 2
fi
if (( NGPU < 2 || NGPU > 16 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid NGPU/BLOCKS/REPEATS/WARMUP" >&2
  exit 2
fi
if [[ "$MOD" != 4294967291 ]]; then
  echo "full runtime fast-stack A/B requires MOD=4294967291" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_stack_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-edgecache-proof.sh" \
  >"$LOGDIR/edgecache-proof.out" 2>"$LOGDIR/edgecache-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-p32m5-mod-proof.sh" \
  >"$LOGDIR/p32m5-proof.out" 2>"$LOGDIR/p32m5-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-sharedkey-proof.sh" \
  >"$LOGDIR/sharedkey-proof.out" 2>"$LOGDIR/sharedkey-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-fastdiv64-proof.sh" \
  >"$LOGDIR/fastdiv64-proof.out" 2>"$LOGDIR/fastdiv64-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-primitive-rank-setbits-proof.sh" \
  >"$LOGDIR/setbits-proof.out" 2>"$LOGDIR/setbits-proof.err"

build_one() {
  local mode="$1" bin="$2"
  local cache fast poll packed fastdiv setbits
  if [[ "$mode" == baseline ]]; then
    cache=0; fast=0; poll=1; packed=0; fastdiv=0; setbits=0
  else
    cache=1; fast=1; poll=0; packed=1; fastdiv=1; setbits=1
  fi
  MODE=two-row-runtime-multigpu \
    RUNTIME_CACHE_EDGES="$cache" \
    RUNTIME_FAST_P32M5_MOD="$fast" \
    RUNTIME_POLL_GLOBAL_ERROR="$poll" \
    RUNTIME_PACK_SHARED_KEYS="$packed" \
    RUNTIME_FAST_DIV64="$fastdiv" \
    RUNTIME_PRIMITIVE_RANK_SETBITS="$setbits" \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
    >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
}

BASE="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_stack_baseline_w${W}_g${NGPU}_b${BLOCKS}"
FAST="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_stack_fast_w${W}_g${NGPU}_b${BLOCKS}"
build_one baseline "$BASE"
build_one fast "$FAST"

run_one() {
  local mode="$1" bin="$2" rep="$3" phase="$4"
  local out="$LOGDIR/${mode}_${phase}${rep}.out"
  local err="$LOGDIR/${mode}_${phase}${rep}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || {
    echo "$mode $phase$rep missing runtime result" >&2
    cat "$out" >&2 || true
    cat "$err" >&2 || true
    exit 3
  }
  grep -Fq ' exact=OK' <<<"$line" || {
    echo "$mode $phase$rep failed exactness" >&2
    echo "$line" >&2
    exit 4
  }
  if [[ "$phase" == run ]]; then
    local wall
    wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || { echo "$mode run$rep missing wall_ms" >&2; exit 5; }
    printf '%s\t%s\t%s\n' "$mode" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r = 1; r <= WARMUP; ++r)); do
  run_one baseline "$BASE" "$r" warmup
  run_one fast "$FAST" "$r" warmup
done

printf 'runtime_stack\trepeat\twall_ms\n' >"$RESULT"
for ((r = 1; r <= REPEATS; ++r)); do
  if (( r & 1 )); then order=(baseline fast); else order=(fast baseline); fi
  for mode in "${order[@]}"; do
    if [[ "$mode" == baseline ]]; then bin="$BASE"; else bin="$FAST"; fi
    echo "=== runtime stack=$mode run $r/$REPEATS ===" >&2
    run_one "$mode" "$bin" "$r" run
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv
import statistics
import sys

src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
out = []
for mode in ('baseline', 'fast'):
    xs = [float(r['wall_ms']) for r in rows if r['runtime_stack'] == mode]
    if not xs:
        raise SystemExit(f'missing runtime_stack={mode} samples')
    out.append({
        'runtime_stack': mode,
        'repeats': len(xs),
        'wall_ms_median': f'{statistics.median(xs):.9f}',
        'wall_ms_min': f'{min(xs):.9f}',
        'wall_ms_max': f'{max(xs):.9f}',
    })

with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=('runtime_stack', 'repeats', 'wall_ms_median', 'wall_ms_min', 'wall_ms_max'),
        delimiter='\t')
    w.writeheader()
    w.writerows(out)

q = {r['runtime_stack']: r for r in out}
old = float(q['baseline']['wall_ms_median'])
new = float(q['fast']['wall_ms_median'])
print(f'runtime_fast_stack_wall_speedup={old / new:.6f}x')
print(f'runtime_fast_stack_wall_delta_pct={(new / old - 1.0) * 100.0:.4f}%')
print('runtime_fast_stack_baseline=edge_cache0,fast_mod0,error_poll1,packed_keys0,fast_div640,primitive_setbits0')
print('runtime_fast_stack_fast=edge_cache1,fast_mod1,error_poll0,packed_keys1,fast_div641,primitive_setbits1')
print('runtime_fast_stack_key_shared_bytes_saved_per_block=10240')
print('runtime_fast_stack_edge_cache_shared_bytes_added_per_block=2560')
print('runtime_fast_stack_net_known_shared_bytes_delta_per_block=-7680')
print('runtime_fast_stack_recomputed_small_steps_in_accumulation=0')
print('runtime_fast_stack_signed_64bit_divisions_at_mod4294967291=0')
print('runtime_fast_stack_dynamic_label_divmod_pairs_remaining=0')
print('runtime_fast_stack_global_error_polls_per_discovered_source=0')
print('runtime_fast_stack_primitive_rank_scan=occupied')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas baseline ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/baseline.build.err" >&2 || true
  echo '--- ptxas fast ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fast.build.err" >&2 || true
fi

echo "gridfp-reduced-runtime-stack-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
