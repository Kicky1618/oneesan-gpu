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
if (( MOD < 3 )); then
  echo "MOD must be >=3" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_sharedkey_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-sharedkey-proof.sh" \
  >"$LOGDIR/sharedkey-proof.out" 2>"$LOGDIR/sharedkey-proof.err"

build_one() {
  local packed="$1" bin="$2"
  MODE=two-row-runtime-multigpu \
    RUNTIME_CACHE_EDGES=1 \
    RUNTIME_FAST_P32M5_MOD=1 \
    RUNTIME_POLL_GLOBAL_ERROR=0 \
    RUNTIME_PACK_SHARED_KEYS="$packed" \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
    >"$LOGDIR/packed${packed}.build.out" 2>"$LOGDIR/packed${packed}.build.err"
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_sharedkey0_w${W}_g${NGPU}_b${BLOCKS}"
BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_sharedkey1_w${W}_g${NGPU}_b${BLOCKS}"
build_one 0 "$BIN0"
build_one 1 "$BIN1"

run_one() {
  local packed="$1" bin="$2" rep="$3" phase="$4"
  local out="$LOGDIR/packed${packed}_${phase}${rep}.out"
  local err="$LOGDIR/packed${packed}_${phase}${rep}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || {
    echo "pack_shared_keys=$packed $phase$rep missing runtime result" >&2
    cat "$out" >&2 || true
    cat "$err" >&2 || true
    exit 3
  }
  grep -Fq ' exact=OK' <<<"$line" || {
    echo "pack_shared_keys=$packed $phase$rep failed exactness" >&2
    echo "$line" >&2
    exit 4
  }
  if [[ "$phase" == run ]]; then
    local wall
    wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || { echo "pack_shared_keys=$packed run$rep missing wall_ms" >&2; exit 5; }
    printf '%s\t%s\t%s\n' "$packed" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r = 1; r <= WARMUP; ++r)); do
  run_one 0 "$BIN0" "$r" warmup
  run_one 1 "$BIN1" "$r" warmup
done

printf 'pack_shared_keys\trepeat\twall_ms\n' >"$RESULT"
for ((r = 1; r <= REPEATS; ++r)); do
  if (( r & 1 )); then order=(0 1); else order=(1 0); fi
  for packed in "${order[@]}"; do
    if [[ "$packed" == 0 ]]; then bin="$BIN0"; else bin="$BIN1"; fi
    echo "=== runtime pack-shared-keys=$packed run $r/$REPEATS ===" >&2
    run_one "$packed" "$bin" "$r" run
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
for mode in ('0', '1'):
    xs = [float(r['wall_ms']) for r in rows if r['pack_shared_keys'] == mode]
    if not xs:
        raise SystemExit(f'missing pack_shared_keys={mode} samples')
    out.append({
        'pack_shared_keys': mode,
        'repeats': len(xs),
        'wall_ms_median': f'{statistics.median(xs):.9f}',
        'wall_ms_min': f'{min(xs):.9f}',
        'wall_ms_max': f'{max(xs):.9f}',
    })

with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=('pack_shared_keys', 'repeats', 'wall_ms_median', 'wall_ms_min', 'wall_ms_max'),
        delimiter='\t')
    w.writeheader()
    w.writerows(out)

q = {r['pack_shared_keys']: r for r in out}
old = float(q['0']['wall_ms_median'])
new = float(q['1']['wall_ms_median'])
print(f'runtime_sharedkey_wall_speedup={old / new:.6f}x')
print(f'runtime_sharedkey_wall_delta_pct={(new / old - 1.0) * 100.0:.4f}%')
print('runtime_sharedkey_unpacked_key_bytes=16')
print('runtime_sharedkey_packed_key_bytes=8')
print('runtime_sharedkey_entries_per_block=1280')
print('runtime_sharedkey_unpacked_bytes_per_block=20480')
print('runtime_sharedkey_packed_bytes_per_block=10240')
print('runtime_sharedkey_bytes_saved_per_block=10240')
print('runtime_sharedkey_reduction_pct=50')
print('runtime_sharedkey_edge_cache=1')
print('runtime_sharedkey_fast_p32m5_mod=1')
print('runtime_sharedkey_poll_global_error=0')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas unpacked ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/packed0.build.err" >&2 || true
  echo '--- ptxas packed ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/packed1.build.err" >&2 || true
fi

echo "gridfp-reduced-runtime-sharedkey-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
