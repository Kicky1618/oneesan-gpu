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
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_fastdiv64_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-fastdiv64-proof.sh" \
  >"$LOGDIR/fastdiv64-proof.out" 2>"$LOGDIR/fastdiv64-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-group-magic-proof.sh" \
  >"$LOGDIR/owner-group-magic-proof.out" 2>"$LOGDIR/owner-group-magic-proof.err"

build_one() {
  local fastdiv="$1" bin="$2"
  MODE=two-row-runtime-multigpu \
    RUNTIME_CACHE_EDGES=1 \
    RUNTIME_FAST_P32M5_MOD=1 \
    RUNTIME_POLL_GLOBAL_ERROR=0 \
    RUNTIME_PACK_SHARED_KEYS=1 \
    RUNTIME_FAST_DIV64="$fastdiv" \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
    >"$LOGDIR/fastdiv${fastdiv}.build.out" 2>"$LOGDIR/fastdiv${fastdiv}.build.err"
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_fastdiv0_w${W}_g${NGPU}_b${BLOCKS}"
BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_fastdiv1_w${W}_g${NGPU}_b${BLOCKS}"
build_one 0 "$BIN0"
build_one 1 "$BIN1"

run_one() {
  local fastdiv="$1" bin="$2" rep="$3" phase="$4"
  local out="$LOGDIR/fastdiv${fastdiv}_${phase}${rep}.out"
  local err="$LOGDIR/fastdiv${fastdiv}_${phase}${rep}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || {
    echo "fast_div64=$fastdiv $phase$rep missing runtime result" >&2
    cat "$out" >&2 || true
    cat "$err" >&2 || true
    exit 3
  }
  grep -Fq ' exact=OK' <<<"$line" || {
    echo "fast_div64=$fastdiv $phase$rep failed exactness" >&2
    echo "$line" >&2
    exit 4
  }
  if [[ "$phase" == run ]]; then
    local wall
    wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || { echo "fast_div64=$fastdiv run$rep missing wall_ms" >&2; exit 5; }
    printf '%s\t%s\t%s\n' "$fastdiv" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r = 1; r <= WARMUP; ++r)); do
  run_one 0 "$BIN0" "$r" warmup
  run_one 1 "$BIN1" "$r" warmup
done

printf 'fast_div64\trepeat\twall_ms\n' >"$RESULT"
for ((r = 1; r <= REPEATS; ++r)); do
  if (( r & 1 )); then order=(0 1); else order=(1 0); fi
  for fastdiv in "${order[@]}"; do
    if [[ "$fastdiv" == 0 ]]; then bin="$BIN0"; else bin="$BIN1"; fi
    echo "=== runtime fast-div64=$fastdiv run $r/$REPEATS ===" >&2
    run_one "$fastdiv" "$bin" "$r" run
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
    xs = [float(r['wall_ms']) for r in rows if r['fast_div64'] == mode]
    if not xs:
        raise SystemExit(f'missing fast_div64={mode} samples')
    out.append({
        'fast_div64': mode,
        'repeats': len(xs),
        'wall_ms_median': f'{statistics.median(xs):.9f}',
        'wall_ms_min': f'{min(xs):.9f}',
        'wall_ms_max': f'{max(xs):.9f}',
    })

with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=('fast_div64', 'repeats', 'wall_ms_median', 'wall_ms_min', 'wall_ms_max'),
        delimiter='\t')
    w.writeheader()
    w.writerows(out)

q = {r['fast_div64']: r for r in out}
old = float(q['0']['wall_ms_median'])
new = float(q['1']['wall_ms_median'])
print(f'runtime_fastdiv64_wall_speedup={old / new:.6f}x')
print(f'runtime_fastdiv64_wall_delta_pct={(new / old - 1.0) * 100.0:.4f}%')
print('runtime_fastdiv64_scope=owner_label_component_group_and_primitive_divmods')
print('runtime_fastdiv64_removed_dynamic_divmod_pairs_per_owner_label=2')
print('runtime_fastdiv64_turn_compress_divmod_pairs_remaining=2')
print('runtime_fastdiv64_primitive_magic_entries=29')
print('runtime_fastdiv64_owner_group_magic_entries=154')
print('runtime_fastdiv64_owner_group_magic_bytes=1232')
print('runtime_fastdiv64_edge_cache=1')
print('runtime_fastdiv64_fast_p32m5_mod=1')
print('runtime_fastdiv64_poll_global_error=0')
print('runtime_fastdiv64_pack_shared_keys=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas slow div64 ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fastdiv0.build.err" >&2 || true
  echo '--- ptxas fast div64 ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fastdiv1.build.err" >&2 || true
fi

echo "gridfp-reduced-runtime-fastdiv64-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
