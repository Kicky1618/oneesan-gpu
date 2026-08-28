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
RUNTIME_CACHE_EDGES="${RUNTIME_CACHE_EDGES:-1}"

if [[ "$MOD" != 4294967291 ]]; then
  echo "this A/B is specific to MOD=4294967291" >&2
  exit 2
fi
if (( W < 8 || W > 28 || W % 2 != 0 )); then
  echo "W must be even and in [8,28]" >&2
  exit 2
fi
if (( NGPU < 2 || NGPU > 16 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid NGPU/BLOCKS/REPEATS/WARMUP" >&2
  exit 2
fi
if [[ "$RUNTIME_CACHE_EDGES" != 0 && "$RUNTIME_CACHE_EDGES" != 1 ]]; then
  echo "RUNTIME_CACHE_EDGES must be 0 or 1" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_p32m5_mod_ab_w${W}_g${NGPU}_b${BLOCKS}_cache${RUNTIME_CACHE_EDGES}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-p32m5-mod-proof.sh" \
  >"$LOGDIR/p32m5-proof.out" 2>"$LOGDIR/p32m5-proof.err"

build_one() {
  local fast="$1" bin="$2"
  MODE=two-row-runtime-multigpu \
    RUNTIME_CACHE_EDGES="$RUNTIME_CACHE_EDGES" \
    RUNTIME_FAST_P32M5_MOD="$fast" \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
    >"$LOGDIR/fast${fast}.build.out" 2>"$LOGDIR/fast${fast}.build.err"
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_p32m5_fast0_w${W}_g${NGPU}_b${BLOCKS}"
BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_p32m5_fast1_w${W}_g${NGPU}_b${BLOCKS}"
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
    echo "fast_mod=$fast $phase$rep missing runtime result" >&2
    cat "$out" >&2 || true
    cat "$err" >&2 || true
    exit 3
  }
  grep -Fq ' exact=OK' <<<"$line" || {
    echo "fast_mod=$fast $phase$rep failed exactness" >&2
    echo "$line" >&2
    exit 4
  }
  if [[ "$phase" == run ]]; then
    local wall
    wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || { echo "fast_mod=$fast run$rep missing wall_ms" >&2; exit 5; }
    printf '%s\t%s\t%s\n' "$fast" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r = 1; r <= WARMUP; ++r)); do
  run_one 0 "$BIN0" "$r" warmup
  run_one 1 "$BIN1" "$r" warmup
done

printf 'fast_p32m5_mod\trepeat\twall_ms\n' >"$RESULT"
for ((r = 1; r <= REPEATS; ++r)); do
  if (( r & 1 )); then order=(0 1); else order=(1 0); fi
  for fast in "${order[@]}"; do
    if [[ "$fast" == 0 ]]; then bin="$BIN0"; else bin="$BIN1"; fi
    echo "=== runtime fast-p32m5-mod=$fast run $r/$REPEATS ===" >&2
    run_one "$fast" "$bin" "$r" run
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
    xs = [float(r['wall_ms']) for r in rows if r['fast_p32m5_mod'] == mode]
    if not xs:
        raise SystemExit(f'missing fast_p32m5_mod={mode} samples')
    out.append({
        'fast_p32m5_mod': mode,
        'repeats': len(xs),
        'wall_ms_median': f'{statistics.median(xs):.9f}',
        'wall_ms_min': f'{min(xs):.9f}',
        'wall_ms_max': f'{max(xs):.9f}',
    })

with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=('fast_p32m5_mod', 'repeats', 'wall_ms_median', 'wall_ms_min', 'wall_ms_max'),
        delimiter='\t')
    w.writeheader()
    w.writerows(out)

q = {r['fast_p32m5_mod']: r for r in out}
old = float(q['0']['wall_ms_median'])
new = float(q['1']['wall_ms_median'])
print(f'runtime_p32m5_mod_wall_speedup={old / new:.6f}x')
print(f'runtime_p32m5_mod_wall_delta_pct={(new / old - 1.0) * 100.0:.4f}%')
print('runtime_p32m5_modulus=4294967291')
print('runtime_p32m5_identity=2^32==5_mod_p')
print('runtime_p32m5_max_acc_mag=171798691600')
print('runtime_p32m5_max_hi=39')
print('runtime_p32m5_max_folded=4294967490')
print('runtime_p32m5_conditional_subtractions_max=1')
print('runtime_p32m5_signed_64bit_divisions=0')
print(f'summary={dst}')
PY

echo "gridfp-reduced-runtime-p32m5-mod-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS edge_cache=$RUNTIME_CACHE_EDGES result=$RESULT" >&2
