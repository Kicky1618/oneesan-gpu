#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
BLOCKS="${BLOCKS:-256}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-4096}"
RUNS="${RUNS:-7}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || RUNS < 1 )); then
  echo "invalid BLOCKS/THREADS/ITERS/RUNS" >&2; exit 2
fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-w28-ngpu8-direct-proof.sh"
PTX_ARCH="${PTX_ARCH:-sm_80}"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-w28-ngpu8-direct-integration-ptx-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_owner_w28_ngpu8_production_microprobe.cu"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_runtime_owner_w28_ngpu8_production_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

build_one() {
  local direct="$1" bin="$2"
  local flags=()
  [[ "$PTXAS_VERBOSE" == 1 ]] && flags+=("-Xptxas=-v")
  "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" "${flags[@]}" \
    -DRP_RUNTIME_OWNER_U32LIMB=1 \
    -DRP_RUNTIME_OWNER_W28_NGPU8_DIRECT="$direct" \
    "$SRC" -o "$bin" >"$LOGDIR/direct_${direct}.build.out" \
    2>"$LOGDIR/direct_${direct}.build.err"
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_runtime_owner_w28_ngpu8_production_generic"
BIN1="$ONEESAN_BUILD_DIR/gridfp_runtime_owner_w28_ngpu8_production_direct"
build_one 0 "$BIN0"
build_one 1 "$BIN1"

printf 'direct\trun\tmedian_ms\tchecksum\n' >"$RESULT"
run_one() {
  local direct="$1" bin="$2" run="$3"
  local out="$LOGDIR/direct_${direct}_run${run}.out"
  "$bin" "$BLOCKS" "$THREADS" "$ITERS" 1 >"$out"
  local line ms checksum
  line="$(grep '^gridfp-runtime-owner-w28-ngpu8-production-microprobe OK ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2; exit 3; }
  grep -Fq "direct=$direct" <<<"$line" || { echo "$line" >&2; exit 4; }
  grep -Fq 'exact=OK' <<<"$line" || { echo "$line" >&2; exit 5; }
  ms="$(sed -nE 's/.* median_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$checksum" ]] || { echo "$line" >&2; exit 6; }
  printf '%s\t%s\t%s\t%s\n' "$direct" "$run" "$ms" "$checksum" >>"$RESULT"
}

for ((r=1; r<=RUNS; ++r)); do
  if (( r & 1 )); then order=(0 1); else order=(1 0); fi
  for direct in "${order[@]}"; do
    [[ "$direct" == 0 ]] && bin="$BIN0" || bin="$BIN1"
    echo "=== production W28x8 owner direct=$direct run $r/$RUNS ===" >&2
    run_one "$direct" "$bin" "$r"
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
out = []
checks = {}
for mode in ('0', '1'):
    rs = [r for r in rows if r['direct'] == mode]
    xs = [float(r['median_ms']) for r in rs]
    cs = {r['checksum'] for r in rs}
    if not xs or len(cs) != 1:
        raise SystemExit(f'invalid direct={mode} results/checksums')
    checks[mode] = next(iter(cs))
    out.append({'direct': mode, 'runs': len(xs),
                'median_ms': f'{statistics.median(xs):.9f}',
                'min_ms': f'{min(xs):.9f}', 'max_ms': f'{max(xs):.9f}',
                'checksum': checks[mode]})
if checks['0'] != checks['1']:
    raise SystemExit(f'checksum mismatch generic={checks["0"]} direct={checks["1"]}')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=('direct','runs','median_ms','min_ms','max_ms','checksum'), delimiter='\t')
    w.writeheader(); w.writerows(out)
q = {r['direct']: r for r in out}
old = float(q['0']['median_ms']); new = float(q['1']['median_ms'])
print(f'runtime_owner_w28_ngpu8_direct_speedup={old/new:.6f}x')
print(f'runtime_owner_w28_ngpu8_direct_delta_pct={(new/old-1)*100:.4f}%')
print('runtime_owner_w28_ngpu8_generic=production_header_pure_u32_runtime_W_runtime_ngpu')
print('runtime_owner_w28_ngpu8_direct=production_header_magic9513_shift17_runtime_guard')
print('runtime_owner_w28_ngpu8_direct_meta_loads_hot_path=0')
print('runtime_owner_w28_ngpu8_direct_scale_mul_hot_path=0')
print('runtime_owner_w28_ngpu8_direct_product_lo_hot_path=0')
print('runtime_owner_w28_ngpu8_direct_mul64=0')
print(f'checksum={checks["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas production generic owner ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/direct_0.build.err" >&2 || true
  echo '--- ptxas production direct owner ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/direct_1.build.err" >&2 || true
fi

echo "gridfp-runtime-owner-w28-ngpu8-production-ab OK runs=$RUNS blocks=$BLOCKS threads=$THREADS iters=$ITERS result=$RESULT" >&2
