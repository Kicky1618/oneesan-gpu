#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_80}"
BLOCKS="${BLOCKS:-256}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-4096}"
RUNS="${RUNS:-7}"
CHUNK="${CHUNK:-65536}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || RUNS < 1 || CHUNK < 2 )); then
  echo "invalid BLOCKS/THREADS/ITERS/RUNS/CHUNK" >&2
  exit 2
fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-select-ptx-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_address8_select_microprobe.cu"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_shard_address8_select_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

build_one() {
  local select="$1" bin="$2"
  local flags=()
  [[ "$PTXAS_VERBOSE" == 1 ]] && flags+=("-Xptxas=-v")
  "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" "${flags[@]}" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=1 -DB300_SHARD_ADDRESS_SELECT="$select" \
    "$SRC" -o "$bin" >"$LOGDIR/select_${select}.build.out" 2>"$LOGDIR/select_${select}.build.err"
}
BIN0="$ONEESAN_BUILD_DIR/b300_shard_address8_branchy"
BIN1="$ONEESAN_BUILD_DIR/b300_shard_address8_select"
build_one 0 "$BIN0"
build_one 1 "$BIN1"

printf 'select\trun\tmedian_ms\tGload_s\tchecksum\n' >"$RESULT"
run_one() {
  local select="$1" bin="$2" run="$3" out="$LOGDIR/select_${select}_run${run}.out"
  "$bin" "$BLOCKS" "$THREADS" "$ITERS" 1 "$CHUNK" >"$out"
  local line ms rate checksum
  line="$(grep '^gridfp-b300-shard-address8-select-microprobe OK ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2; exit 3; }
  grep -Fq "select=$select" <<<"$line" || { echo "$line" >&2; exit 4; }
  grep -Fq 'exact=OK' <<<"$line" || { echo "$line" >&2; exit 5; }
  ms="$(sed -nE 's/.* median_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  rate="$(sed -nE 's/.* Gload_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$rate" && -n "$checksum" ]] || exit 6
  printf '%s\t%s\t%s\t%s\t%s\n' "$select" "$run" "$ms" "$rate" "$checksum" >>"$RESULT"
}
for ((r=1; r<=RUNS; ++r)); do
  if (( r & 1 )); then order=(0 1); else order=(1 0); fi
  for select in "${order[@]}"; do
    [[ "$select" == 0 ]] && bin="$BIN0" || bin="$BIN1"
    echo "=== b300 shard-address8 select=$select run $r/$RUNS ===" >&2
    run_one "$select" "$bin" "$r"
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
    rs = [r for r in rows if r['select'] == mode]
    xs = [float(r['median_ms']) for r in rs]
    rates = [float(r['Gload_s']) for r in rs]
    cs = {r['checksum'] for r in rs}
    if not xs or len(cs) != 1:
        raise SystemExit(f'invalid select={mode} results')
    checks[mode] = next(iter(cs))
    out.append({
        'select': mode,
        'runs': len(xs),
        'median_ms': f'{statistics.median(xs):.9f}',
        'min_ms': f'{min(xs):.9f}',
        'max_ms': f'{max(xs):.9f}',
        'median_Gload_s': f'{statistics.median(rates):.9f}',
        'checksum': checks[mode],
    })
if checks['0'] != checks['1']:
    raise SystemExit(f'checksum mismatch branchy={checks["0"]} select={checks["1"]}')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=('select','runs','median_ms','min_ms','max_ms','median_Gload_s','checksum'), delimiter='\t')
    w.writeheader(); w.writerows(out)
q = {r['select']: r for r in out}
old = float(q['0']['median_ms']); new = float(q['1']['median_ms'])
print(f'b300_shard_address8_select_speedup={old/new:.6f}x')
print(f'b300_shard_address8_select_delta_pct={(new/old-1)*100:.4f}%')
print('b300_shard_address8_select_old=three_if_compare_subtract_stages')
print('b300_shard_address8_select_new=three_explicit_select_compare_subtract_stages')
print(f'checksum={checks["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas shard branchy ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/select_0.build.err" >&2 || true
  echo '--- ptxas shard select ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/select_1.build.err" >&2 || true
fi

echo "b300-shard-address8-select-ab OK runs=$RUNS blocks=$BLOCKS threads=$THREADS iters=$ITERS chunk=$CHUNK result=$RESULT" >&2
