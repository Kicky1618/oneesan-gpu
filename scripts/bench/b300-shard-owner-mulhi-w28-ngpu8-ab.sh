#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_80}"
BLOCKS="${BLOCKS:-256}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-8192}"
RUNS="${RUNS:-9}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || RUNS < 1 )); then
  echo "invalid BLOCKS/THREADS/ITERS/RUNS" >&2
  exit 2
fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-ptx-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_owner_mulhi_w28_ngpu8_microprobe.cu"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_shard_owner_mulhi_w28_ngpu8_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

build_one() {
  local mode="$1" bin="$2"
  local flags=()
  [[ "$PTXAS_VERBOSE" == 1 ]] && flags+=("-Xptxas=-v")
  "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" "${flags[@]}" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=1 -DB300_SHARD_OWNER_MODE="$mode" \
    "$SRC" -o "$bin" >"$LOGDIR/mode_${mode}.build.out" 2>"$LOGDIR/mode_${mode}.build.err"
}
BINS=()
for mode in 0 1 2; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/b300_shard_owner_mulhi_w28_g8_mode${mode}"
  build_one "$mode" "${BINS[$mode]}"
done

printf 'mode\trun\tmedian_ms\tGaddr_s\tchecksum\n' >"$RESULT"
run_one() {
  local mode="$1" run="$2" out="$LOGDIR/mode_${mode}_run${run}.out"
  "${BINS[$mode]}" "$BLOCKS" "$THREADS" "$ITERS" 1 >"$out"
  local line ms rate checksum
  line="$(grep '^gridfp-b300-shard-owner-mulhi-w28-ngpu8-microprobe OK ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2; exit 3; }
  grep -Fq "mode=$mode" <<<"$line" || { echo "$line" >&2; exit 4; }
  grep -Fq 'exact=OK' <<<"$line" || { echo "$line" >&2; exit 5; }
  ms="$(sed -nE 's/.* median_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  rate="$(sed -nE 's/.* Gaddr_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$rate" && -n "$checksum" ]] || exit 6
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$ms" "$rate" "$checksum" >>"$RESULT"
}
for ((r=1; r<=RUNS; ++r)); do
  case $(( (r-1) % 3 )) in
    0) order=(0 1 2) ;;
    1) order=(1 2 0) ;;
    2) order=(2 0 1) ;;
  esac
  for mode in "${order[@]}"; do
    echo "=== b300 W28x8 shard owner mode=$mode run $r/$RUNS ===" >&2
    run_one "$mode" "$r"
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
out = []
checks = {}
for mode in ('0','1','2'):
    rs = [r for r in rows if r['mode'] == mode]
    xs = [float(r['median_ms']) for r in rs]
    rates = [float(r['Gaddr_s']) for r in rs]
    cs = {r['checksum'] for r in rs}
    if not xs or len(cs) != 1:
        raise SystemExit(f'invalid mode={mode} results')
    checks[mode] = next(iter(cs))
    out.append({
        'mode': mode,
        'runs': len(xs),
        'median_ms': f'{statistics.median(xs):.9f}',
        'min_ms': f'{min(xs):.9f}',
        'max_ms': f'{max(xs):.9f}',
        'median_Gaddr_s': f'{statistics.median(rates):.9f}',
        'checksum': checks[mode],
    })
if len(set(checks.values())) != 1:
    raise SystemExit(f'checksum mismatch {checks}')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=('mode','runs','median_ms','min_ms','max_ms','median_Gaddr_s','checksum'), delimiter='\t')
    w.writeheader(); w.writerows(out)
q = {r['mode']: r for r in out}
base = float(q['0']['median_ms'])
for mode, name in [('1','mulhi_mul'),('2','mulhi_table')]:
    new = float(q[mode]['median_ms'])
    print(f'b300_shard_owner_{name}_speedup={base/new:.6f}x')
    print(f'b300_shard_owner_{name}_delta_pct={(new/base-1)*100:.4f}%')
print('mode0=three_compare_subtract_stages')
print('mode1=mulhi_shift_plus_owner_times_chunk')
print('mode2=mulhi_shift_plus_constant_base_lookup')
print('main_magic=195888106327 main_high_shift=9')
print('block_magic=139905900989 block_high_shift=7')
print(f'checksum={checks["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  for mode in 0 1 2; do
    echo "--- ptxas shard-owner mode=$mode ---" >&2
    grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode_${mode}.build.err" >&2 || true
  done
fi

echo "b300-shard-owner-mulhi-w28-ngpu8-ab OK runs=$RUNS blocks=$BLOCKS threads=$THREADS iters=$ITERS result=$RESULT" >&2
