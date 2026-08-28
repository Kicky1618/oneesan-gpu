#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-128}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-2}"
CASES="${CASES:-motzkin 25 50 75 100}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 ||
      ITERS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid microprobe dimensions" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
for input_case in $CASES; do
  [[ "$input_case" == motzkin ]] && continue
  if [[ ! "$input_case" =~ ^[0-9]+$ ]] ||
     (( input_case < 0 || input_case > 100 )); then
    echo "CASES entries must be 'motzkin' or integers in 0..100" >&2
    exit 2
  fi
done

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_runtime_turn_discovery_nonn_scan_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-turn-discovery-nonn-scan-proof.sh" \
  >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_turn_discovery_nonn_scan_microprobe.cu"
BIN0="$ONEESAN_BUILD_DIR/gridfp_runtime_turn_discovery_nonn_scan_microprobe0"
BIN1="$ONEESAN_BUILD_DIR/gridfp_runtime_turn_discovery_nonn_scan_microprobe1"
PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")

build_one() {
  local fast="$1" bin="$2"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" \
    -DRP_RUNTIME_TURN_DISCOVERY_NONN_SCAN="$fast" \
    "$SRC" -o "$bin" \
    >"$LOGDIR/fast${fast}.build.out" 2>"$LOGDIR/fast${fast}.build.err"
}
build_one 0 "$BIN0"
build_one 1 "$BIN1"

printf 'input_case\tnonn_scan\trepeat\tkernel_ms\tns_per_call\tactual_nonn_fraction\tchecksum\n' >"$RESULT"
run_one() {
  local input_case="$1" fast="$2" bin="$3" rep="$4"
  local safe_case="${input_case//[^A-Za-z0-9_-]/_}"
  local out="$LOGDIR/case${safe_case}_fast${fast}_run${rep}.out"
  local err="$LOGDIR/case${safe_case}_fast${fast}_run${rep}.err"
  "$bin" "$BLOCKS" "$THREADS" "$ITERS" "$input_case" "$WARMUP" \
    >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-runtime-turn-discovery-nonn-scan-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || {
    cat "$out" >&2 || true
    cat "$err" >&2 || true
    exit 3
  }
  grep -Fq " nonn_scan=$fast " <<<"$line" || {
    echo "unexpected scan mode: $line" >&2
    exit 4
  }
  grep -Fq " input_case=$input_case " <<<"$line" || {
    echo "unexpected input case: $line" >&2
    exit 4
  }
  local ms ns actual checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"
  actual="$(sed -nE 's/.* actual_nonn_fraction=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$actual" && -n "$checksum" ]] || exit 5
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$input_case" "$fast" "$rep" "$ms" "$ns" "$actual" "$checksum" >>"$RESULT"
}

for input_case in $CASES; do
  for ((r=1; r<=REPEATS; ++r)); do
    if ((r & 1)); then order=(0 1); else order=(1 0); fi
    for fast in "${order[@]}"; do
      [[ "$fast" == 0 ]] && bin="$BIN0" || bin="$BIN1"
      echo "=== W28 turn discovery case=$input_case nonn_scan=$fast run $r/$REPEATS ===" >&2
      run_one "$input_case" "$fast" "$bin" "$r"
    done
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv
import statistics
import sys

src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
cases = list(dict.fromkeys(r['input_case'] for r in rows))
out = []
for input_case in cases:
    checksums = {}
    actuals = {}
    medians = {}
    for mode in ('0', '1'):
        rs = [r for r in rows
              if r['input_case'] == input_case and r['nonn_scan'] == mode]
        if not rs:
            raise SystemExit(f'missing input_case={input_case} nonn_scan={mode}')
        cs = {r['checksum'] for r in rs}
        if len(cs) != 1:
            raise SystemExit(
                f'nondeterministic checksum input_case={input_case} mode={mode}: {sorted(cs)}')
        checksums[mode] = next(iter(cs))
        av = {r['actual_nonn_fraction'] for r in rs}
        if len(av) != 1:
            raise SystemExit(f'nondeterministic input density input_case={input_case} mode={mode}')
        actuals[mode] = next(iter(av))
        ns = [float(r['ns_per_call']) for r in rs]
        ms = [float(r['kernel_ms']) for r in rs]
        medians[mode] = statistics.median(ns)
        out.append({
            'input_case': input_case,
            'nonn_scan': mode,
            'repeats': len(rs),
            'actual_nonn_fraction': actuals[mode],
            'kernel_ms_median': f'{statistics.median(ms):.9f}',
            'ns_per_call_median': f'{medians[mode]:.9f}',
            'kernel_ms_min': f'{min(ms):.9f}',
            'kernel_ms_max': f'{max(ms):.9f}',
            'checksum': checksums[mode],
        })
    if checksums['0'] != checksums['1']:
        raise SystemExit(
            f'checksum mismatch input_case={input_case} '
            f'full={checksums["0"]} nonn={checksums["1"]}')
    if actuals['0'] != actuals['1']:
        raise SystemExit(f'input mismatch input_case={input_case}')
    key = ''.join(c if c.isalnum() else '_' for c in input_case)
    print(
        f'turn_discovery_nonn_scan_case_{key}_speedup='
        f'{medians["0"] / medians["1"]:.6f}x')
    print(
        f'turn_discovery_nonn_scan_case_{key}_delta_pct='
        f'{(medians["1"] / medians["0"] - 1) * 100:.4f}%')
    print(
        f'turn_discovery_nonn_scan_case_{key}_actual_nonn_fraction='
        f'{actuals["0"]}')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys(), delimiter='\t')
    w.writeheader()
    w.writerows(out)
print('turn_discovery_nonn_scan_old=full_position_scan')
print('turn_discovery_nonn_scan_new=non_n_setbit_scan')
print('turn_discovery_nonn_scan_realistic_case=motzkin_nn_conditioned')
print('turn_discovery_nonn_scan_W=28')
print('turn_discovery_nonn_scan_extra_constant_bytes=0')
print('turn_discovery_nonn_scan_extra_shared_bytes=0')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas full-position scan ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' \
    "$LOGDIR/fast0.build.err" >&2 || true
  echo '--- ptxas non-N set-bit scan ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' \
    "$LOGDIR/fast1.build.err" >&2 || true
fi

echo "gridfp-runtime-turn-discovery-nonn-scan-microprobe OK repeats=$REPEATS cases='$CASES' result=$RESULT" >&2
