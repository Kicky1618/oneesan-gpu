#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-128}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-2}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 ||
      REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid microprobe dimensions" >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2; exit 2
fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_materialize_primitive_packed_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-materialize-primitive-last-r-proof.sh" \
  >"$LOGDIR/last-r-proof.out" 2>"$LOGDIR/last-r-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-materialize-primitive-packed-proof.sh" \
  >"$LOGDIR/packed-proof.out" 2>"$LOGDIR/packed-proof.err"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_materialize_primitive_last_r_microprobe.cu"
BIN0="$ONEESAN_BUILD_DIR/gridfp_materialize_primitive_packed_microprobe0"
BIN1="$ONEESAN_BUILD_DIR/gridfp_materialize_primitive_packed_microprobe1"
PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")

build_one() {
  local packed="$1" bin="$2"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" \
    -DRP_FAST_MATERIALIZE_PRIMITIVE_SETBITS=1 \
    -DRP_FAST_MATERIALIZE_PRIMITIVE_LAST_R=1 \
    -DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED="$packed" \
    "$SRC" -o "$bin" \
    >"$LOGDIR/packed${packed}.build.out" 2>"$LOGDIR/packed${packed}.build.err"
}
build_one 0 "$BIN0"
build_one 1 "$BIN1"

printf 'packed\trepeat\tkernel_ms\tns_per_call\tchecksum\n' >"$RESULT"
run_one() {
  local packed="$1" bin="$2" rep="$3"
  local out="$LOGDIR/packed${packed}_run${rep}.out"
  local err="$LOGDIR/packed${packed}_run${rep}.err"
  "$bin" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-materialize-primitive-last-r-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' setbits=1 ' <<<"$line" || exit 4
  grep -Fq ' last_r=1 ' <<<"$line" || exit 4
  local ms ns checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 5
  printf '%s\t%s\t%s\t%s\t%s\n' "$packed" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}

for ((r=1; r<=REPEATS; ++r)); do
  if ((r & 1)); then order=(0 1); else order=(1 0); fi
  for packed in "${order[@]}"; do
    [[ "$packed" == 0 ]] && bin="$BIN0" || bin="$BIN1"
    echo "=== primitive packed=$packed run $r/$REPEATS ===" >&2
    run_one "$packed" "$bin" "$r"
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]; cs={}
for mode in ('0','1'):
 rs=[r for r in rows if r['packed']==mode]
 if not rs: raise SystemExit(f'missing packed={mode}')
 ss={r['checksum'] for r in rs}
 if len(ss)!=1: raise SystemExit(f'nondeterministic checksum packed={mode}: {sorted(ss)}')
 cs[mode]=next(iter(ss)); ms=[float(r['kernel_ms']) for r in rs]; ns=[float(r['ns_per_call']) for r in rs]
 out.append({'packed':mode,'repeats':len(rs),'kernel_ms_median':f'{statistics.median(ms):.9f}','ns_per_call_median':f'{statistics.median(ns):.9f}','checksum':cs[mode]})
if cs['0']!=cs['1']: raise SystemExit(f'checksum mismatch full={cs["0"]} packed={cs["1"]}')
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['packed']:r for r in out}; old=float(q['0']['ns_per_call_median']); new=float(q['1']['ns_per_call_median'])
print(f'materialize_primitive_packed_microprobe_speedup={old/new:.6f}x')
print(f'materialize_primitive_packed_microprobe_delta_pct={(new/old-1)*100:.4f}%')
print('materialize_primitive_packed_old_threshold_table_bytes=6960')
print('materialize_primitive_packed_new_materialize_table_bytes=416')
print('materialize_primitive_packed_new_threshold_load_bits=32')
print('materialize_primitive_packed_exact=1')
print(f'checksum={cs["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas full 64-bit primitive thresholds ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/packed0.build.err" >&2 || true
  echo '--- ptxas packed 32-bit materialize thresholds ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/packed1.build.err" >&2 || true
fi

echo "gridfp-materialize-primitive-packed-microprobe OK repeats=$REPEATS result=$RESULT" >&2
