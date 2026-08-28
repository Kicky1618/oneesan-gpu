#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-512}"
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

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_primitive1_u32_table_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-primitive1-u32-table-proof.sh" \
  >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_primitive1_u32_table_microprobe.cu"
PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
BINS=()
for mode in 0 1 2 3; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_primitive1_u32_table_microprobe${mode}"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" -DRP_PRIMITIVE1_TABLE_MODE="$mode" \
    "$SRC" -o "${BINS[$mode]}" \
    >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
done

printf 'mode\trepeat\tkernel_ms\tns_per_call\tchecksum\n' >"$RESULT"
run_one() {
  local mode="$1" rep="$2"
  local out="$LOGDIR/mode${mode}_run${rep}.out"
  local err="$LOGDIR/mode${mode}_run${rep}.err"
  "${BINS[$mode]}" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-primitive1-u32-table-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq " mode=$mode " <<<"$line" || exit 4
  local ms ns checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 5
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}

for ((r=1; r<=REPEATS; ++r)); do
  case $(((r-1)%4)) in
    0) order=(0 1 2 3);; 1) order=(1 2 3 0);;
    2) order=(2 3 0 1);; *) order=(3 0 1 2);;
  esac
  for mode in "${order[@]}"; do
    echo "=== primitive1 table mode=$mode run $r/$REPEATS ===" >&2
    run_one "$mode" "$r"
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]; cs={}
names={'0':'primitive_2d_u64','1':'sector_1d_u64','2':'dedicated_1d_u32','3':'immediate_switch'}
for mode in ('0','1','2','3'):
    rs=[r for r in rows if r['mode']==mode]
    if not rs: raise SystemExit(f'missing mode={mode}')
    ss={r['checksum'] for r in rs}
    if len(ss)!=1: raise SystemExit(f'nondeterministic checksum mode={mode}: {sorted(ss)}')
    cs[mode]=next(iter(ss)); ms=[float(r['kernel_ms']) for r in rs]; ns=[float(r['ns_per_call']) for r in rs]
    out.append({'mode':mode,'name':names[mode],'repeats':len(rs),'kernel_ms_median':f'{statistics.median(ms):.9f}','ns_per_call_median':f'{statistics.median(ns):.9f}','checksum':cs[mode]})
if len(set(cs.values()))!=1: raise SystemExit(f'checksum mismatch: {cs}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['mode']:r for r in out}; base=float(q['0']['ns_per_call_median'])
for mode in ('1','2','3'):
    cur=float(q[mode]['ns_per_call_median']); name=names[mode]
    print(f'primitive1_{name}_speedup={base/cur:.6f}x')
    print(f'primitive1_{name}_delta_pct={(cur/base-1)*100:.4f}%')
print('primitive1_mode0=RP_PRIMITIVE_2d_u64')
print('primitive1_mode1=RP_SECTOR_PRIMITIVE_1d_u64')
print('primitive1_mode2=dedicated_14x_u32')
print('primitive1_mode3=immediate_switch')
print('primitive1_u32_table_bytes=56')
print('primitive1_exact=1')
print(f'checksum={cs["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  for mode in 0 1 2 3; do
    echo "--- ptxas primitive1 table mode=$mode ---" >&2
    grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true
  done
fi

echo "gridfp-primitive1-u32-table-microprobe OK repeats=$REPEATS result=$RESULT" >&2
