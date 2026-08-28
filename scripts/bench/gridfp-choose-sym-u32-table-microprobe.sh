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

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_choose_sym_u32_table_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh" \
  >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_choose_sym_u32_table_microprobe.cu"
PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
BINS=()
for mode in 0 1 2; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_choose_sym_u32_table_microprobe${mode}"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" -DRP_CHOOSE_TABLE_MODE="$mode" \
    "$SRC" -o "${BINS[$mode]}" \
    >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
done

printf 'pattern\tmode\trepeat\tkernel_ms\tns_per_call\tchecksum\n' >"$RESULT"
run_one() {
  local pattern="$1" mode="$2" rep="$3"
  local out="$LOGDIR/p${pattern}_m${mode}_run${rep}.out"
  local err="$LOGDIR/p${pattern}_m${mode}_run${rep}.err"
  "${BINS[$mode]}" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" "$pattern" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-choose-sym-u32-table-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq " mode=$mode " <<<"$line" || exit 4
  grep -Fq " pattern=$pattern " <<<"$line" || exit 4
  local ms ns checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 5
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$pattern" "$mode" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}

for ((r=1; r<=REPEATS; ++r)); do
  for pattern in 0 1; do
    case $(((r-1)%3)) in
      0) order=(0 1 2);; 1) order=(1 2 0);; *) order=(2 0 1);;
    esac
    for mode in "${order[@]}"; do
      echo "=== choose table pattern=$pattern mode=$mode run $r/$REPEATS ===" >&2
      run_one "$pattern" "$mode" "$r"
    done
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
out=[]
for pattern in ('0','1'):
    checks={}
    for mode in ('0','1','2'):
        rs=[r for r in rows if r['pattern']==pattern and r['mode']==mode]
        if not rs: raise SystemExit(f'missing pattern={pattern} mode={mode}')
        ss={r['checksum'] for r in rs}
        if len(ss)!=1: raise SystemExit(f'nondeterministic checksum pattern={pattern} mode={mode}')
        checks[mode]=next(iter(ss))
        ns=[float(r['ns_per_call']) for r in rs]
        out.append({'pattern':pattern,'mode':mode,'repeats':len(rs),'ns_per_call_median':f'{statistics.median(ns):.9f}','checksum':checks[mode]})
    if len(set(checks.values())) != 1:
        raise SystemExit(f'checksum mismatch pattern={pattern}: {checks}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={(r['pattern'],r['mode']):r for r in out}
for pattern,name in (('0','middle'),('1','mirrored')):
    old=float(q[(pattern,'0')]['ns_per_call_median'])
    for mode,label in (('1','sym900'),('2','tri1740')):
        new=float(q[(pattern,mode)]['ns_per_call_median'])
        print(f'choose_{label}_{name}_speedup={old/new:.6f}x')
        print(f'choose_{label}_{name}_delta_pct={(new/old-1)*100:.4f}%')
print('choose_table_mode0=full_2d_u64_6728B')
print('choose_table_mode1=symmetric_u32_900B')
print('choose_table_mode2=triangular_u32_1740B')
print('choose_sym_saved_bytes=5828')
print('choose_tri_saved_bytes=4988')
print('choose_table_load_bits=32')
print('choose_table_exact=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  for mode in 0 1 2; do
    echo "--- ptxas choose table mode=$mode ---" >&2
    grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true
  done
fi

echo "gridfp-choose-sym-u32-table-microprobe OK repeats=$REPEATS result=$RESULT" >&2
