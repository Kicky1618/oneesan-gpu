#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"; THREADS="${THREADS:-256}"; ITERS="${ITERS:-512}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-2}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
PATTERNS="${PATTERNS:-0 1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "invalid microprobe dimensions" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
for p in $PATTERNS; do [[ "$p" == 0 || "$p" == 1 ]] || { echo "PATTERNS must contain only 0 or 1" >&2; exit 2; }; done

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_motzkin_tri_u64_table_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-budget-proof.sh" >"$LOGDIR/budget-proof.out" 2>"$LOGDIR/budget-proof.err"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_motzkin_tri_u64_table_microprobe.cu"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
BINS=()
for mode in 0 1; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_motzkin_tri_u64_table_microprobe${mode}"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" \
    -DRP_MOTZKIN_TABLE_MODE="$mode" "$SRC" -o "${BINS[$mode]}" \
    >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
done

printf 'pattern\tmode\trepeat\tkernel_ms\tns_per_call\tchecksum\n' >"$RESULT"
run_one() {
  local pattern="$1" mode="$2" rep="$3" out="$LOGDIR/p${1}_m${2}_run${3}.out" err="$LOGDIR/p${1}_m${2}_run${3}.err"
  "${BINS[$mode]}" "$BLOCKS" "$THREADS" "$ITERS" "$pattern" "$WARMUP" >"$out" 2>"$err"
  local line="$(grep '^gridfp-motzkin-tri-u64-table-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq " mode=$mode " <<<"$line"; grep -Fq " pattern=$pattern " <<<"$line"
  local ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  local ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"
  local checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 5
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$pattern" "$mode" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}
for pattern in $PATTERNS; do
  for ((r=1;r<=REPEATS;++r)); do if ((r&1)); then order=(0 1); else order=(1 0); fi; for mode in "${order[@]}"; do echo "=== motzkin table pattern=$pattern mode=$mode run $r/$REPEATS ===" >&2; run_one "$pattern" "$mode" "$r"; done; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for pattern in sorted({r['pattern'] for r in rows},key=int):
 checks={}
 for mode in ('0','1'):
  rs=[r for r in rows if r['pattern']==pattern and r['mode']==mode]
  if not rs: raise SystemExit(f'missing pattern={pattern} mode={mode}')
  ss={r['checksum'] for r in rs}
  if len(ss)!=1: raise SystemExit(f'nondeterministic checksum pattern={pattern} mode={mode}')
  checks[mode]=next(iter(ss)); ns=[float(r['ns_per_call']) for r in rs]
  out.append({'pattern':pattern,'mode':mode,'repeats':len(rs),'ns_per_call_median':f'{statistics.median(ns):.9f}','checksum':checks[mode]})
 if checks['0']!=checks['1']: raise SystemExit(f'checksum mismatch pattern={pattern}: {checks}')
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
for pattern in sorted({r['pattern'] for r in out},key=int):
 q={r['mode']:r for r in out if r['pattern']==pattern}; old=float(q['0']['ns_per_call_median']); new=float(q['1']['ns_per_call_median'])
 print(f'motzkin_tri_u64_pattern{pattern}_speedup={old/new:.6f}x'); print(f'motzkin_tri_u64_pattern{pattern}_delta_pct={(new/old-1)*100:.4f}%')
print('motzkin_tri_u64_pattern0=random_valid_cells'); print('motzkin_tri_u64_pattern1=low_height_rows'); print('motzkin_tri_u64_old_bytes=6960'); print('motzkin_tri_u64_new_bytes=3480'); print('motzkin_tri_u64_saved_bytes=3480'); print('motzkin_tri_u64_exact=1'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then for mode in 0 1; do echo "--- ptxas motzkin table mode=$mode ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true; done; fi
echo "gridfp-motzkin-tri-u64-table-microprobe OK repeats=$REPEATS result=$RESULT" >&2
