#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/gridfp-runtime-ab-env.sh"

W="${W:-10}"
NGPU="${NGPU:-2}"
BLOCKS="${BLOCKS:-256}"
MOD="${MOD:-4294967291}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-1}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "compact codec table runtime A/B supports even W=8..10" >&2; exit 2
fi
[[ "$MOD" == 4294967291 ]] || { echo "compact codec table runtime A/B requires MOD=4294967291" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_codec_tables_sym_u32_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PREINCLUDE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_codec_tables_sym_u32_preinclude.cuh"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
[[ -f "$PREINCLUDE" ]] || { echo "missing preinclude: $PREINCLUDE" >&2; exit 2; }

ARCH="$ARCH" LOG="$LOGDIR/nvcc-prepend-smoke.log" OUT="$ONEESAN_BUILD_DIR/gridfp_codec_tables_nvcc_prepend_smoke_should_not_exist" \
bash "$ONEESAN_ROOT/scripts/bench/gridfp-build-nvcc-prepend-smoke.sh" >"$LOGDIR/nvcc-prepend-smoke.out" 2>"$LOGDIR/nvcc-prepend-smoke.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh" >"$LOGDIR/choose-proof.out" 2>"$LOGDIR/choose-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-primitive-sym-u32-table-proof.sh" >"$LOGDIR/primitive-proof.out" 2>"$LOGDIR/primitive-proof.err"

mode_flags() {
  case "$1" in 0) echo "0 0";; 1) echo "1 0";; 2) echo "0 1";; 3) echo "1 1";; *) return 2;; esac
}

build_one() {
  local mode="$1" bin="$2" choose primitive
  read -r choose primitive <<<"$(mode_flags "$mode")"
  local inject="-include $PREINCLUDE -DRP_RUNTIME_CODEC_CHOOSE_SYM_U32=$choose -DRP_RUNTIME_CODEC_PRIMITIVE_SYM_U32=$primitive -DRP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT=0 -DRP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_LAST_R=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED=0"
  env "${GRIDFP_RUNTIME_AB_ENV[@]}" \
    NVCC_PREPEND_FLAGS="$inject" MODE=two-row-runtime-multigpu \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
    >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
  gridfp_runtime_ab_assert_build "$LOGDIR/mode${mode}.build.out"
  grep -Fq "nvcc_prepend=$inject)" "$LOGDIR/mode${mode}.build.out" || { echo "codec-table mode=$mode build did not report expected prepend flags" >&2; exit 6; }
  printf 'mode=%s choose=%s primitive=%s\n' "$mode" "$choose" "$primitive" >"$LOGDIR/mode${mode}.flags"
}

BINS=()
for mode in 0 1 2 3; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_codec_tables_sym_u32_m${mode}_w${W}_g${NGPU}_b${BLOCKS}"
  build_one "$mode" "${BINS[$mode]}"
done

run_one() {
  local mode="$1" rep="$2" phase="$3"
  local out="$LOGDIR/mode${mode}_${phase}${rep}.out" err="$LOGDIR/mode${mode}_${phase}${rep}.err"
  "${BINS[$mode]}" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "mode=$mode $phase$rep missing runtime result" >&2; cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' exact=OK' <<<"$line" || { echo "mode=$mode $phase$rep failed exactness" >&2; echo "$line" >&2; exit 4; }
  if [[ "$phase" == run ]]; then
    local wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || exit 5
    printf '%s\t%s\t%s\n' "$mode" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r=1; r<=WARMUP; ++r)); do for mode in 0 1 2 3; do run_one "$mode" "$r" warmup; done; done
printf 'mode\trepeat\twall_ms\n' >"$RESULT"
for ((r=1; r<=REPEATS; ++r)); do
  case $(((r-1)%4)) in 0) order=(0 1 2 3);; 1) order=(1 2 3 0);; 2) order=(2 3 0 1);; *) order=(3 0 1 2);; esac
  for mode in "${order[@]}"; do echo "=== runtime compact codec tables mode=$mode run $r/$REPEATS ===" >&2; run_one "$mode" "$r" run; done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t'))
names={'0':'baseline','1':'choose','2':'primitive','3':'choose_primitive'}; out=[]
for mode,name in names.items():
    xs=[float(r['wall_ms']) for r in rows if r['mode']==mode]
    if not xs: raise SystemExit(f'missing mode={mode}')
    out.append({'mode':mode,'name':name,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['mode']:r for r in out}; base=float(q['0']['wall_ms_median'])
for mode in ('1','2','3'):
    cur=float(q[mode]['wall_ms_median']); name=names[mode]
    print(f'runtime_codec_tables_{name}_wall_speedup={base/cur:.6f}x')
    print(f'runtime_codec_tables_{name}_wall_delta_pct={(cur/base-1)*100:.4f}%')
print('runtime_codec_tables_old_choose_bytes=6728')
print('runtime_codec_tables_old_primitive_bytes=6960')
print('runtime_codec_tables_new_choose_bytes=900')
print('runtime_codec_tables_new_primitive_bytes=900')
print('runtime_codec_tables_combined_physical_saved_bytes=11888')
print('runtime_codec_tables_turn_direct_compress_inverse=1')
print('runtime_codec_tables_signature_filter=0')
print('runtime_codec_tables_index_cache=0')
print('runtime_codec_tables_exact=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  for mode in 0 1 2 3; do echo "--- ptxas compact codec tables mode=$mode ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true; done
fi
echo "gridfp-reduced-runtime-codec-tables-sym-u32-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
