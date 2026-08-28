#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/gridfp-runtime-ab-env.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"; REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "owner label hotpath runtime A/B supports even W=8..10" >&2; exit 2; fi
[[ "$MOD" == 4294967291 ]] || { echo "owner label hotpath A/B requires MOD=4294967291" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_owner_label_hotpath_ab_w${W}_g${NGPU}_b${BLOCKS}}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-ab-env-proof.sh" >"$LOGDIR/runtime-ab-env-proof.out" 2>"$LOGDIR/runtime-ab-env-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-prefix-carry-begin-proof.sh" >"$LOGDIR/prefix-carry-proof.out" 2>"$LOGDIR/prefix-carry-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-local-sector-carry-begin-proof.sh" >"$LOGDIR/local-carry-proof.out" 2>"$LOGDIR/local-carry-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-component-support-adjacent-marks-proof.sh" >"$LOGDIR/adjacent-support-proof.out" 2>"$LOGDIR/adjacent-support-proof.err"

mode_flags() { case "$1" in 0) echo "0 0 0";; 1) echo "1 0 0";; 2) echo "0 1 0";; 3) echo "0 0 1";; 4) echo "1 1 1";; *) return 2;; esac; }
build_one() {
  local mode="$1" bin="$2" prefix_carry local_carry adjacent
  read -r prefix_carry local_carry adjacent <<<"$(mode_flags "$mode")"
  local inject="-DRP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN=$prefix_carry -DRP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN=$local_carry -DRP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS=$adjacent"
  env "${GRIDFP_RUNTIME_AB_ENV[@]}" NVCC_PREPEND_FLAGS="$inject" MODE=two-row-runtime-multigpu ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
  gridfp_runtime_ab_assert_build "$LOGDIR/mode${mode}.build.out"
  grep -Fq "nvcc_prepend=$inject)" "$LOGDIR/mode${mode}.build.out" || { echo "owner label mode=$mode build did not report expected prepend" >&2; exit 6; }
  printf 'mode=%s prefix_carry=%s local_carry=%s adjacent=%s\n' "$mode" "$prefix_carry" "$local_carry" "$adjacent" >"$LOGDIR/mode${mode}.flags"
}
BINS=(); for mode in 0 1 2 3 4; do BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_owner_label_hotpath_m${mode}_w${W}_g${NGPU}_b${BLOCKS}"; build_one "$mode" "${BINS[$mode]}"; done
run_one() {
  local mode="$1" rep="$2" phase="$3" out="$LOGDIR/mode${1}_${3}${2}.out" err="$LOGDIR/mode${1}_${3}${2}.err"
  "${BINS[$mode]}" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "owner_label_hotpath mode=$mode $phase$rep missing runtime result" >&2; cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' exact=OK' <<<"$line" || { echo "owner_label_hotpath mode=$mode $phase$rep failed exactness" >&2; echo "$line" >&2; exit 4; }
  if [[ "$phase" == run ]]; then local wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$wall" ]] || exit 5; printf '%s\t%s\t%s\n' "$mode" "$rep" "$wall" >>"$RESULT"; fi
}
for ((r=1;r<=WARMUP;++r)); do for mode in 0 1 2 3 4; do run_one "$mode" "$r" warmup; done; done
printf 'mode\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do case $(((r-1)%5)) in 0) order=(0 1 2 3 4);; 1) order=(1 2 3 4 0);; 2) order=(2 3 4 0 1);; 3) order=(3 4 0 1 2);; *) order=(4 0 1 2 3);; esac; for mode in "${order[@]}"; do echo "=== runtime owner-label-hotpath mode=$mode run $r/$REPEATS ===" >&2; run_one "$mode" "$r" run; done; done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); names={'0':'baseline','1':'prefix_carry','2':'owner_local_carry','3':'adjacent_marks','4':'all_three'}; out=[]
for mode,name in names.items():
 xs=[float(r['wall_ms']) for r in rows if r['mode']==mode]
 if not xs: raise SystemExit(f'missing mode={mode}')
 out.append({'mode':mode,'name':name,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['mode']:r for r in out}; base=float(q['0']['wall_ms_median'])
for mode in ('1','2','3','4'):
 cur=float(q[mode]['wall_ms_median']); name=names[mode]; print(f'runtime_owner_label_{name}_wall_speedup={base/cur:.6f}x'); print(f'runtime_owner_label_{name}_wall_delta_pct={(cur/base-1)*100:.4f}%')
print('runtime_owner_label_exact=1'); print('runtime_owner_label_turn_direct_compress_inverse=1'); print('runtime_owner_label_unrelated_signature_filter=0'); print('runtime_owner_label_index_cache=0'); print('runtime_owner_label_w28_trees=0'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then for mode in 0 1 2 3 4; do echo "--- ptxas owner-label-hotpath mode=$mode ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true; done; fi
echo "gridfp-reduced-runtime-owner-label-hotpath-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
