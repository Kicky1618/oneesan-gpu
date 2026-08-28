#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/gridfp-runtime-ab-env.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "choose table runtime A/B supports even W=8..10" >&2; exit 2; fi
[[ "$MOD" == 4294967291 ]] || { echo "choose table runtime A/B requires MOD=4294967291" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_choose_u32_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PREINCLUDE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_choose_sym_u32_preinclude.cuh"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"; [[ -f "$PREINCLUDE" ]] || { echo "missing preinclude: $PREINCLUDE" >&2; exit 2; }

ARCH="$ARCH" LOG="$LOGDIR/nvcc-prepend-smoke.log" OUT="$ONEESAN_BUILD_DIR/gridfp_choose_nvcc_prepend_smoke_should_not_exist" \
bash "$ONEESAN_ROOT/scripts/bench/gridfp-build-nvcc-prepend-smoke.sh" >"$LOGDIR/nvcc-prepend-smoke.out" 2>"$LOGDIR/nvcc-prepend-smoke.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh" >"$LOGDIR/choose-proof.out" 2>"$LOGDIR/choose-proof.err"

build_one() {
  local mode="$1" bin="$2"
  local inject="-include $PREINCLUDE -DRP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE=$mode -DRP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT=0 -DRP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_LAST_R=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED=0"
  env "${GRIDFP_RUNTIME_AB_ENV[@]}" NVCC_PREPEND_FLAGS="$inject" MODE=two-row-runtime-multigpu \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
  gridfp_runtime_ab_assert_build "$LOGDIR/mode${mode}.build.out"
  grep -Fq "nvcc_prepend=$inject)" "$LOGDIR/mode${mode}.build.out" || { echo "choose table mode=$mode build did not report expected prepend flags" >&2; exit 6; }
}

BINS=()
for mode in 0 1 2 3; do BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_choose_u32_m${mode}_w${W}_g${NGPU}_b${BLOCKS}"; build_one "$mode" "${BINS[$mode]}"; done

run_one() {
  local mode="$1" rep="$2" phase="$3" out="$LOGDIR/mode${1}_${3}${2}.out" err="$LOGDIR/mode${1}_${3}${2}.err"
  "${BINS[$mode]}" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "mode=$mode $phase$rep missing runtime result" >&2; cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' exact=OK' <<<"$line" || { echo "mode=$mode $phase$rep failed exactness" >&2; echo "$line" >&2; exit 4; }
  if [[ "$phase" == run ]]; then local wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$wall" ]] || exit 5; printf '%s\t%s\t%s\n' "$mode" "$rep" "$wall" >>"$RESULT"; fi
}
for ((r=1;r<=WARMUP;++r)); do for mode in 0 1 2 3; do run_one "$mode" "$r" warmup; done; done
printf 'mode\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do
  case $(((r-1)%4)) in 0) order=(0 1 2 3);; 1) order=(1 2 3 0);; 2) order=(2 3 0 1);; *) order=(3 0 1 2);; esac
  for mode in "${order[@]}"; do echo "=== runtime choose table mode=$mode run $r/$REPEATS ===" >&2; run_one "$mode" "$r" run; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
names={'0':'full_u64','1':'sym_u32','2':'tri_u32','3':'full_u32'}
times={m:{} for m in names}
for r in rows:
 m=r['mode']; rep=int(r['repeat']); wall=float(r['wall_ms'])
 if m not in times: raise SystemExit(f'unknown mode={m}')
 if rep in times[m]: raise SystemExit(f'duplicate mode={m} repeat={rep}')
 times[m][rep]=wall
base_reps=set(times['0'])
if not base_reps: raise SystemExit('missing baseline')
out=[]
for mode,name in names.items():
 if set(times[mode])!=base_reps: raise SystemExit(f'repeat set mismatch mode={mode}')
 xs=list(times[mode].values()); paired=[times['0'][r]/times[mode][r] for r in sorted(base_reps)]
 out.append({'mode':mode,'name':name,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}','paired_speedup_median':f'{statistics.median(paired):.9f}','paired_speedup_min':f'{min(paired):.9f}','paired_speedup_max':f'{max(paired):.9f}'})
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['mode']:r for r in out}; old=float(q['0']['wall_ms_median'])
for mode in ('1','2','3'):
 new=float(q[mode]['wall_ms_median']); name=names[mode]
 print(f'runtime_choose_{name}_wall_speedup={old/new:.6f}x'); print(f'runtime_choose_{name}_wall_delta_pct={(new/old-1)*100:.4f}%')
 print(f'runtime_choose_{name}_paired_speedup_median={q[mode]["paired_speedup_median"]}x'); print(f'runtime_choose_{name}_paired_speedup_min={q[mode]["paired_speedup_min"]}x'); print(f'runtime_choose_{name}_all_pairs_faster={int(float(q[mode]["paired_speedup_min"])>1.0)}')
winner=max((q[m] for m in ('1','2','3')), key=lambda r: float(r['paired_speedup_median']))
print(f'runtime_choose_winner_mode={winner["mode"]}'); print(f'runtime_choose_winner_name={winner["name"]}'); print(f'runtime_choose_winner_paired_speedup_median={winner["paired_speedup_median"]}x')
print('runtime_choose_mode0_bytes=6728'); print('runtime_choose_mode1_sym_bytes=900'); print('runtime_choose_mode2_tri_bytes=1740'); print('runtime_choose_mode3_full_u32_bytes=3364')
print('runtime_choose_sym_saved_bytes=5828'); print('runtime_choose_tri_saved_bytes=4988'); print('runtime_choose_full_u32_saved_bytes=3364')
print('runtime_choose_signature_filter=0'); print('runtime_choose_index_cache=0'); print('runtime_choose_exact=1'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then for mode in 0 1 2 3; do echo "--- ptxas choose table mode=$mode ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true; done; fi
echo "gridfp-reduced-runtime-choose-sym-u32-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
