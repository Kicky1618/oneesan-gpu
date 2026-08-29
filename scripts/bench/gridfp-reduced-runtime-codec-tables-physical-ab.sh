#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/gridfp-runtime-ab-env.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
CANDIDATE_MODE="${CANDIDATE_MODE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "physical codec-table runtime A/B supports even W=8..10" >&2; exit 2
fi
[[ "$MOD" == 4294967291 ]] || { echo "physical codec-table runtime A/B requires MOD=4294967291" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

layout_flags() {
  case "$1" in
    0) echo "0 0 baseline 13688";;
    1) echo "1 1 max_compact 1800";;
    2) echo "2 1 tri_choose_compact_primitive 2640";;
    3) echo "3 1 full_choose_compact_primitive 4264";;
    4) echo "1 2 sym_choose_full_primitive 4380";;
    5) echo "3 2 full_shape_both 6844";;
    *) return 2;;
  esac
}
read -r CCHOOSE CPRIMITIVE CNAME CBYTES <<<"$(layout_flags "$CANDIDATE_MODE")" || {
  echo "CANDIDATE_MODE must be 1..5" >&2; exit 2; }
[[ "$CANDIDATE_MODE" != 0 ]] || { echo "CANDIDATE_MODE must be 1..5" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_codec_tables_physical_m${CANDIDATE_MODE}_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-ab-env-proof.sh" >"$LOGDIR/runtime-ab-env-proof.out" 2>"$LOGDIR/runtime-ab-env-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-replacement-proof.sh" >"$LOGDIR/physical-proof.out" 2>"$LOGDIR/physical-proof.err"
ARCH="$ARCH" LOG="$LOGDIR/nvcc-prepend-smoke.log" OUT="$ONEESAN_BUILD_DIR/gridfp_codec_physical_nvcc_prepend_smoke_should_not_exist" \
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-build-nvcc-prepend-smoke.sh" >"$LOGDIR/nvcc-prepend-smoke.out" 2>"$LOGDIR/nvcc-prepend-smoke.err"

build_one() {
  local label="$1" choose="$2" primitive="$3" bin="$4"
  local inject="-DRP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE=$choose -DRP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE=$primitive -DRP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN=0 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT=0 -DRP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_LAST_R=0 -DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED=0"
  env "${GRIDFP_RUNTIME_AB_ENV[@]}" NVCC_PREPEND_FLAGS="$inject" MODE=two-row-runtime-multigpu \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/${label}.build.out" 2>"$LOGDIR/${label}.build.err"
  gridfp_runtime_ab_assert_build "$LOGDIR/${label}.build.out"
  grep -Fq "nvcc_prepend=$inject)" "$LOGDIR/${label}.build.out" || {
    echo "physical codec-table $label build did not report expected prepend flags" >&2; exit 6; }
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_codec_physical_baseline_w${W}_g${NGPU}_b${BLOCKS}"
BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_codec_physical_m${CANDIDATE_MODE}_w${W}_g${NGPU}_b${BLOCKS}"
build_one baseline 0 0 "$BIN0"
build_one candidate "$CCHOOSE" "$CPRIMITIVE" "$BIN1"

run_one() {
  local variant="$1" bin="$2" rep="$3" phase="$4"
  local out="$LOGDIR/${variant}_${phase}${rep}.out" err="$LOGDIR/${variant}_${phase}${rep}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$variant $phase$rep missing runtime result" >&2; cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' exact=OK' <<<"$line" || { echo "$variant $phase$rep failed exactness" >&2; echo "$line" >&2; exit 4; }
  if [[ "$phase" == run ]]; then
    local wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
    [[ -n "$wall" ]] || exit 5
    printf '%s\t%s\t%s\n' "$variant" "$rep" "$wall" >>"$RESULT"
  fi
}

for ((r=1; r<=WARMUP; ++r)); do run_one baseline "$BIN0" "$r" warmup; run_one candidate "$BIN1" "$r" warmup; done
printf 'variant\trepeat\twall_ms\n' >"$RESULT"
for ((r=1; r<=REPEATS; ++r)); do
  if ((r&1)); then order=(baseline candidate); else order=(candidate baseline); fi
  for variant in "${order[@]}"; do
    [[ "$variant" == baseline ]] && bin="$BIN0" || bin="$BIN1"
    echo "=== physical codec table $variant run $r/$REPEATS ===" >&2
    run_one "$variant" "$bin" "$r" run
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" "$CANDIDATE_MODE" "$CNAME" "$CCHOOSE" "$CPRIMITIVE" "$CBYTES" <<'PY'
import csv,statistics,sys
src,dst,mode,name,choose,primitive,cbytes=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
times={'baseline':{},'candidate':{}}
for r in rows:
    v=r['variant']; rep=int(r['repeat']); wall=float(r['wall_ms'])
    if v not in times: raise SystemExit(f'unknown variant={v}')
    if rep in times[v]: raise SystemExit(f'duplicate variant={v} repeat={rep}')
    times[v][rep]=wall
if not times['baseline'] or set(times['baseline']) != set(times['candidate']): raise SystemExit('repeat mismatch')
reps=sorted(times['baseline'])
base=[times['baseline'][r] for r in reps]; cand=[times['candidate'][r] for r in reps]
paired=[times['baseline'][r]/times['candidate'][r] for r in reps]
out=[
 {'mode':'0','name':'baseline','choose_mode':'0','primitive_mode':'0','constant_bytes':'13688','repeats':len(reps),'wall_ms_median':f'{statistics.median(base):.9f}','paired_speedup_median':'1.000000000','paired_speedup_min':'1.000000000','paired_speedup_max':'1.000000000'},
 {'mode':mode,'name':name,'choose_mode':choose,'primitive_mode':primitive,'constant_bytes':cbytes,'repeats':len(reps),'wall_ms_median':f'{statistics.median(cand):.9f}','paired_speedup_median':f'{statistics.median(paired):.9f}','paired_speedup_min':f'{min(paired):.9f}','paired_speedup_max':f'{max(paired):.9f}'},
]
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
med=statistics.median(paired)
print(f'physical_codec_candidate_mode={mode}')
print(f'physical_codec_candidate_name={name}')
print(f'physical_codec_candidate_choose_mode={choose}')
print(f'physical_codec_candidate_primitive_mode={primitive}')
print(f'physical_codec_candidate_constant_bytes={cbytes}')
print(f'physical_codec_candidate_saved_constant_bytes={13688-int(cbytes)}')
print(f'physical_codec_paired_speedup_median={med:.9f}x')
print(f'physical_codec_paired_speedup_min={min(paired):.9f}x')
print(f'physical_codec_paired_speedup_max={max(paired):.9f}x')
print(f'physical_codec_all_pairs_faster={int(min(paired)>1.0)}')
print('physical_codec_legacy_upload_sink_space=global')
print('physical_codec_legacy_constant_retained=0')
print('physical_codec_exact=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas physical codec baseline ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/baseline.build.err" >&2 || true
  echo "--- ptxas physical codec candidate mode=$CANDIDATE_MODE ---" >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/candidate.build.err" >&2 || true
fi

echo "gridfp-reduced-runtime-codec-tables-physical-ab OK candidate_mode=$CANDIDATE_MODE W=$W ngpu=$NGPU repeats=$REPEATS result=$RESULT" >&2
