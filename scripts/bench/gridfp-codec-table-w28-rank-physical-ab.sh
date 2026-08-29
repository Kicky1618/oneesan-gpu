#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W=28
K=13
LOGICAL_NGPU="${LOGICAL_NGPU:-8}"
OWNERS="${OWNERS:-0}"
DIRECTIONS="${DIRECTIONS:-0 1}"
BLOCKS="${BLOCKS:-256}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-16}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-1}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
CANDIDATE_MODE="${CANDIDATE_MODE:-1}"
if (( LOGICAL_NGPU < 2 || LOGICAL_NGPU > 16 || BLOCKS < 1 || THREADS < 1 ||
      THREADS > 1024 || ITERS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid W28 physical rank A/B dimensions" >&2; exit 2
fi
for owner in $OWNERS; do (( owner >= 0 && owner < LOGICAL_NGPU )) || { echo "invalid owner=$owner" >&2; exit 2; }; done
for reverse in $DIRECTIONS; do [[ "$reverse" == 0 || "$reverse" == 1 ]] || { echo "DIRECTIONS must contain only 0 or 1" >&2; exit 2; }; done
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= 1 )) || { echo "need at least one visible GPU" >&2; exit 2; }

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
read -r CCHOOSE CPRIMITIVE CNAME CBYTES <<<"$(layout_flags "$CANDIDATE_MODE")" || { echo "CANDIDATE_MODE must be 1..5" >&2; exit 2; }
[[ "$CANDIDATE_MODE" != 0 ]] || { echo "CANDIDATE_MODE must be 1..5" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_w28_rank_physical_m${CANDIDATE_MODE}_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_codec_table_w28_rank_microprobe.cu"
[[ -f "$SRC" ]] || { echo "missing W28 rank probe source: $SRC" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-physical-replacement-proof.sh" >"$LOGDIR/physical-proof.out" 2>"$LOGDIR/physical-proof.err"

PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
build_one() {
  local label="$1" choose="$2" primitive="$3" bin="$4"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" \
    -DRP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE="$choose" \
    -DRP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE="$primitive" \
    -DRP_RUNTIME_CODEC_CHOOSE_U32_MODE="$choose" \
    -DRP_RUNTIME_CODEC_PRIMITIVE_U32_MODE="$primitive" \
    -DRP_RUNTIME_PRIMITIVE_RANK_SETBITS=1 \
    -DRP_RUNTIME_SUPPORT_RANK_SETBITS=1 \
    -DRP_RUNTIME_FUSE_PRIMITIVE_SUPPORT_RANK=1 \
    -DRP_RUNTIME_DIRECT_BLOCKED_RANK=1 \
    -DRP_RUNTIME_BROADWORD_SUPPORT=1 \
    -DRP_RUNTIME_SECTOR_OFFSET_TABLE=1 \
    -DRP_RUNTIME_CACHE_SECTOR_ROW_BASE=1 \
    -DRP_RUNTIME_OUTER_GROUP_TABLE=1 \
    -DRP_RUNTIME_OWNER_FROM_BOUNDARIES=1 \
    -DRP_RUNTIME_OWNER_RECIPROCAL=1 \
    -DRP_RUNTIME_OWNER_FIXED54=0 \
    -DRP_RUNTIME_OWNER_FIXED52=1 \
    -DRP_RUNTIME_OWNER_U32LIMB=0 \
    -DRP_RUNTIME_OWNER_W28_NGPU8_DIRECT=0 \
    -DRP_FAST_OWNER_SUPPORT_BITPACK=1 \
    -DRP_FAST_MATERIALIZE_PRIMITIVE_SETBITS=1 \
    -DRP_FAST_SUPPORT_UNRANK_EARLY_EXIT=1 \
    -DRP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN=0 \
    -DRP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN=0 \
    -DRP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT=0 \
    -DRP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE=0 \
    -DRP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS=0 \
    -DRP_FAST_MATERIALIZE_PRIMITIVE_LAST_R=0 \
    -DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED=0 \
    "$SRC" -o "$bin" >"$LOGDIR/${label}.build.out" 2>"$LOGDIR/${label}.build.err"
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_codec_table_w28_rank_physical_baseline"
BIN1="$ONEESAN_BUILD_DIR/gridfp_codec_table_w28_rank_physical_m${CANDIDATE_MODE}"
build_one baseline 0 0 "$BIN0"
build_one candidate "$CCHOOSE" "$CPRIMITIVE" "$BIN1"

printf 'owner\tdirection\tvariant\trepeat\tkernel_ms\tns_per_sample\tchecksum\n' >"$RESULT"
run_one() {
  local owner="$1" reverse="$2" variant="$3" bin="$4" choose="$5" primitive="$6" rep="$7"
  local out="$LOGDIR/o${owner}_d${reverse}_${variant}_r${rep}.out" err="$LOGDIR/o${owner}_d${reverse}_${variant}_r${rep}.err"
  "$bin" "$W" "$K" "$LOGICAL_NGPU" "$owner" "$reverse" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" >"$out" 2>"$err"
  local line="$(grep '^gridfp-codec-table-w28-rank-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' error=0' <<<"$line" || { echo "W28 physical rank device error: $line" >&2; exit 4; }
  grep -Fq " choose_mode=$choose " <<<"$line" || { echo "choose mode label mismatch: $line" >&2; exit 5; }
  grep -Fq " primitive_mode=$primitive " <<<"$line" || { echo "primitive mode label mismatch: $line" >&2; exit 5; }
  local ms ns checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_sample=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 6
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$owner" "$reverse" "$variant" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}

for owner in $OWNERS; do
  for reverse in $DIRECTIONS; do
    for ((r=1;r<=REPEATS;++r)); do
      if ((r&1)); then order=(baseline candidate); else order=(candidate baseline); fi
      for variant in "${order[@]}"; do
        if [[ "$variant" == baseline ]]; then bin="$BIN0"; choose=0; primitive=0; else bin="$BIN1"; choose="$CCHOOSE"; primitive="$CPRIMITIVE"; fi
        echo "=== W28 physical rank owner=$owner reverse=$reverse $variant run $r/$REPEATS ===" >&2
        run_one "$owner" "$reverse" "$variant" "$bin" "$choose" "$primitive" "$r"
      done
    done
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" "$CANDIDATE_MODE" "$CNAME" "$CCHOOSE" "$CPRIMITIVE" "$CBYTES" <<'PY'
import csv,statistics,sys
src,dst,mode,name,choose,primitive,cbytes=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
times={'baseline':{},'candidate':{}}; checks={}
for r in rows:
    key=(int(r['owner']),int(r['direction']),int(r['repeat']))
    v=r['variant']; ns=float(r['ns_per_sample']); checksum=r['checksum']
    if v not in times: raise SystemExit(f'unknown variant={v}')
    if key in times[v]: raise SystemExit(f'duplicate variant={v} key={key}')
    times[v][key]=ns; checks.setdefault(key,{})[v]=checksum
if not times['baseline'] or set(times['baseline'])!=set(times['candidate']): raise SystemExit('sample key mismatch')
for key,by in checks.items():
    if set(by)!={'baseline','candidate'} or len(set(by.values()))!=1: raise SystemExit(f'checksum mismatch key={key}: {by}')
keys=sorted(times['baseline'])
base=[times['baseline'][k] for k in keys]; cand=[times['candidate'][k] for k in keys]
paired=[times['baseline'][k]/times['candidate'][k] for k in keys]
out=[
 {'mode':'0','name':'baseline','choose_mode':'0','primitive_mode':'0','candidate_physical_bytes':'13688','repeats':len(keys),'paired_speedup_median':'1.000000000','paired_speedup_min':'1.000000000','paired_speedup_max':'1.000000000','ns_per_sample_median':f'{statistics.median(base):.9f}'},
 {'mode':mode,'name':name,'choose_mode':choose,'primitive_mode':primitive,'candidate_physical_bytes':cbytes,'repeats':len(keys),'paired_speedup_median':f'{statistics.median(paired):.9f}','paired_speedup_min':f'{min(paired):.9f}','paired_speedup_max':f'{max(paired):.9f}','ns_per_sample_median':f'{statistics.median(cand):.9f}'},
]
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
print(f'w28_physical_candidate_mode={mode}')
print(f'w28_physical_candidate_name={name}')
print(f'w28_physical_candidate_constant_bytes={cbytes}')
print(f'w28_physical_paired_speedup_median={statistics.median(paired):.9f}x')
print(f'w28_physical_paired_speedup_min={min(paired):.9f}x')
print(f'w28_physical_paired_speedup_max={max(paired):.9f}x')
print(f'w28_physical_all_pairs_faster={int(min(paired)>1.0)}')
print('w28_physical_checksum_exact=1')
print('w28_physical_legacy_constant_retained=0')
print('w28_physical_gpu_count_required=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas W28 physical baseline ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/baseline.build.err" >&2 || true
  echo "--- ptxas W28 physical candidate mode=$CANDIDATE_MODE ---" >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/candidate.build.err" >&2 || true
fi

echo "gridfp-codec-table-w28-rank-physical-ab OK candidate_mode=$CANDIDATE_MODE logical_ngpu=$LOGICAL_NGPU owners='$OWNERS' directions='$DIRECTIONS' repeats=$REPEATS result=$RESULT" >&2
