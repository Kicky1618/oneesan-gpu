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
if (( LOGICAL_NGPU < 2 || LOGICAL_NGPU > 16 || BLOCKS < 1 || THREADS < 1 ||
      THREADS > 1024 || ITERS < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid W28 rank microprobe dimensions" >&2; exit 2
fi
for owner in $OWNERS; do (( owner >= 0 && owner < LOGICAL_NGPU )) || { echo "invalid owner=$owner" >&2; exit 2; }; done
for reverse in $DIRECTIONS; do [[ "$reverse" == 0 || "$reverse" == 1 ]] || { echo "DIRECTIONS must contain only 0 or 1" >&2; exit 2; }; done
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= 1 )) || { echo "need at least one visible GPU" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_w28_rank_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_codec_table_w28_rank_microprobe.cu"
PREINCLUDE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_codec_tables_sym_u32_preinclude.cuh"
[[ -f "$SRC" && -f "$PREINCLUDE" ]] || { echo "missing W28 rank probe input" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-proxy-coverage-proof.sh" >"$LOGDIR/proxy-proof.out" 2>"$LOGDIR/proxy-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh" >"$LOGDIR/choose-proof.out" 2>"$LOGDIR/choose-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-primitive-sym-u32-table-proof.sh" >"$LOGDIR/primitive-proof.out" 2>"$LOGDIR/primitive-proof.err"

mode_flags() {
  case "$1" in
    0) echo "0 0";; 1) echo "1 1";; 2) echo "2 1";;
    3) echo "3 1";; 4) echo "1 2";; 5) echo "3 2";;
    *) return 2;;
  esac
}

PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
BINS=()
for mode in 0 1 2 3 4 5; do
  read -r choose primitive <<<"$(mode_flags "$mode")"
  BINS[$mode]="$ONEESAN_BUILD_DIR/gridfp_codec_table_w28_rank_m${mode}"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" \
    -include "$PREINCLUDE" \
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
    "$SRC" -o "${BINS[$mode]}" >"$LOGDIR/mode${mode}.build.out" 2>"$LOGDIR/mode${mode}.build.err"
done

printf 'owner\tdirection\tmode\trepeat\tkernel_ms\tns_per_sample\tchecksum\n' >"$RESULT"
run_one() {
  local owner="$1" reverse="$2" mode="$3" rep="$4"
  local out="$LOGDIR/o${owner}_d${reverse}_m${mode}_r${rep}.out" err="$LOGDIR/o${owner}_d${reverse}_m${mode}_r${rep}.err"
  "${BINS[$mode]}" "$W" "$K" "$LOGICAL_NGPU" "$owner" "$reverse" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" >"$out" 2>"$err"
  local line="$(grep '^gridfp-codec-table-w28-rank-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' error=0' <<<"$line" || { echo "W28 rank device error: $line" >&2; exit 4; }
  local choose primitive; read -r choose primitive <<<"$(mode_flags "$mode")"
  grep -Fq " choose_mode=$choose " <<<"$line" || exit 5
  grep -Fq " primitive_mode=$primitive " <<<"$line" || exit 5
  local ms ns checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_sample=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 6
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$owner" "$reverse" "$mode" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}

for owner in $OWNERS; do
  for reverse in $DIRECTIONS; do
    for ((r=1;r<=REPEATS;++r)); do
      case $(((r-1)%6)) in
        0) order=(0 1 2 3 4 5);; 1) order=(1 2 3 4 5 0);; 2) order=(2 3 4 5 0 1);;
        3) order=(3 4 5 0 1 2);; 4) order=(4 5 0 1 2 3);; *) order=(5 0 1 2 3 4);;
      esac
      for mode in "${order[@]}"; do
        echo "=== W28 rank owner=$owner reverse=$reverse mode=$mode run $r/$REPEATS ===" >&2
        run_one "$owner" "$reverse" "$mode" "$r"
      done
    done
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
meta={
 '0':('baseline',0,0,13688),
 '1':('max_compact',1,1,1800),
 '2':('tri_choose_compact_primitive',2,1,2640),
 '3':('full_choose_compact_primitive',3,1,4264),
 '4':('sym_choose_full_primitive',1,2,4380),
 '5':('full_shape_both',3,2,6844),
}
keyed={m:{} for m in meta}; checks={}
for r in rows:
    key=(int(r['owner']),int(r['direction']),int(r['repeat']))
    m=r['mode']; ns=float(r['ns_per_sample']); checksum=r['checksum']
    if key in keyed[m]: raise SystemExit(f'duplicate mode={m} key={key}')
    keyed[m][key]=ns
    checks.setdefault((key[0],key[1],key[2]),{})[m]=checksum
base_keys=set(keyed['0'])
if not base_keys: raise SystemExit('missing baseline samples')
for m in meta:
    if set(keyed[m])!=base_keys: raise SystemExit(f'key set mismatch mode={m}')
for key,by_mode in checks.items():
    if set(by_mode)!=set(meta): raise SystemExit(f'missing checksum mode key={key}')
    if len(set(by_mode.values()))!=1: raise SystemExit(f'checksum mismatch key={key}: {by_mode}')
out=[]
for m,(name,choose,primitive,candidate_bytes) in meta.items():
    xs=list(keyed[m].values())
    paired=[keyed['0'][k]/keyed[m][k] for k in sorted(base_keys)]
    out.append({'mode':m,'name':name,'choose_mode':choose,'primitive_mode':primitive,
                'candidate_physical_bytes':candidate_bytes,'repeats':len(xs),
                'wall_ms_median':'0.000000000','wall_ms_min':'0.000000000','wall_ms_max':'0.000000000',
                'paired_speedup_median':f'{statistics.median(paired):.9f}',
                'paired_speedup_min':f'{min(paired):.9f}','paired_speedup_max':f'{max(paired):.9f}',
                'ns_per_sample_median':f'{statistics.median(xs):.9f}'})
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['mode']:r for r in out}
for m in ('1','2','3','4','5'):
    name=q[m]['name']
    print(f'w28_rank_{name}_paired_speedup_median={q[m]["paired_speedup_median"]}x')
    print(f'w28_rank_{name}_paired_speedup_min={q[m]["paired_speedup_min"]}x')
    print(f'w28_rank_{name}_all_pairs_faster={int(float(q[m]["paired_speedup_min"])>1.0)}')
winner=max((q[m] for m in ('1','2','3','4','5')), key=lambda r: float(r['paired_speedup_median']))
print(f'w28_rank_winner_mode={winner["mode"]}')
print(f'w28_rank_winner_name={winner["name"]}')
print(f'w28_rank_winner_paired_speedup_median={winner["paired_speedup_median"]}x')
print('w28_rank_checksum_exact=1')
print('w28_rank_physical_gpu_count_required=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  for mode in 0 1 2 3 4 5; do echo "--- ptxas W28 rank mode=$mode ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/mode${mode}.build.err" >&2 || true; done
fi
echo "gridfp-codec-table-w28-rank-microprobe OK logical_ngpu=$LOGICAL_NGPU owners='$OWNERS' directions='$DIRECTIONS' repeats=$REPEATS result=$RESULT" >&2
