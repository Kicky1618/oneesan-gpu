#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/gridfp-runtime-ab-env.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 10 || W % 2 != 0 || NGPU < 2 || NGPU > 8 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "find-index bucket reuse A/B supports even W=8..10" >&2; exit 2; fi
[[ "$MOD" == 4294967291 ]] || { echo "find-index bucket reuse A/B requires MOD=4294967291" >&2; exit 2; }
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_find_index_bucket_reuse_ab_w${W}_g${NGPU}_b${BLOCKS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-find-index-cache-proof.sh" >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"

build_one() {
  local fast="$1" bin="$2"
  local inject="-DRP_RUNTIME_FIND_INDEX_BUCKET_REUSE=$fast"
  env "${GRIDFP_RUNTIME_AB_ENV[@]}" \
    RUNTIME_FIND_INDEX_CACHE=1 RUNTIME_FIND_INDEX_BUCKETS=64 RUNTIME_FIND_INDEX_WAYS=2 \
    NVCC_PREPEND_FLAGS="$inject" MODE=two-row-runtime-multigpu \
    ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/fast${fast}.build.out" 2>"$LOGDIR/fast${fast}.build.err"
  grep -Fq 'runtime_find_index_cache=1' "$LOGDIR/fast${fast}.build.out"
  grep -Fq 'runtime_find_index_storage_bytes=64' "$LOGDIR/fast${fast}.build.out"
  grep -Fq 'runtime_find_index_ways=2' "$LOGDIR/fast${fast}.build.out"
  grep -Fq "nvcc_prepend=$inject)" "$LOGDIR/fast${fast}.build.out" || { echo "bucket_reuse=$fast build did not report expected prepend flags" >&2; exit 6; }
}

BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_find_index_bucket_reuse0_w${W}_g${NGPU}_b${BLOCKS}"
BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_find_index_bucket_reuse1_w${W}_g${NGPU}_b${BLOCKS}"
build_one 0 "$BIN0"; build_one 1 "$BIN1"

run_one() {
  local fast="$1" bin="$2" rep="$3" phase="$4" out="$LOGDIR/fast${1}_${4}${3}.out" err="$LOGDIR/fast${1}_${4}${3}.err"
  "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"
  local line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "bucket_reuse=$fast $phase$rep missing runtime result" >&2; cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq ' exact=OK' <<<"$line" || { echo "bucket_reuse=$fast $phase$rep failed exactness" >&2; echo "$line" >&2; exit 4; }
  if [[ "$phase" == run ]]; then local wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$wall" ]] || exit 5; printf '%s\t%s\t%s\n' "$fast" "$rep" "$wall" >>"$RESULT"; fi
}
for ((r=1;r<=WARMUP;++r)); do run_one 0 "$BIN0" "$r" warmup; run_one 1 "$BIN1" "$r" warmup; done
printf 'find_index_bucket_reuse\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do if ((r&1)); then order=(0 1); else order=(1 0); fi; for fast in "${order[@]}"; do [[ "$fast" == 0 ]] && bin="$BIN0" || bin="$BIN1"; echo "=== runtime find-index bucket-reuse=$fast run $r/$REPEATS ===" >&2; run_one "$fast" "$bin" "$r" run; done; done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('0','1'):
 xs=[float(r['wall_ms']) for r in rows if r['find_index_bucket_reuse']==mode]
 if not xs: raise SystemExit(f'missing find_index_bucket_reuse={mode}')
 out.append({'find_index_bucket_reuse':mode,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['find_index_bucket_reuse']:r for r in out}; old=float(q['0']['wall_ms_median']); new=float(q['1']['wall_ms_median'])
print(f'runtime_find_index_bucket_reuse_wall_speedup={old/new:.6f}x'); print(f'runtime_find_index_bucket_reuse_wall_delta_pct={(new/old-1)*100:.4f}%')
print('runtime_find_index_bucket_reuse_storage_bytes=64'); print('runtime_find_index_bucket_reuse_ways=2'); print('runtime_find_index_bucket_reuse_hash_buckets=32'); print('runtime_find_index_bucket_reuse_memo=occupancy_high_bits'); print('runtime_find_index_bucket_reuse_record_rehash_after_miss=0'); print('runtime_find_index_bucket_reuse_shared_bytes_added=0'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then for fast in 0 1; do echo "--- ptxas find_index_bucket_reuse=$fast ---" >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/fast${fast}.build.err" >&2 || true; done; fi
echo "gridfp-reduced-runtime-find-index-bucket-reuse-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
