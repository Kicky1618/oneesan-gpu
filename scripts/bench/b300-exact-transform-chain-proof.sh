#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
CHECK_HIGH_CHUNK="${CHECK_HIGH_CHUNK:-0}"
[[ "$CHECK_HIGH_CHUNK" == 0 || "$CHECK_HIGH_CHUNK" == 1 ]] || { echo "CHECK_HIGH_CHUNK must be 0 or 1" >&2; exit 2; }
TMP="$(mktemp -d "${TMPDIR:-/tmp}/b300-exact-chain.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

scripts=(
  gen-b300-main-mate-cache.py gen-b300-main-pull.py gen-b300-block-pull.py
  gen-b300-block-mate-cache.py gen-b300-low-window-drop-cache.py
  gen-b300-low-drop-chunk.py gen-b300-low-block-cache.py gen-b300-main-pull-ilp2.py
  gen-b300-batch-shard-address8.py gen-b300-runtime-threads.py
)
[[ "$CHECK_HIGH_CHUNK" == 1 ]] && scripts+=(gen-b300-high-drop-chunk.py)
for x in "${scripts[@]}"; do python3 -m py_compile "$ONEESAN_ROOT/scripts/build/$x"; done
bash -n "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
bash -n "$ONEESAN_ROOT/scripts/run/b300x8-exact.sh"

apply(){ local in="$1" script="$2" out="$3"; python3 "$ONEESAN_ROOT/scripts/build/$script" "$in" "$out" >/dev/null; }

a="$SRC"
for spec in \
  'gen-b300-main-mate-cache.py:01' \
  'gen-b300-main-pull.py:02' \
  'gen-b300-block-pull.py:03' \
  'gen-b300-block-mate-cache.py:04' \
  'gen-b300-low-window-drop-cache.py:05' \
  'gen-b300-low-drop-chunk.py:06' \
  'gen-b300-low-block-cache.py:07' \
  'gen-b300-main-pull-ilp2.py:08' \
  'gen-b300-batch-shard-address8.py:09' \
  'gen-b300-runtime-threads.py:10'; do
  script="${spec%%:*}"; tag="${spec##*:}"; b="$TMP/default-$tag.cu"; apply "$a" "$script" "$b"; a="$b"
done
grep -Fq 'main_pull_direct_pair_source_rank' "$a"
grep -Fq 'main_pull_kernel_ilp2' "$a"
grep -Fq 'b300_pack_low_window_main_mate' "$a"
grep -Fq 'D_LOW_DROP_CHUNK' "$a"
grep -Fq 'b300_pack_low_block_cache' "$a"
grep -Fq 'b300_low_block_lift_rank' "$a"
grep -Fq 'D_LOW14_MATES' "$a"
grep -Fq 'BatchShardAddress8' "$a"
grep -Fq 'GRIDFP_THREADS' "$a"

high=0
if [[ "$CHECK_HIGH_CHUNK" == 1 ]]; then
  a="$TMP/default-06.cu"
  b="$TMP/high-07.cu"; apply "$a" gen-b300-high-drop-chunk.py "$b"; a="$b"
  b="$TMP/high-08.cu"; apply "$a" gen-b300-low-block-cache.py "$b"; a="$b"
  b="$TMP/high-09.cu"; apply "$a" gen-b300-main-pull-ilp2.py "$b"; a="$b"
  grep -Fq 'b300_high_chunk_drop_rank' "$a"
  grep -Fq 'b300_high_chunk_lift_rank' "$a"
  grep -Fq 'b300_low_block_lift_rank' "$a"
  grep -Fq 'main_pull_kernel_ilp2' "$a"
  high=1
fi

echo "b300-exact-transform-chain-proof OK default_chain=1 ilp2=1 high_chunk_composition=$high python_syntax=1 shell_syntax=1 gpu_required=0"
