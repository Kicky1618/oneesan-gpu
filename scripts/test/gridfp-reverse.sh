#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --gpu ) ]]; then
  echo "usage: $0 [--gpu]" >&2
  exit 2
fi
OUT="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-reverse-test.XXXXXX")"
echo "Test artifacts: $OUT"
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra -fsanitize=undefined \
  "$ROOT/tests/gridfp_reverse_test.cpp" -o "$OUT/reverse"
"$OUT/reverse"
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra -fsanitize=undefined \
  "$ROOT/tests/shard_address_test.cpp" -o "$OUT/shard"
"$OUT/shard"
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra -fsanitize=undefined \
  "$ROOT/tests/ternary_rank_index_test.cpp" -o "$OUT/ternary"
"$OUT/ternary"
if [[ "${1:-}" == --gpu ]]; then
  nvcc -O3 -std=c++17 -arch="${ARCH:-native}" \
    "$ROOT/tests/shard_address_gpu_test.cu" -o "$OUT/shard-gpu" \
    >"$OUT/shard.build.log" 2>&1
  "$OUT/shard-gpu"
  nvcc -O3 -std=c++17 -arch="${ARCH:-native}" \
    -DTARGET_W=19 -DLOW_LUT_K=9 -DHIGH_LUT_K=9 \
    "$ROOT/tests/factor_rank_reuse_test.cu" -o "$OUT/rank-reuse" \
    >"$OUT/build.log" 2>&1
  "$OUT/rank-reuse"
  nvcc -O3 -std=c++17 -arch="${ARCH:-native}" \
    -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
    "$ROOT/tests/frontier_graph_lifetime_test.cu" -o "$OUT/graph-lifetime" \
    >"$OUT/graph.build.log" 2>&1
  "$OUT/graph-lifetime"
  nvcc -O3 -std=c++17 -arch="${ARCH:-native}" \
    -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
    "$ROOT/tests/factor_transfer_map_test.cu" -o "$OUT/transfer" \
    >"$OUT/transfer.build.log" 2>&1
  "$OUT/transfer"
  nvcc -O3 -std=c++17 -arch="${ARCH:-native}" \
    -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
    "$ROOT/tests/factor_transfer_batch_test.cu" -o "$OUT/transfer-batch" \
    >"$OUT/transfer-batch.build.log" 2>&1
  "$OUT/transfer-batch"
fi
