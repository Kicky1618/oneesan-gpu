#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/oneesan-groupbatch-test.XXXXXX")}"
mkdir -p "$OUT"
echo "Test artifacts: $OUT"
for n in 9 13; do
  low=$(((n+1)/2))
  nvcc -O3 -std=c++17 -lineinfo -arch="${ARCH:-native}" \
    -DTARGET_W=$((n+1)) -DLOW_LUT_K="$low" -DHIGH_LUT_K=$((n-low)) \
    "$ROOT/tests/groupbatch_operator_test.cu" -lcuda -o "$OUT/operator-n$n" \
    >"$OUT/build-n$n.log" 2>&1
done
for graph in 0 1 2; do
  for wrap in 0 1; do
    for match in 0 1; do
      for dest in 0 1; do
        log="$OUT/oracle-$graph-$wrap-$match-$dest.log"
        GRIDFP_GROUPBATCH_GRAPH="$graph" GRIDFP_GROUPBATCH_WRAP32="$wrap" \
        GRIDFP_GROUPBATCH_MATCHLUT="$match" GRIDFP_GROUPBATCH_P1_DEST="$dest" \
          "$OUT/operator-n9" >"$log" 2>&1
        tail -n 1 "$log"
      done
    done
  done
done
for mode in 0 1; do
  log="$OUT/oracle-n13-$mode.log"
  GRIDFP_GROUPBATCH_GRAPH=$((mode+1)) GRIDFP_GROUPBATCH_WRAP32="$mode" \
  GRIDFP_GROUPBATCH_MATCHLUT="$mode" GRIDFP_GROUPBATCH_P1_DEST=$((1-mode)) \
    "$OUT/operator-n13" >"$log" 2>&1
  tail -n 1 "$log"
done
if [[ "${MEMCHECK:-0}" == 1 ]]; then
  GRIDFP_GROUPBATCH_GRAPH=2 GRIDFP_GROUPBATCH_WRAP32=1 \
  GRIDFP_GROUPBATCH_MATCHLUT=1 GRIDFP_GROUPBATCH_P1_DEST=0 \
    compute-sanitizer --tool memcheck --error-exitcode 9 "$OUT/operator-n13" \
    >"$OUT/memcheck.log" 2>&1
  tail -n 2 "$OUT/memcheck.log"
fi
