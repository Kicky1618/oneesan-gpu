#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

COMPILE_SMOKE="${COMPILE_SMOKE:-1}"
N="${N:-21}"; ARCH="${ARCH:-native}"
[[ "$COMPILE_SMOKE" == 0 || "$COMPILE_SMOKE" == 1 ]] || exit 2

bash "$ONEESAN_ROOT/scripts/bench/rankformula-directgather64-proof.sh"
for rel in \
  scripts/build/b300-directgather-colilp-fast.sh \
  scripts/build/b300-directgather64-fast.sh \
  scripts/bench/b300-directgather64-ab.sh \
  scripts/bench/b300-directgather-ncu-high.sh; do
  bash -n "$ONEESAN_ROOT/$rel"
done

if [[ "$COMPILE_SMOKE" == 1 ]]; then
  command -v nvcc >/dev/null || { echo "nvcc required for COMPILE_SMOKE=1" >&2; exit 2; }
  out="$ONEESAN_BUILD_DIR/preflight_directgather64_n${N}"
  N="$N" ARCH="$ARCH" OUT="$out" COL_ILP=1 DEPTHMAJOR=1 DIRECTGATHER64=1 \
    FORCE7=0 MLP_WINDOW4=1 PAIR_MLP=0 PREFETCH_NEXT=0 PM_ACCUM=1 \
    PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"
  [[ -x "$out" ]] || { echo "missing compile smoke binary $out" >&2; exit 3; }
fi

echo "rankformula-directgather64-preflight OK host_pack_exact=1 shell_syntax=1 compile_smoke=$COMPILE_SMOKE" >&2
