#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
W="${W:-10}"
N="${N:-21}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RUN_BUILD_SMOKE="${RUN_BUILD_SMOKE:-1}"

[[ "$RUN_BUILD_SMOKE" == 0 || "$RUN_BUILD_SMOKE" == 1 ]] || { echo "RUN_BUILD_SMOKE must be 0 or 1" >&2; exit 2; }

echo '=== rankformula W28 exhaustive formula proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankformula-plan.sh"
echo '=== rankformula sparse-base owner proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankformula-sparse-base-proof.sh"

for sparse in 0 1; do
  for fused in 0 1; do
    echo "=== rankformula exact CUDA: sparse=$sparse fused13=$fused ===" >&2
    RANKFORMULA_SPARSE_BASE="$sparse" RANKDELTA8_FUSED13="$fused" \
      ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" \
      RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-cross5-selftest.sh"
  done
done

if [[ "$RUN_BUILD_SMOKE" == 1 ]]; then
  echo '=== rankformula B300 production compile smoke ===' >&2
  N="$N" ARCH="$ARCH" HIGH_CTX=warpstriped_delta_direct_affine_rankformula_cross5 \
    DEPTHCODE_DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKDELTA8_FUSED13=1 RANKFORMULA_SPARSE_BASE=1 \
    TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 \
    OUT="$ONEESAN_BUILD_DIR/rankformula_preflight_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh"
fi

echo "rankformula-preflight OK w=$W n=$N arch=$ARCH lut=$RANKSTREAM_LUT_LOAD exact_formula_w28=1 sparse_0_1=1 fused13_0_1=1 build_smoke=$RUN_BUILD_SMOKE" >&2
