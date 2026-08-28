#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; W="${W:-10}"; N="${N:-21}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RUN_BUILD_SMOKE="${RUN_BUILD_SMOKE:-1}"
[[ "$RUN_BUILD_SMOKE" == 0 || "$RUN_BUILD_SMOKE" == 1 ]] || exit 2

echo '=== metadata-free exact host unrank ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-proof.sh"
echo '=== metadata-free locator block bounds ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-blocks-proof.sh"
echo '=== rankformula formula/base structural proofs ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankformula-plan.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-base-delta-proof.sh"

echo '=== nometa4 exact CUDA forward/reverse ===' >&2
ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
  bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-selftest.sh"

if [[ "$RUN_BUILD_SMOKE" == 1 ]]; then
  echo '=== nometa4 B300 production compile smoke ===' >&2
  N="$N" ARCH="$ARCH" DEPTHCODE_DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 OUT="$ONEESAN_BUILD_DIR/rankformula_nometa4_preflight_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4.sh"
fi

echo "rankformula-nometa4-preflight OK w=$W n=$N block=4 metadata_bytes_per_code=0 max_locator_steps=3 ballot_unrank_exact=1 build_smoke=$RUN_BUILD_SMOKE" >&2
