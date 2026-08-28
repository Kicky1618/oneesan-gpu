#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; W="${W:-10}"; N="${N:-21}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"; RUN_BUILD_SMOKE="${RUN_BUILD_SMOKE:-1}"
[[ "$RUN_BUILD_SMOKE" == 0 || "$RUN_BUILD_SMOKE" == 1 ]] || exit 2
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-blocks-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa4-group64-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lut-proof.sh"
ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
  bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-selftest.sh"
if [[ "$RUN_BUILD_SMOKE" == 1 ]]; then
  N="$N" ARCH="$ARCH" DEPTHCODE_DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 OUT="$ONEESAN_BUILD_DIR/rankformula_nometa4_abstract_preflight_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh"
fi
echo "rankformula-nometa4-abstract-preflight OK w=$W n=$N metadata_bytes_per_code=0 block=4 packed_group64=1 group_table_bytes_all=1158104 abstract_lut_bytes=94206 ballot_runtime_loads=0 build_smoke=$RUN_BUILD_SMOKE" >&2
