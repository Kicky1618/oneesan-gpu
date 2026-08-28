#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; W="${W:-10}"; N="${N:-21}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"; RUN_BUILD_SMOKE="${RUN_BUILD_SMOKE:-1}"
[[ "$RUN_BUILD_SMOKE" == 0 || "$RUN_BUILD_SMOKE" == 1 ]] || exit 2

bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-blocks-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa4-group64-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-warpshare-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lut-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lazy-load-proof.sh"

for spec in '4 0' '8 0' '8 1'; do
  read -r block ws <<<"$spec"
  echo "=== abstract nometa exact CUDA block=$block warpshare=$ws ===" >&2
  RANKFORMULA_NOMETA_BLOCK="$block" RANKFORMULA_NOMETA_WARPSHARE="$ws" \
    ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-block-selftest.sh"
done

if [[ "$RUN_BUILD_SMOKE" == 1 ]]; then
  echo '=== abstract nometa B300 production compile smoke block=8 warpshare=1 ===' >&2
  N="$N" ARCH="$ARCH" RANKFORMULA_NOMETA_BLOCK=8 RANKFORMULA_NOMETA_WARPSHARE=1 \
    DEPTHCODE_DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 OUT="$ONEESAN_BUILD_DIR/rankformula_nometa_abstract_b8_warpshare_preflight_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh"
fi

echo "rankformula-nometa4-abstract-preflight OK w=$W n=$N metadata_bytes_per_code=0 exact_modes=b4_scalar,b8_scalar,b8_warpshare packed_group64=1 support_positions_runtime=0 b4_aux_bytes_all=1158104 b8_aux_bytes_all=857642 b8_warpshare_table_loads_model=544003 abstract_lut_bytes=94206 ballot_runtime_loads=0 lazy_source_loads=1 build_smoke_b8_warpshare=$RUN_BUILD_SMOKE" >&2
