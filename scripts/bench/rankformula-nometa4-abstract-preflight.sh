#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; W="${W:-10}"; N="${N:-21}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"; RUN_BUILD_SMOKE="${RUN_BUILD_SMOKE:-1}"
[[ "$RUN_BUILD_SMOKE" == 0 || "$RUN_BUILD_SMOKE" == 1 ]] || exit 2

bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-blocks-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa4-group64-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-group64-selfindex-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-warpshare-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-coopgroup-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lut-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lazy-load-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-select8-proof.sh"

for spec in \
  '4 0 0 1 0' \
  '8 0 0 1 0' \
  '8 1 0 1 0' \
  '8 1 1 1 0' \
  '16 1 1 1 0' \
  '16 1 1 0 0' \
  '16 1 1 0 1'; do
  read -r block ws coop unroll select8 <<<"$spec"
  echo "=== abstract nometa exact CUDA block=$block warpshare=$ws coopgroup=$coop unroll=$unroll select8=$select8 ===" >&2
  RANKFORMULA_NOMETA_BLOCK="$block" RANKFORMULA_NOMETA_WARPSHARE="$ws" RANKFORMULA_NOMETA_COOPGROUP="$coop" \
    RANKFORMULA_NOMETA_COOP_UNROLL="$unroll" RANKFORMULA_ABSTRACT_SELECT8="$select8" \
    ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-block-selftest.sh"
done

if [[ "$RUN_BUILD_SMOKE" == 1 ]]; then
  echo '=== abstract nometa B300 production compile smoke block=16 coopgroup=1 rolled select8=1 ===' >&2
  N="$N" ARCH="$ARCH" RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 \
    RANKFORMULA_NOMETA_COOP_UNROLL=0 RANKFORMULA_ABSTRACT_SELECT8=1 \
    DEPTHCODE_DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 OUT="$ONEESAN_BUILD_DIR/rankformula_nometa_abstract_b16_rolled_select8_preflight_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh"
fi

echo "rankformula-nometa4-abstract-preflight OK w=$W n=$N metadata_bytes_per_code=0 exact_modes=b4_scalar,b8_scalar,b8_warpshare,b8_coopgroup,b16_coop_unrolled,b16_coop_rolled,b16_coop_rolled_select8 packed_group64_bits=59 self_group_index=1 support_positions_runtime=0 b16_aux_bytes_all=707406 b16_coopgroup_table_loads_model=215509 b16_avg_early_ballots=2.251382 base_abstract_lut_bytes=94206 select8_abstract_lut_bytes=171866 select8_table_bytes=91780 depth14_15_fast_zero=1 ballot_runtime_loads=0 build_smoke_b16_rolled_select8=$RUN_BUILD_SMOKE" >&2
