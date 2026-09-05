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
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-group56-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-group56-coop-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-group61-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-group61-coop-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-warpshare-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-coopgroup-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lut-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lazy-load-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-select8-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-depth4-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-srcpack10-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-compact-proof.sh"

for spec in \
  '4 0 0 1 0 0 0' \
  '8 0 0 1 0 0 0' \
  '8 1 0 1 0 0 0' \
  '8 1 1 1 0 0 0' \
  '16 1 1 1 0 0 0' \
  '16 1 1 0 0 0 0' \
  '16 1 1 0 1 0 0' \
  '16 1 1 0 1 0 1' \
  '16 1 1 0 1 1 1'; do
  read -r block ws coop unroll select8 depth4 srcpack10 <<<"$spec"
  echo "=== abstract nometa exact CUDA block=$block warpshare=$ws coopgroup=$coop unroll=$unroll group56=0 group61=0 select8=$select8 depth4=$depth4 srcpack10=$srcpack10 ===" >&2
  RANKFORMULA_NOMETA_BLOCK="$block" RANKFORMULA_NOMETA_WARPSHARE="$ws" RANKFORMULA_NOMETA_COOPGROUP="$coop" \
    RANKFORMULA_NOMETA_COOP_UNROLL="$unroll" RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=0 \
    RANKFORMULA_ABSTRACT_SELECT8="$select8" RANKFORMULA_ABSTRACT_DEPTH4="$depth4" \
    RANKFORMULA_ABSTRACT_SRCPACK10="$srcpack10" ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-block-selftest.sh"
done

for layout in 56 61; do
  g56=0; g61=0; [[ "$layout" == 56 ]] && g56=1 || g61=1
  echo "=== abstract nometa exact CUDA block=16 coopgroup=1 rolled group${layout}=1 depth4=1 srcpack10=1 ===" >&2
  RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 \
    RANKFORMULA_NOMETA_COOP_UNROLL=0 RANKFORMULA_NOMETA_GROUP56="$g56" RANKFORMULA_NOMETA_GROUP61="$g61" \
    RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 \
    ARCH="$ARCH" W="$W" DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-block-selftest.sh"
done

if [[ "$RUN_BUILD_SMOKE" == 1 ]]; then
  echo '=== abstract nometa B300 production compile smoke block=16 coopgroup=1 rolled group61=1 depth4=1 srcpack10=1 ===' >&2
  N="$N" ARCH="$ARCH" RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 \
    RANKFORMULA_NOMETA_COOP_UNROLL=0 RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
    RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 \
    DEPTHCODE_DECODE_LOAD="$DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 OUT="$ONEESAN_BUILD_DIR/rankformula_nometa_abstract_b16_rolled_group61_depth4_srcpack10_preflight_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh"
fi

echo "rankformula-nometa4-abstract-preflight OK w=$W n=$N metadata_bytes_per_code=0 exact_modes=b4_scalar,b8_scalar,b8_warpshare,b8_coopgroup,b16_coop_unrolled,b16_coop_rolled,b16_coop_rolled_select8,b16_coop_rolled_select8_srcpack10,b16_coop_rolled_depth4_srcpack10,b16_group56_compact,b16_group61_compact packed_group64_bits=59 group56_pack_bits=56 group61_pack_bits=61 group61_direct_end=1 group61_direct_source_base=1 group61_max_end=30114 group61_max_source_base=29113 group61_coop_exact=1201917 group61_coop_successor_loads=65159 support_positions_runtime=0 b16_aux_bytes_all=707406 b16_coopgroup_table_loads_model=215509 b16_avg_early_ballots=2.251382 base_abstract_lut_bytes=94206 depth4_srcpack10_lut_bytes=86058 group56_device_lut_bytes=85578 group61_layout_stage_device_lut_bytes=86058 compact_e2e_proof=1 compact_production_selected=2492769 depth14_15_fast_zero=1 ballot_runtime_loads=0 build_smoke_b16_rolled_group61=$RUN_BUILD_SMOKE" >&2
