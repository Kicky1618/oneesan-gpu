#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
RUN_B300_BUILD="${RUN_B300_BUILD:-0}"
N="${N:-27}"
for x in RUN_B300_BUILD; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done

echo '=== rankdelta8 exact W28 host proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankdelta8-plan.sh"

for align in 0 1; do
  for fused in 0 1; do
    echo "=== rankdelta8 CUDA exact align32=$align fused16=$fused ===" >&2
    RANKDELTA8_ALIGN32="$align" RANKCHUNK32_FUSED16="$fused" \
      RANKSTREAM_LUT_LOAD=ldg DECODE_LOAD=ldg PM_ACCUM=0 ARCH="$ARCH" W=10 \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankdelta8-cross5-selftest.sh"
  done
done

if [[ "$RUN_B300_BUILD" == 1 ]]; then
  for align in 0 1; do
    echo "=== rankdelta8 B300 build align32=$align ===" >&2
    N="$N" ARCH="$ARCH" HIGH_CTX=warpstriped_delta_direct_affine_rankdelta8_cross5 \
      RANKDELTA8_ALIGN32="$align" RANKCHUNK32_FUSED16=1 \
      DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg TRANSPOSE_MODE=pipeline \
      PTXAS_VERBOSE=1 \
      bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh"
  done
fi

echo "rankdelta8-preflight OK arch=$ARCH run_b300_build=$RUN_B300_BUILD host_w28_exact=1 cuda_align_modes=2 cuda_fused_modes=2" >&2
