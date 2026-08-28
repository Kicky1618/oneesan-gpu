#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
ARCH="${ARCH:-native}"
RUN_PTXAS="${RUN_PTXAS:-1}"
RUN_B300_AB="${RUN_B300_AB:-0}"
RUN_DELTA_AB="${RUN_DELTA_AB:-0}"
RUN_CROSS5_AB="${RUN_CROSS5_AB:-0}"
RUN_DIRECT_AB="${RUN_DIRECT_AB:-0}"
RUN_AFFINE_AB="${RUN_AFFINE_AB:-0}"
RUN_PREKEY_AB="${RUN_PREKEY_AB:-0}"
RUN_RANK16_AB="${RUN_RANK16_AB:-0}"
RUN_RANK_BACKENDS_AB="${RUN_RANK_BACKENDS_AB:-0}"
AB_N="${AB_N:-21}"
REPEATS="${REPEATS:-3}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"

for x in RUN_PTXAS RUN_B300_AB RUN_DELTA_AB RUN_CROSS5_AB RUN_DIRECT_AB RUN_AFFINE_AB RUN_PREKEY_AB RUN_RANK16_AB RUN_RANK_BACKENDS_AB; do
  v="${!x}"
  if [[ "$v" != 0 && "$v" != 1 ]]; then echo "$x must be 0 or 1" >&2; exit 2; fi
done
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac

echo '=== host schedule proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-warpstriped-schedule-selftest.sh"
echo '=== W28/all-legal 10-bit bound ===' >&2
N="$N" bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-bound.sh"
echo '=== all-legal closure ternary-delta proof ===' >&2
N="$N" bash "$ONEESAN_ROOT/scripts/bench/closure-ternary-delta-proof.sh"
echo '=== CROSS5 automaton proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/cross5-automaton-proof.sh"
echo '=== W28 LOW rank-transition/owner/packing proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/low-rank16-plan.sh"
echo '=== rankstream32 warp block-base sharing proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankstream32-warpbase-proof.sh"
echo '=== rankchunk32 K14 layout/prefix/shuffle proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
echo '=== rankchunk32 K14 CUDA device codec: compact + bytepack ===' >&2
ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-codec-cuda-selftest.sh"
echo '=== CROSS5 CUDA helper equivalence, including prekey: PM 0/1 ===' >&2
PM_ACCUM=0 ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/cross5-cuda-selftest.sh"
PM_ACCUM=1 ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/cross5-cuda-selftest.sh"
echo '=== direct depthcode build-plan invariants ===' >&2
N="$N" ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-build-plan.sh"
echo '=== stable CUDA reference matrix ===' >&2
ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-selftest-matrix.sh"
echo '=== direct-resolve CROSS5 CUDA matrix ===' >&2
ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-direct-cross5-selftest-matrix.sh"

echo '=== affine/prekey/rank16/rankstream/rankstream32/rankchunk32 CUDA matrices ===' >&2
for pm in 0 1; do
  for load in global ldg; do
    PM_ACCUM="$pm" DECODE_LOAD="$load" ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-affine-cross5-selftest.sh"
    PM_ACCUM="$pm" DECODE_LOAD="$load" ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-prekey-cross5-selftest.sh"
    PM_ACCUM="$pm" DECODE_LOAD="$load" ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rank16-cross5-selftest.sh"
    PM_ACCUM="$pm" DECODE_LOAD="$load" ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankstream-cross5-selftest.sh"
    PM_ACCUM="$pm" DECODE_LOAD="$load" ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankstream32-cross5-selftest.sh"
    for ranklut in constant ldg ldg256; do
      PM_ACCUM="$pm" DECODE_LOAD="$load" RANKSTREAM_LUT_LOAD="$ranklut" RUN_LAYOUT_PROOF=0 \
        ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh"
    done
  done
done

echo '=== rankchunk32 bytepack full HIGH smoke ===' >&2
PM_ACCUM=0 DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg RANKCHUNK32_BYTEPACK=1 RUN_LAYOUT_PROOF=0 \
  ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh"

if [[ "$RUN_PTXAS" == 1 ]]; then
  echo '=== ptxas resource comparison, including rankstream32/rankchunk32 ===' >&2
  N="$N" ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-highctx-ptxas.sh"
fi
if [[ "$RUN_DELTA_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-resolved-delta-ab.sh"; fi
if [[ "$RUN_CROSS5_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-cross5-ab.sh"; fi
if [[ "$RUN_DIRECT_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-direct-resolve-ab.sh"; fi
if [[ "$RUN_AFFINE_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-affine-row-ab.sh"; fi
if [[ "$RUN_PREKEY_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-prekey-ab.sh"; fi
if [[ "$RUN_RANK16_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-rank16-ab.sh"; fi
if [[ "$RUN_RANK_BACKENDS_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-rank-backends-ab.sh"; fi
if [[ "$RUN_B300_AB" == 1 ]]; then N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-warpstriped-ab.sh"; fi

echo "b300-depthcode-warpstriped-preflight OK n=$N arch=$ARCH run_ptxas=$RUN_PTXAS run_delta_ab=$RUN_DELTA_AB run_cross5_ab=$RUN_CROSS5_AB run_direct_ab=$RUN_DIRECT_AB run_affine_ab=$RUN_AFFINE_AB run_prekey_ab=$RUN_PREKEY_AB run_rank16_ab=$RUN_RANK16_AB run_rank_backends_ab=$RUN_RANK_BACKENDS_AB run_b300_ab=$RUN_B300_AB ternary_delta_proved=1 cross5_cuda=1 low_rank_owner_proved=1 rankstream32_pack_proved=1 rankstream32_warpbase_proved=1 rankchunk32_warpbase_proved=1 rankchunk32_bytepack_prefix_proved=1 rankchunk32_k14_codec_cuda_both=1 rankchunk32_bytepack_high_cuda=1 direct_resolve_cuda=1 affine_rows_cuda=1 prekey_cuda=1 rank16_cuda=1 rankstream_cuda=1 rankstream32_cuda=1 rankchunk32_cuda=1 rankchunk32_ldg256_cuda=1" >&2