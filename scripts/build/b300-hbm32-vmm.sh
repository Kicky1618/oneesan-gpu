#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
if [[ "$N" != 27 ]]; then
  echo "b300-hbm32-vmm.sh is currently specialized for n=27 / W=28" >&2
  exit 2
fi
W=$((N + 1))
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-13}"
HIGH_LUT_K="${HIGH_LUT_K:-13}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-0}"
for name in MAIN_MATE_CACHE MAIN_PULL; do
  value="${!name}"
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
if [[ "$MAIN_PULL" == 1 && "$MAIN_MATE_CACHE" != 1 ]]; then
  echo "MAIN_PULL=1 requires MAIN_MATE_CACHE=1" >&2
  exit 2
fi

SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_hbm32_vmm_n${N}.cu}"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_vmm_n${N}}")"

require_nvcc_version_at_least "$NVCC" 13 0 "B300 sm_103/VMM production"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-generate-proof.sh"
if [[ "$MAIN_PULL" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-main-pull-production-generate-proof.sh"
fi

BUILD_SRC="$SRC"
if [[ "$MAIN_MATE_CACHE" == 1 ]]; then
  MATE_SRC="$ONEESAN_BUILD_DIR/generated_b300_hbm32_vmm_n${N}_main_mate.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$BUILD_SRC" "$MATE_SRC"
  BUILD_SRC="$MATE_SRC"
fi
if [[ "$MAIN_PULL" == 1 ]]; then
  PULL_SRC="$ONEESAN_BUILD_DIR/generated_b300_hbm32_vmm_n${N}_main_pull.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$BUILD_SRC" "$PULL_SRC"
  BUILD_SRC="$PULL_SRC"
fi

python3 "$GEN" "$BUILD_SRC" "$GENSRC"
python3 "$PRUNE" "$GENSRC" "$GENSRC"

TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -I"$ONEESAN_ROOT/src/cuda/b300" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$GENSRC" -lcuda -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  transform_source=$BUILD_SRC"
echo "  generated_source=$GENSRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL"
echo "  authoritative_storage=contiguous_multi_gpu_vmm"
echo "  gather_scatter_addressing=direct_global_index"
echo "  interval_io=direct_global_index_shard_free_compact24"
echo "  init_answer_addressing=direct_global_index_device0_context"
echo "  logical_shard_chunks=0"
echo "  logical_shard_views=0"
echo "  legacy_shard_address_scaffolding=0"
echo "  stale_shard_symbols=0"
echo "  stale_width_symbols=0"
if [[ "$MAIN_PULL" == 1 ]]; then
  echo "  p_gt_1_main_update=destination_pull"
  echo "  p_gt_1_main_identity_copy=0"
  echo "  p_gt_1_main_atomic_scatter=0"
  echo "  p_gt_1_blocked_to_main_scatter_kernel=0"
  echo "  pull_vmm_composition_preflight=1"
fi
