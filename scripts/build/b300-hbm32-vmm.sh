#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
if [[ "$N" != 27 ]]; then
  echo "b300-hbm32-vmm.sh is currently specialized for n=27 / W=28" >&2
  exit 2
fi
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-13}"
HIGH_LUT_K="${HIGH_LUT_K:-13}"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_hbm32_vmm_n${N}.cu}"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_vmm_n${N}}")"

command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC"

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -I"$ONEESAN_ROOT/src/cuda/b300" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DB300_FAST_SHARD_ADDRESS8=0 \
  "$GENSRC" -lcuda -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  generated_source=$GENSRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
echo "  authoritative_storage=contiguous_multi_gpu_vmm"
echo "  gather_scatter_addressing=direct_global_index"
echo "  interval_io=direct_global_index_shard_free"
echo "  logical_views=host_init_answer_only"
