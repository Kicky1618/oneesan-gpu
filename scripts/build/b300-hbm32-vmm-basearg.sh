#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"
[[ "$N" == 27 ]] || { echo "b300-hbm32-vmm-basearg.sh is specialized for n=27 / W=28" >&2; exit 2; }
W=$((N+1));ARCH="${ARCH:-native}";LOW_LUT_K="${LOW_LUT_K:-13}";HIGH_LUT_K="${HIGH_LUT_K:-13}"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
LOWER="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_hbm32_vmm_basearg_n${N}.cu}"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_vmm_basearg_n${N}}")"
command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC"
python3 "$PRUNE" "$GENSRC" "$GENSRC"
python3 "$LOWER" "$GENSRC" "$GENSRC"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$GENSRC" -lcuda -o "$OUT"
echo "built $OUT"
echo "  generated_source=$GENSRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
echo "  authoritative_storage=contiguous_multi_gpu_vmm"
echo "  vmm_base_source=kernel_param"
echo "  vmm_base_symbols=0"
echo "  vmm_base_symbol_copies=0"
echo "  interval_io=direct_global_index_shard_free_compact24"
echo "  logical_shard_chunks=0 logical_shard_views=0"
