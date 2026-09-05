#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";[[ "$N" == 27 ]]||exit 2;W=$((N+1));NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";LOW_LUT_K="${LOW_LUT_K:-13}";HIGH_LUT_K="${HIGH_LUT_K:-13}"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_persistent_workers_n27.cu}";OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_vmm_static_lpt_persistent_workers_n27}")"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 persistent static workers"
OUT="$GENSRC" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-persistent-workers-production-generate-proof.sh"
TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" "$GENSRC" -lcuda -o "$OUT"
echo "built $OUT"
echo "  scheduler=static_lpt persistent_workers=8 window_barrier=1"
echo "  metadata_replication=0 per_group_meta_copy_bytes=0 per_group_interval_h2d=0"
echo "  expected_thread_creations_old=448 expected_thread_creations_new=8 reduction=56x"
echo "  expected_cudaSetDevice_old=917504 expected_cudaSetDevice_new=8 reduction=114688x"
echo "  row_limit_env=B300_ROW_LIMIT default_rows=28"
