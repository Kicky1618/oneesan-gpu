#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";[[ "$N" == 27 ]]||exit 2;W=$((N+1));NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";LOW_LUT_K="${LOW_LUT_K:-13}";HIGH_LUT_K="${HIGH_LUT_K:-13}"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_cross_window_n27.cu}";OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_vmm_static_lpt_cross_window_n27}")"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 cross-device window barrier"
OUT="$GENSRC" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-cross-window-production-generate-proof.sh"
TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" "$GENSRC" -lcuda -o "$OUT"
echo "built $OUT"
echo "  scheduler=static_lpt persistent_workers=8 window_batching=1 cross_device_window_events=1"
echo "  per_group_host_syncs=0 expected_host_gpu_sync_calls=56 old_windowbatch_calls=448"
echo "  metadata_replication=0 per_group_meta_copy_bytes=0 per_group_interval_h2d=0"
echo "  row_limit_env=B300_ROW_LIMIT default_rows=28"
