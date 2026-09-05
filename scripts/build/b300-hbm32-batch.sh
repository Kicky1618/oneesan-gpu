#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
require_uint N "$N" || exit 2
if (( N < 2 || N > 27 )); then echo "N must be in 2..27 for the B300 production solvers" >&2; exit 2; fi
W=$((N + 1))
ARCH="${ARCH:-native}"
ROW6_VMM="${ROW6_VMM:-0}"
ROW6_PRERANK="${ROW6_PRERANK:-0}"
ROW6_OCCVMM="${ROW6_OCCVMM:-0}"
ROW7_TENSOR="${ROW7_TENSOR:-0}"
ROW8_TENSOR="${ROW8_TENSOR:-0}"
OWNERFUSED="${OWNERFUSED:-0}"
BLOCKFUSED="${BLOCKFUSED:-0}"
GROUPBATCH="${GROUPBATCH:-0}"
require_uint ROW6_VMM "$ROW6_VMM" || exit 2
require_uint ROW6_PRERANK "$ROW6_PRERANK" || exit 2
require_uint ROW6_OCCVMM "$ROW6_OCCVMM" || exit 2
require_uint ROW7_TENSOR "$ROW7_TENSOR" || exit 2
require_uint ROW8_TENSOR "$ROW8_TENSOR" || exit 2
require_uint OWNERFUSED "$OWNERFUSED" || exit 2
require_uint BLOCKFUSED "$BLOCKFUSED" || exit 2
require_uint GROUPBATCH "$GROUPBATCH" || exit 2
if (( ROW6_VMM != 0 && ROW6_VMM != 1 )); then echo "ROW6_VMM must be 0 or 1" >&2; exit 2; fi
if (( ROW6_PRERANK != 0 && ROW6_PRERANK != 1 )); then echo "ROW6_PRERANK must be 0 or 1" >&2; exit 2; fi
if (( ROW6_OCCVMM != 0 && ROW6_OCCVMM != 1 )); then echo "ROW6_OCCVMM must be 0 or 1" >&2; exit 2; fi
if (( ROW7_TENSOR != 0 && ROW7_TENSOR != 1 )); then echo "ROW7_TENSOR must be 0 or 1" >&2; exit 2; fi
if (( ROW8_TENSOR != 0 && ROW8_TENSOR != 1 )); then echo "ROW8_TENSOR must be 0 or 1" >&2; exit 2; fi
if (( OWNERFUSED != 0 && OWNERFUSED != 1 )); then echo "OWNERFUSED must be 0 or 1" >&2; exit 2; fi
if (( BLOCKFUSED != 0 && BLOCKFUSED != 1 )); then echo "BLOCKFUSED must be 0 or 1" >&2; exit 2; fi
if (( GROUPBATCH != 0 && GROUPBATCH != 1 )); then echo "GROUPBATCH must be 0 or 1" >&2; exit 2; fi
if (( ROW7_TENSOR && ROW8_TENSOR )); then echo "ROW7_TENSOR and ROW8_TENSOR are mutually exclusive" >&2; exit 2; fi
if (( ROW6_VMM + ROW6_PRERANK + ROW6_OCCVMM > 1 )); then echo "ROW6_VMM, ROW6_PRERANK and ROW6_OCCVMM are mutually exclusive" >&2; exit 2; fi
if (( (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM) && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "ROW6 VMM/pre-rank/occvmm modes are mutually exclusive with ROW7_TENSOR/ROW8_TENSOR" >&2; exit 2; fi
if (( (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM) && N < 27 )); then echo "ROW6 VMM production modes currently require N>=27" >&2; exit 2; fi
if (( OWNERFUSED && (N < 20 || N >= 27) )); then echo "OWNERFUSED production path currently supports N=20..26" >&2; exit 2; fi
if (( BLOCKFUSED && N < 20 )); then echo "BLOCKFUSED production path currently supports N=20..27" >&2; exit 2; fi
if (( GROUPBATCH && N < 20 )); then echo "GROUPBATCH production path currently supports N=20..27" >&2; exit 2; fi
if (( OWNERFUSED + BLOCKFUSED + GROUPBATCH > 1 )); then echo "OWNERFUSED, BLOCKFUSED and GROUPBATCH are mutually exclusive" >&2; exit 2; fi
if (( OWNERFUSED && (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM || ROW7_TENSOR || ROW8_TENSOR) )); then echo "OWNERFUSED is mutually exclusive with other specialized modes" >&2; exit 2; fi
if (( BLOCKFUSED && (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM || ROW7_TENSOR || ROW8_TENSOR) )); then echo "BLOCKFUSED is mutually exclusive with other specialized modes" >&2; exit 2; fi
if (( GROUPBATCH && (ROW6_VMM || ROW6_PRERANK || ROW6_OCCVMM || ROW7_TENSOR || ROW8_TENSOR) )); then echo "GROUPBATCH is mutually exclusive with other specialized modes" >&2; exit 2; fi
NVCC="$(command -v nvcc)"
SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  if (( ROW8_TENSOR )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row8tensor_batch.cu"
  elif (( ROW7_TENSOR )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row7tensor_batch.cu"
  elif (( ROW6_OCCVMM )); then
    SRC="src/cuda/b300/occmajor_authvmm_rank16_batch.cu"
  elif (( ROW6_PRERANK )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_batch.cu"
  elif (( ROW6_VMM )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_vmm_batch.cu"
  elif (( GROUPBATCH )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_groupbatch_batch.cu"
  elif (( BLOCKFUSED )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_batch.cu"
  elif (( OWNERFUSED )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_ownerfused_batch.cu"
  elif (( N >= 27 )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
  else
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN_batch.cu"
  fi
fi
SRC="$(repo_path "$SRC")"
ROW6_CRT20_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu")"
ROW6_VMM_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_vmm_batch.cu")"
ROW6_PRERANK_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_batch.cu")"
OWNERFUSED_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_ownerfused_batch.cu")"
BLOCKFUSED_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_batch.cu")"
GROUPBATCH_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_groupbatch_batch.cu")"
ROW6_OCCVMM_SRC="$(repo_path "src/cuda/b300/occmajor_authvmm_rank16_batch.cu")"
ROW7_TENSOR_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row7tensor_batch.cu")"
ROW8_TENSOR_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row8tensor_batch.cu")"
PROVENANCE_AUX=()
PROVENANCE_COMPILE_EXTRA=()
LINK_ARGS=()
EXTRA_NVCC_ARGS=()
if [[ "$SRC" == "$GROUPBATCH_SRC" && -n "${GROUPBATCH_CHUNK:-}" ]]; then
  require_uint GROUPBATCH_CHUNK "$GROUPBATCH_CHUNK" || exit 2
  if (( GROUPBATCH_CHUNK < 256 )); then echo "GROUPBATCH_CHUNK must be >=256" >&2; exit 2; fi
  EXTRA_NVCC_ARGS+=( -DGROUPBATCH_CHUNK_ELEMS="$GROUPBATCH_CHUNK" )
  PROVENANCE_COMPILE_EXTRA+=( --compile-arg=-DGROUPBATCH_CHUNK_ELEMS="$GROUPBATCH_CHUNK" )
fi
if [[ "$SRC" == "$ROW6_CRT20_SRC" || "$SRC" == "$ROW6_VMM_SRC" || "$SRC" == "$ROW6_PRERANK_SRC" || "$SRC" == "$ROW6_OCCVMM_SRC" || "$SRC" == "$OWNERFUSED_SRC" || "$SRC" == "$BLOCKFUSED_SRC" || "$SRC" == "$GROUPBATCH_SRC" ]]; then
  python3 "$ONEESAN_ROOT/scripts/tools/verify_row6_crt20.py" >/dev/null
  PROVENANCE_AUX+=(
    --auxiliary-dependency row6-crt20-generator "$ONEESAN_ROOT/scripts/tools/gen_row6_crt20.py"
    --auxiliary-dependency row6-crt20-verifier "$ONEESAN_ROOT/scripts/tools/verify_row6_crt20.py"
    --auxiliary-dependency row6-path-bound-source "$ONEESAN_ROOT/scripts/solve/path_bound.py"
    --auxiliary-dependency row6-rational-certificate "$ONEESAN_ROOT/formal/certificates/row6_rational_dump.txt.xz"
  )
fi
if [[ "$SRC" == "$ROW6_VMM_SRC" || "$SRC" == "$ROW6_PRERANK_SRC" || "$SRC" == "$ROW6_OCCVMM_SRC" || "$SRC" == "$OWNERFUSED_SRC" || "$SRC" == "$BLOCKFUSED_SRC" || "$SRC" == "$GROUPBATCH_SRC" ]]; then
  LINK_ARGS+=( -lcuda )
  PROVENANCE_COMPILE_EXTRA+=( --compile-arg=-lcuda )
fi
if [[ "$SRC" == "$ROW7_TENSOR_SRC" ]]; then
  [[ -f "$ONEESAN_ROOT/src/cuda/b300/row7_exact_compact_u128.bin" ]] || { echo "missing row7 exact compact table" >&2; exit 2; }
  LINK_ARGS+=( -lcublas )
  EXTRA_NVCC_ARGS+=( -Xcompiler=-pthread )
  PROVENANCE_COMPILE_EXTRA+=( --compile-arg=-Xcompiler=-pthread --compile-arg=-lcublas )
  PROVENANCE_AUX+=(
    --auxiliary-dependency row7-exact-compact "$ONEESAN_ROOT/src/cuda/b300/row7_exact_compact_u128.bin"
  )
fi
if [[ "$SRC" == "$ROW8_TENSOR_SRC" ]]; then
  [[ -f "$ONEESAN_ROOT/src/cuda/b300/row8_pivots_w19.bin" ]] || { echo "missing row8 pivot certificate" >&2; exit 2; }
  LINK_ARGS+=( -lcublas )
  EXTRA_NVCC_ARGS+=( -Xcompiler=-pthread )
  PROVENANCE_COMPILE_EXTRA+=( --compile-arg=-Xcompiler=-pthread --compile-arg=-lcublas )
  [[ -f "$ONEESAN_ROOT/src/cuda/b300/row8_structural_int_v1.bin" ]] || { echo "missing row8 structural integer cache" >&2; exit 2; }
  [[ -f "$ONEESAN_ROOT/src/cuda/b300/row8_gap01.bin" ]] || { echo "missing row8 gap 0/1 cache" >&2; exit 2; }
  PROVENANCE_AUX+=(
    --auxiliary-dependency row8-pivots-w19 "$ONEESAN_ROOT/src/cuda/b300/row8_pivots_w19.bin"
    --auxiliary-dependency row8-structural-int-v1 "$ONEESAN_ROOT/src/cuda/b300/row8_structural_int_v1.bin"
    --auxiliary-dependency row8-gap01 "$ONEESAN_ROOT/src/cuda/b300/row8_gap01.bin"
  )
fi
DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_batch_n${N}"
if [[ "$SRC" == "$ROW6_VMM_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_vmm_batch_n${N}"; fi
if [[ "$SRC" == "$ROW6_PRERANK_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_vmm_prerank_batch_n${N}"; fi
if [[ "$SRC" == "$OWNERFUSED_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_ownerfused_batch_n${N}"; fi
if [[ "$SRC" == "$BLOCKFUSED_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_blockfused_batch_n${N}"; fi
if [[ "$SRC" == "$GROUPBATCH_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_groupbatch_batch_n${N}"; fi
if [[ "$SRC" == "$ROW6_OCCVMM_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_occvmm_batch_n${N}"; fi
if [[ "$SRC" == "$ROW7_TENSOR_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_row7tensor_batch_n${N}"; fi
if [[ "$SRC" == "$ROW8_TENSOR_SRC" ]]; then DEFAULT_OUT="oneesan_cuda_gridfp_b300_hbm32_row8tensor_batch_n${N}"; fi
OUT="$(build_path "${OUT:-$DEFAULT_OUT}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"

if [[ -z "$LOW_LUT_K" ]]; then
  if [[ "$SRC" == "$ROW7_TENSOR_SRC" || "$SRC" == "$ROW8_TENSOR_SRC" || "$SRC" == "$OWNERFUSED_SRC" || "$SRC" == "$BLOCKFUSED_SRC" || "$SRC" == "$GROUPBATCH_SRC" ]]; then LOW_LUT_K=$(((N + 1) / 2));
  elif (( N >= 27 )); then LOW_LUT_K=14; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if [[ "$SRC" == "$ROW7_TENSOR_SRC" || "$SRC" == "$ROW8_TENSOR_SRC" || "$SRC" == "$OWNERFUSED_SRC" || "$SRC" == "$BLOCKFUSED_SRC" || "$SRC" == "$GROUPBATCH_SRC" ]]; then HIGH_LUT_K=$((N - LOW_LUT_K));
  elif (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "${EXTRA_NVCC_ARGS[@]}" \
  "$SRC" -o "$OUT" "${LINK_ARGS[@]}"

python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" create \
  --root "$ONEESAN_ROOT" \
  --binary "$OUT" \
  --source "$SRC" \
  --compiler "$NVCC" \
  --compile-arg=-O3 \
  --compile-arg=-std=c++17 \
  --compile-arg=-lineinfo \
  --compile-arg=-arch="$ARCH" \
  --compile-arg=-DTARGET_W="$W" \
  --compile-arg=-DLOW_LUT_K="$LOW_LUT_K" \
  --compile-arg=-DHIGH_LUT_K="$HIGH_LUT_K" \
  --compile-arg="$SRC" \
  --compile-arg=-o \
  --compile-arg="$OUT" \
  "${PROVENANCE_COMPILE_EXTRA[@]}" \
  "${PROVENANCE_AUX[@]}" \
  >/dev/null

echo "built $OUT"
echo "  source=$SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
echo "  provenance=${OUT}.provenance.json"
