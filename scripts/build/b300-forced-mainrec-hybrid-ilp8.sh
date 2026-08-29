#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo 'mainrec hybrid ILP8 build currently targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"
ILP8_MIN_STATES="${ILP8_MIN_STATES:-1048576}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
DUALMASK="${DUALMASK:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
OUT="$(build_path "${OUT:-b300_forced_mainrec_hybrid8_t${ILP8_MIN_STATES}_n27}")"
BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"

[[ "$ILP8_MIN_STATES" =~ ^[0-9]+$ ]] || { echo 'ILP8_MIN_STATES must be non-negative integer' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0/1' >&2; exit 2; }
[[ "$RANDOM_CG" == 0 || "$RANDOM_CG" == 1 ]] || { echo 'RANDOM_CG must be 0/1' >&2; exit 2; }
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) echo 'RANDOM_CG_L2_FETCH_BYTES must be 0,64,128,256' >&2; exit 2;; esac
[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]] || { echo 'DUALMASK must be 0/1' >&2; exit 2; }
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] || { echo 'MAXRREGCOUNT must be non-negative integer' >&2; exit 2; }
[[ "$PTXAS_VERBOSE" == 0 || "$PTXAS_VERBOSE" == 1 ]] || { echo 'PTXAS_VERBOSE must be 0/1' >&2; exit 2; }
if [[ "$RANDOM_CG" == 0 && "$RANDOM_CG_L2_FETCH_BYTES" != 0 ]]; then
  echo 'RANDOM_CG_L2_FETCH_BYTES requires RANDOM_CG=1' >&2; exit 2
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-transform-preflight.sh"

TAG="mainrec_hybrid8_t${ILP8_MIN_STATES}_hd${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_dual${DUALMASK}_r${MAXRREGCOUNT}_$$"
ISO="${HYBRID_BUILD_DIR:-$ONEESAN_BUILD_DIR/hybrid_mainrec/$TAG}"
mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")" "$(dirname "$BUILD_ERR")"
BASE_BIN="$ISO/base_ilp2.bin"
BASE_OUT="$ISO/base.build.out"
BASE_ERR="$ISO/base.build.err"

# Generate the exact production main-recurrence ILP2 source first.  We keep
# DUALMASK off here so hybridization changes only the main recurrence kernel;
# optional cache/block transforms are layered afterward in a deterministic order.
ONEESAN_BUILD_DIR="$ISO/base_gen" ONEESAN_TMP_DIR="$ISO/base_tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK=0 PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$BASE_ERR"
[[ -x "$BASE_BIN" ]] || { echo 'production ILP2 base binary missing' >&2; exit 3; }
BASE_SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"
[[ -n "$BASE_SRC" && -f "$BASE_SRC" ]] || { echo 'could not resolve production recurrence source' >&2; exit 3; }
grep -Fq 'main_recurrence=1' "$BASE_OUT"
grep -Fq 'main_pull_ilp=2' "$BASE_OUT"
grep -Fq 'main_pull_kernel_ilp2' "$BASE_SRC"
grep -Fq 'b300_main_pull_ilp2_blocks(ms.size,threads)' "$BASE_SRC"
grep -Fq 'high_rec_groups=' "$BASE_SRC"

HYBRID_SRC="$ISO/mainrec_hybrid8.cu"
HYBRID_LOG="$ISO/hybrid.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py" \
  "$BASE_SRC" "$HYBRID_SRC" "$ILP8_MIN_STATES" >"$HYBRID_LOG"
grep -Fq 'b300_main_recurrence_hybrid_ilp8=1' "$HYBRID_LOG"
grep -Fq "ilp8_min_states=$ILP8_MIN_STATES" "$HYBRID_LOG"
grep -Fq 'batch_abi_preserved=1' "$HYBRID_LOG"
grep -Fq 'main_pull_kernel_ilp8_hybrid' "$HYBRID_SRC"
grep -Fq "if(ms.size>=Code($ILP8_MIN_STATES))" "$HYBRID_SRC"
grep -Fq 'b300_main_recurrence_ilp8_hybrid_blocks(ms.size,threads)' "$HYBRID_SRC"
grep -Fq 'b300_main_pull_ilp2_blocks(ms.size,threads)' "$HYBRID_SRC"

FINAL_SRC="$HYBRID_SRC"
CG_SRC=""
if [[ "$RANDOM_CG" == 1 ]]; then
  CG_SRC="$ISO/mainrec_hybrid8_cg.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py" \
    "$FINAL_SRC" "$CG_SRC" "$RANDOM_CG_L2_FETCH_BYTES" >"$ISO/cg.transform.out"
  FINAL_SRC="$CG_SRC"
  grep -Fq 'hybrid_policy_consistent=1' "$ISO/cg.transform.out"
  grep -Fq 'b300_mainrec_random_load_cg(in+pj7)' "$FINAL_SRC"
  grep -Fq 'b300_mainrec_random_load_cg(in_block+bj7)' "$FINAL_SRC"
fi

DUAL_SRC=""
if [[ "$DUALMASK" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"
  DUAL_SRC="$ISO/mainrec_hybrid8_dualmask.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$FINAL_SRC" "$DUAL_SRC" >"$ISO/dualmask.transform.out"
  FINAL_SRC="$DUAL_SRC"
  grep -Fq 'b300_block_endpoint_masks(d)' "$FINAL_SRC"
  grep -Fq 'main_pull_kernel_ilp8_hybrid' "$FINAL_SRC"
  grep -Fq "if(ms.size>=Code($ILP8_MIN_STATES))" "$FINAL_SRC"
fi

# The post-transform chain must retain both kernels and one selector.  This is
# also a guard against a later CG/dualmask transform accidentally rewriting away
# only one side of the hybrid.
[[ "$(grep -Fc '__global__ void main_pull_kernel_ilp8_hybrid(' "$FINAL_SRC")" -eq 1 ]] || { echo 'hybrid ILP8 kernel count mismatch' >&2; exit 4; }
[[ "$(grep -Fc "if(ms.size>=Code($ILP8_MIN_STATES))" "$FINAL_SRC")" -eq 1 ]] || { echo 'hybrid selector count mismatch' >&2; exit 4; }
grep -Fq 'main_pull_kernel_ilp2' "$FINAL_SRC"
grep -Fq 'base+=Code(8)*grid' "$FINAL_SRC"
grep -Fq 'const Count pair7=' "$FINAL_SRC"
grep -Fq 'const Count block7=' "$FINAL_SRC"

PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=()
(( MAXRREGCOUNT > 0 )) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
: >"$BUILD_ERR"
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 \
  "$FINAL_SRC" -o "$OUT" >"$ISO/final.compile.out" 2>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'hybrid batch binary missing after nvcc' >&2; exit 3; }

cat "$BASE_OUT"
cat "$HYBRID_LOG"
[[ "$RANDOM_CG" == 0 ]] || cat "$ISO/cg.transform.out"
[[ "$DUALMASK" == 0 ]] || cat "$ISO/dualmask.transform.out"
echo "built $OUT"
echo "  mainrec_hybrid_ilp8=1 base_ilp=2 high_ilp=8 ilp8_min_states=$ILP8_MIN_STATES batch_abi=forced2window_opt_batch"
echo "  high_drop_chunk=$HIGH_DROP_CHUNK random_cg=$RANDOM_CG random_cg_l2_fetch_bytes=$RANDOM_CG_L2_FETCH_BYTES dualmask=$DUALMASK maxrregcount=$MAXRREGCOUNT"
echo "  transform_order=production_mainrec_ilp2,hybrid_ilp8${RANDOM_CG:+,optional_random_cg}${DUALMASK:+,optional_dualmask}"
echo "  source_production=$BASE_SRC"
echo "  source_after_hybrid=$HYBRID_SRC"
[[ -z "$CG_SRC" ]] || echo "  source_after_cg=$CG_SRC"
[[ -z "$DUAL_SRC" ]] || echo "  source_after_dualmask=$DUAL_SRC"
echo "  build_source=$FINAL_SRC"
echo "  ptxas_log=$BUILD_ERR"
