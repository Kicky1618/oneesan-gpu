#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'distance builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_nextself_distance_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
NEXTSELF_WIDTH="${NEXTSELF_WIDTH:-8}"; NEXTSELF_DISTANCE="${NEXTSELF_DISTANCE:-1}"; NEXTSELF_EVICT="${NEXTSELF_EVICT:-default}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; PREFETCH_L2="${PREFETCH_L2:-0}"
DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$NEXTSELF_WIDTH" in 1|2|4|8) ;; *) echo 'NEXTSELF_WIDTH must be 1,2,4,8' >&2; exit 2;; esac
case "$NEXTSELF_DISTANCE" in 1|2|4) ;; *) echo 'NEXTSELF_DISTANCE must be 1,2,4' >&2; exit 2;; esac
case "$NEXTSELF_EVICT" in default|normal|last) ;; *) echo 'NEXTSELF_EVICT must be default,normal,last' >&2; exit 2;; esac
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
[[ "$RANDOM_CG" == 1 || "$RANDOM_CG_L2_FETCH_BYTES" == 0 ]] || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="hybns_w${NEXTSELF_WIDTH}_d${NEXTSELF_DISTANCE}_ev${NEXTSELF_EVICT}_h${HIGH_DROP_CHUNK}_t${HYBRID_THRESHOLD}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_dual${DUALMASK}_cb${CLOSURE_BATCH}_r${MAXRREGCOUNT}_$$"
ISO="${NEXTGEN_DISTANCE_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-distance/$TAG}"
mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
BASE_BIN="$ISO/base.bin"; BASE_OUT="$ISO/base.out"; BASE_ERR="$ISO/base.err"

N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP=2 \
  RECURRENCE_HYBRID_ILP8=1 RECURRENCE_HYBRID_ILP8_MIN_STATES="$HYBRID_THRESHOLD" \
  RECURRENCE_HYBRID_ILP8_NEXTSELF=0 RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" \
  DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$BASE_ERR" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$BASE_OUT" 2>"$ISO/base.driver.err"
[[ -x "$BASE_BIN" ]] || { echo 'distance builder canonical base binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  source_after_all=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'distance builder canonical source missing' >&2; exit 3; }

grep -Fq 'main_pull_kernel_ilp8_hybrid' "$SRC"
NEXT="$ISO/hybrid8_nextself_distance.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py" "$SRC" "$NEXT" "$NEXTSELF_WIDTH" "$NEXTSELF_DISTANCE" "$NEXTSELF_EVICT" >"$ISO/transform.out"
grep -Fq "prefetch_width=$NEXTSELF_WIDTH" "$ISO/transform.out"
grep -Fq "prefetch_distance_iterations=$NEXTSELF_DISTANCE" "$ISO/transform.out"
grep -Fq "evict_priority=$NEXTSELF_EVICT" "$ISO/transform.out"

: >"$BUILD_ERR"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'distance builder final binary missing' >&2; exit 5; }

echo "built $OUT"
echo "  nextgen_forced=1 recurrence_hybrid_ilp8=1 recurrence_hybrid_ilp8_min_states=$HYBRID_THRESHOLD recurrence_hybrid_ilp8_nextself=1 recurrence_hybrid_ilp8_nextself_width=$NEXTSELF_WIDTH recurrence_hybrid_ilp8_nextself_distance=$NEXTSELF_DISTANCE recurrence_hybrid_ilp8_nextself_evict=$NEXTSELF_EVICT"
echo "  high_drop_chunk=$HIGH_DROP_CHUNK random_cg=$RANDOM_CG random_cg_l2_fetch_bytes=$RANDOM_CG_L2_FETCH_BYTES prefetch_l2=$PREFETCH_L2 dualmask=$DUALMASK closure_batch=$CLOSURE_BATCH maxrregcount=$MAXRREGCOUNT"
echo "  source_before_distance=$SRC"
echo "  source_after_all=$NEXT"
echo "  canonical_nextgen_proof_gates_reused=1 experimental_distance_transform=1 eviction_hint=$NEXTSELF_EVICT extra_state_bytes=0"
echo "  ptxas_log=$BUILD_ERR"
