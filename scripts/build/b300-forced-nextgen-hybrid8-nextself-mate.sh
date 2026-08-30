#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"; [[ "$N" == 27 ]] || exit 2
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_nextself_mate_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"; NEXTSELF_WIDTH="${NEXTSELF_WIDTH:-8}"; NEXTSELF_DISTANCE="${NEXTSELF_DISTANCE:-1}"
SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$NEXTSELF_WIDTH" in 1|2|4|8);;*) exit 2;; esac
case "$NEXTSELF_DISTANCE" in 1|2|4);;*) exit 2;; esac
for ev in SELF_EVICT MATE_EVICT; do case "${!ev}" in default|normal|last);;*) echo "$ev must be default,normal,last" >&2; exit 2;; esac; done
case "$CLOSURE_BATCH" in 0|2|4);;*) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256);;*) exit 2;; esac
command -v nvcc >/dev/null || exit 2
TAG="mate_w${NEXTSELF_WIDTH}_d${NEXTSELF_DISTANCE}_sev${SELF_EVICT}_mev${MATE_EVICT}_t${HYBRID_THRESHOLD}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_dual${DUALMASK}_cb${CLOSURE_BATCH}_r${MAXRREGCOUNT}_$$"
ISO="${NEXTGEN_MATE_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-mate/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
SELF_BIN="$ISO/self.bin"; SELF_OUT="$ISO/self.out"; SELF_ERR="$ISO/self.err"
N=27 ARCH="$ARCH" OUT="$SELF_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" NEXTSELF_WIDTH="$NEXTSELF_WIDTH" NEXTSELF_DISTANCE="$NEXTSELF_DISTANCE" NEXTSELF_EVICT="$SELF_EVICT" \
  RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$SELF_ERR" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh" >"$SELF_OUT" 2>"$ISO/self.driver.err"
[[ -x "$SELF_BIN" ]] || exit 3
SRC="$(sed -nE 's/^  source_after_all=(.*)$/\1/p' "$SELF_OUT" | tail -n1)"; [[ -n "$SRC" && -f "$SRC" ]] || exit 3
grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2' "$SRC"
NEXT="$ISO/self_plus_mate.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py" "$SRC" "$NEXT" "$NEXTSELF_WIDTH" "$NEXTSELF_DISTANCE" "$MATE_EVICT" >"$ISO/mate.transform.out"
grep -Fq 'b300_mainrec_hybrid8_next_mate_prefetch=1' "$ISO/mate.transform.out"
grep -Fq "evict_priority=$MATE_EVICT" "$ISO/mate.transform.out"
: >"$BUILD_ERR"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || exit 5
echo "built $OUT"
echo "  nextgen_forced=1 recurrence_hybrid_ilp8=1 recurrence_hybrid_ilp8_min_states=$HYBRID_THRESHOLD recurrence_hybrid_ilp8_nextself=1 recurrence_hybrid_ilp8_nextself_width=$NEXTSELF_WIDTH recurrence_hybrid_ilp8_nextself_distance=$NEXTSELF_DISTANCE self_evict=$SELF_EVICT next_mate_prefetch=1 mate_evict=$MATE_EVICT"
echo "  source_self_only=$SRC"
echo "  source_after_all=$NEXT"
echo "  self_builder_proof_gates_reused=1 next_mate_transform=1 eviction_hints=$SELF_EVICT,$MATE_EVICT extra_shared_bytes=0"
echo "  ptxas_log=$BUILD_ERR"
