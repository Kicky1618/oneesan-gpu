#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo 'self/mate geometry builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"
OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_self_mate_geometry_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
SELF_WIDTH="${SELF_WIDTH:-8}"
SELF_DISTANCE="${SELF_DISTANCE:-1}"
MATE_WIDTH="${MATE_WIDTH:-$SELF_WIDTH}"
MATE_DISTANCE="${MATE_DISTANCE:-$SELF_DISTANCE}"
SELF_EVICT="${SELF_EVICT:-default}"
MATE_EVICT="${MATE_EVICT:-default}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
PREFETCH_L2="${PREFETCH_L2:-0}"
DUALMASK="${DUALMASK:-0}"
CLOSURE_BATCH="${CLOSURE_BATCH:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"

for name in SELF_WIDTH MATE_WIDTH; do
  case "${!name}" in 1|2|4|8) ;; *) echo "$name must be one of 1,2,4,8" >&2; exit 2;; esac
done
for name in SELF_DISTANCE MATE_DISTANCE; do
  case "${!name}" in 1|2|4) ;; *) echo "$name must be one of 1,2,4" >&2; exit 2;; esac
done
for name in SELF_EVICT MATE_EVICT; do
  case "${!name}" in default|normal|last) ;; *) echo "$name must be default,normal,last" >&2; exit 2;; esac
done
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || exit 2
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="selfw${SELF_WIDTH}_selfd${SELF_DISTANCE}_matew${MATE_WIDTH}_mated${MATE_DISTANCE}_sev${SELF_EVICT}_mev${MATE_EVICT}_t${HYBRID_THRESHOLD}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_dual${DUALMASK}_cb${CLOSURE_BATCH}_r${MAXRREGCOUNT}_$$"
ISO="${SELF_MATE_GEOMETRY_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-self-mate-geometry/$TAG}"
mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
SELF_BIN="$ISO/self.bin"
SELF_OUT="$ISO/self.build.out"
SELF_ERR="$ISO/self.build.err"

N=27 ARCH="$ARCH" OUT="$SELF_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" \
  NEXTSELF_WIDTH="$SELF_WIDTH" NEXTSELF_DISTANCE="$SELF_DISTANCE" NEXTSELF_EVICT="$SELF_EVICT" \
  RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" \
  DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  BUILD_ERR="$SELF_ERR" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh" \
  >"$SELF_OUT" 2>"$ISO/self.driver.err"
[[ -x "$SELF_BIN" ]] || { echo 'self geometry binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  source_after_all=(.*)$/\1/p' "$SELF_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'self geometry source missing' >&2; exit 3; }
grep -Fq "recurrence_hybrid_ilp8_nextself_width=$SELF_WIDTH recurrence_hybrid_ilp8_nextself_distance=$SELF_DISTANCE" "$SELF_OUT" || exit 3
grep -Fq "recurrence_hybrid_ilp8_nextself_evict=$SELF_EVICT" "$SELF_OUT" || exit 3

NEXT="$ISO/self_plus_mate.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py" \
  "$SRC" "$NEXT" "$MATE_WIDTH" "$MATE_DISTANCE" "$MATE_EVICT" >"$ISO/mate.transform.out"
grep -Fq 'b300_mainrec_hybrid8_next_mate_prefetch=1' "$ISO/mate.transform.out" || exit 3
grep -Fq "prefetch_width=$MATE_WIDTH" "$ISO/mate.transform.out" || exit 3
grep -Fq "prefetch_distance_iterations=$MATE_DISTANCE" "$ISO/mate.transform.out" || exit 3
grep -Fq "evict_priority=$MATE_EVICT" "$ISO/mate.transform.out" || exit 3

: >"$BUILD_ERR"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13)
[[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'self/mate geometry binary missing' >&2; exit 5; }

echo "built $OUT"
echo "  nextgen_forced=1 recurrence_hybrid_ilp8=1 recurrence_hybrid_ilp8_min_states=$HYBRID_THRESHOLD recurrence_hybrid_ilp8_nextself=1"
echo "  self_geometry width=$SELF_WIDTH distance=$SELF_DISTANCE evict=$SELF_EVICT"
echo "  mate_geometry width=$MATE_WIDTH distance=$MATE_DISTANCE evict=$MATE_EVICT"
echo "  geometry_decoupled=1 source_self_only=$SRC source_after_all=$NEXT"
echo "  self_builder_proof_gates_reused=1 next_mate_transform=1 extra_shared_bytes=0"
echo "  ptxas_log=$BUILD_ERR"
