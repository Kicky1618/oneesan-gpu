#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'mate-load policy builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"
POLICY="${MATE_LOAD_POLICY:-${POLICY:-cg}}"
OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_mate_load_${POLICY}_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
SELF_WIDTH="${SELF_WIDTH:-8}"; SELF_DISTANCE="${SELF_DISTANCE:-1}"; MATE_WIDTH="${MATE_WIDTH:-$SELF_WIDTH}"; MATE_DISTANCE="${MATE_DISTANCE:-$SELF_DISTANCE}"
SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$POLICY" in cg|cs) ;; *) echo 'MATE_LOAD_POLICY must be cg or cs' >&2; exit 2;; esac
for name in SELF_WIDTH MATE_WIDTH; do case "${!name}" in 1|2|4|8) ;; *) exit 2;; esac; done
for name in SELF_DISTANCE MATE_DISTANCE; do case "${!name}" in 1|2|4) ;; *) exit 2;; esac; done
for name in SELF_EVICT MATE_EVICT; do case "${!name}" in default|normal|last) ;; *) exit 2;; esac; done
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || exit 2
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="p${POLICY}_selfw${SELF_WIDTH}_selfd${SELF_DISTANCE}_matew${MATE_WIDTH}_mated${MATE_DISTANCE}_sev${SELF_EVICT}_mev${MATE_EVICT}_t${HYBRID_THRESHOLD}_$$"
ISO="${MATE_LOAD_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-mate-load-policy/$TAG}"
mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
CONTROL_BIN="$ISO/control.bin"; CONTROL_OUT="$ISO/control.build.out"; CONTROL_ERR="$ISO/control.build.err"

# Reuse the fully proven geometry/eviction builder to materialize the exact
# Stage-J/K source, then alter only the eight ILP8 mate loads.
N=27 ARCH="$ARCH" OUT="$CONTROL_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" \
  SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" \
  RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" \
  MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$CONTROL_ERR" SELF_MATE_GEOMETRY_BUILD_DIR="$ISO/geometry" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh" >"$CONTROL_OUT" 2>"$ISO/control.driver.err"
[[ -x "$CONTROL_BIN" ]] || { echo 'mate-load control binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  geometry_decoupled=1 source_self_only=.* source_after_all=(.*)$/\1/p' "$CONTROL_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'mate-load source_after_all missing' >&2; exit 3; }

NEXT="$ISO/mate_load_${POLICY}.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-mate-load-policy.py" "$SRC" "$NEXT" "$POLICY" >"$ISO/load.transform.out"
grep -Fq "b300_mainrec_hybrid8_mate_load_policy=1 policy=$POLICY" "$ISO/load.transform.out" || exit 3

: >"$BUILD_ERR"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'mate-load policy binary missing' >&2; exit 5; }

echo "built $OUT"
echo "  nextgen_forced=1 recurrence_hybrid_ilp8=1 mate_load_policy=$POLICY mate_load_scope=ilp8_only"
echo "  self_geometry width=$SELF_WIDTH distance=$SELF_DISTANCE evict=$SELF_EVICT"
echo "  mate_geometry width=$MATE_WIDTH distance=$MATE_DISTANCE evict=$MATE_EVICT"
echo "  control_bin=$CONTROL_BIN source_before_policy=$SRC source_after_all=$NEXT"
echo "  geometry_builder_proof_gates_reused=1 mate_load_policy_transform=1 ilp2_unchanged=1 mate_writes_unchanged=1"
echo "  ptxas_log=$BUILD_ERR control_ptxas_log=$CONTROL_ERR"
