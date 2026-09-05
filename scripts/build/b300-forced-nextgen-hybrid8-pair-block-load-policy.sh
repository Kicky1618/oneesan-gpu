#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'Stage-N pair/block builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_stagen_n27}")"
MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-default}"; PAIR_LOAD_POLICY="${PAIR_LOAD_POLICY:-default}"; BLOCK_LOAD_POLICY="${BLOCK_LOAD_POLICY:-default}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
SELF_WIDTH="${SELF_WIDTH:-8}"; SELF_DISTANCE="${SELF_DISTANCE:-1}"; MATE_WIDTH="${MATE_WIDTH:-$SELF_WIDTH}"; MATE_DISTANCE="${MATE_DISTANCE:-$SELF_DISTANCE}"
SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"; SELF_GUARD="${SELF_GUARD:-branch}"; MATE_GUARD="${MATE_GUARD:-branch}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; STAGEN_CG_L2_FETCH_BYTES="${STAGEN_CG_L2_FETCH_BYTES:-$RANDOM_CG_L2_FETCH_BYTES}"
PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$MATE_LOAD_POLICY" in default|cg|cs) ;; *) echo 'MATE_LOAD_POLICY must be default,cg,cs' >&2; exit 2;; esac
for p in "$PAIR_LOAD_POLICY" "$BLOCK_LOAD_POLICY"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-N Count load policy=$p" >&2; exit 2;; esac; done
for w in "$SELF_WIDTH" "$MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 2;; esac; done
for d in "$SELF_DISTANCE" "$MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 2;; esac; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) exit 2;; esac; done
for g in "$SELF_GUARD" "$MATE_GUARD"; do case "$g" in branch|predicated) ;; *) exit 2;; esac; done
for x in RANDOM_CG_L2_FETCH_BYTES STAGEN_CG_L2_FETCH_BYTES; do case "${!x}" in 0|64|128|256) ;; *) echo "$x must be 0,64,128,256" >&2; exit 2;; esac; done
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || exit 2
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="mp${MATE_LOAD_POLICY}_pp${PAIR_LOAD_POLICY}_bp${BLOCK_LOAD_POLICY}_sw${SELF_WIDTH}d${SELF_DISTANCE}_mw${MATE_WIDTH}d${MATE_DISTANCE}_sev${SELF_EVICT}_mev${MATE_EVICT}_sg${SELF_GUARD}_mg${MATE_GUARD}_cg${RANDOM_CG}l2${RANDOM_CG_L2_FETCH_BYTES}_nl2${STAGEN_CG_L2_FETCH_BYTES}_$$"
ISO="${STAGEN_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-stage-n/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
UP_BIN="$ISO/upstream.bin"; UP_OUT="$ISO/upstream.build.out"; UP_ERR="$ISO/upstream.build.err"
common_env=(N=27 ARCH="$ARCH" OUT="$UP_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" SELF_GUARD="$SELF_GUARD" MATE_GUARD="$MATE_GUARD" RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$UP_ERR")
if [[ "$MATE_LOAD_POLICY" == default ]]; then
  env "${common_env[@]}" SELF_MATE_GEOMETRY_BUILD_DIR="$ISO/upstream-geometry" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh" >"$UP_OUT" 2>"$ISO/upstream.driver.err"
else
  env "${common_env[@]}" MATE_LOAD_POLICY="$MATE_LOAD_POLICY" MATE_LOAD_BUILD_DIR="$ISO/upstream-mateload" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-mate-load-policy.sh" >"$UP_OUT" 2>"$ISO/upstream.driver.err"
fi
[[ -x "$UP_BIN" ]] || { echo 'Stage-N upstream binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  .*source_after_all=(.*)$/\1/p' "$UP_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'Stage-N upstream source_after_all missing' >&2; exit 3; }
grep -Fq "self_geometry width=$SELF_WIDTH distance=$SELF_DISTANCE evict=$SELF_EVICT guard=$SELF_GUARD" "$UP_OUT" || exit 3
grep -Fq "mate_geometry width=$MATE_WIDTH distance=$MATE_DISTANCE evict=$MATE_EVICT guard=$MATE_GUARD" "$UP_OUT" || exit 3
if [[ "$MATE_LOAD_POLICY" != default ]]; then grep -Fq "mate_load_policy=$MATE_LOAD_POLICY mate_load_scope=ilp8_only" "$UP_OUT" || exit 3; fi

NEXT="$ISO/stagen.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-load-policy.py" "$SRC" "$NEXT" "$PAIR_LOAD_POLICY" "$BLOCK_LOAD_POLICY" "$STAGEN_CG_L2_FETCH_BYTES" >"$ISO/stagen.transform.out"
grep -Fq "b300_mainrec_stagen_pair_block_policy=1 pair_policy=$PAIR_LOAD_POLICY block_policy=$BLOCK_LOAD_POLICY" "$ISO/stagen.transform.out" || exit 3
: >"$BUILD_ERR"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'Stage-N binary missing' >&2; exit 5; }
echo "built $OUT"
echo "  nextgen_forced=1 stage_n_pair_block_load=1 mate_load_policy=$MATE_LOAD_POLICY"
echo "  pair_load_policy=$PAIR_LOAD_POLICY block_load_policy=$BLOCK_LOAD_POLICY stagen_cg_l2_fetch_bytes=$STAGEN_CG_L2_FETCH_BYTES"
echo "  self_geometry width=$SELF_WIDTH distance=$SELF_DISTANCE evict=$SELF_EVICT guard=$SELF_GUARD"
echo "  mate_geometry width=$MATE_WIDTH distance=$MATE_DISTANCE evict=$MATE_EVICT guard=$MATE_GUARD"
echo "  upstream_random_cg=$RANDOM_CG upstream_random_cg_l2_fetch_bytes=$RANDOM_CG_L2_FETCH_BYTES"
echo "  source_before_stage_n=$SRC source_after_all=$NEXT upstream_bin=$UP_BIN"
echo "  stage_n_scope=pair_block_count_reads_only self_load_unchanged=1 mate_load_unchanged=1 geometry_eviction_guard_preserved=1"
echo "  ptxas_log=$BUILD_ERR upstream_ptxas_log=$UP_ERR"
