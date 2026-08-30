#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'Stage-O builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_stageo_n27}")"
MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-default}"; PAIR_LOAD_POLICY="${PAIR_LOAD_POLICY:-default}"; BLOCK_LOAD_POLICY="${BLOCK_LOAD_POLICY:-default}"
BASE_CG_L2_BYTES="${BASE_CG_L2_BYTES:-0}"; PAIR_CG_L2_BYTES="${PAIR_CG_L2_BYTES:-0}"; BLOCK_CG_L2_BYTES="${BLOCK_CG_L2_BYTES:-0}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
SELF_WIDTH="${SELF_WIDTH:-8}"; SELF_DISTANCE="${SELF_DISTANCE:-1}"; MATE_WIDTH="${MATE_WIDTH:-$SELF_WIDTH}"; MATE_DISTANCE="${MATE_DISTANCE:-$SELF_DISTANCE}"
SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"; SELF_GUARD="${SELF_GUARD:-branch}"; MATE_GUARD="${MATE_GUARD:-branch}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-$BASE_CG_L2_BYTES}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"; BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
for p in "$MATE_LOAD_POLICY" "$PAIR_LOAD_POLICY" "$BLOCK_LOAD_POLICY"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-O policy=$p" >&2; exit 2;; esac; done
for name in BASE_CG_L2_BYTES PAIR_CG_L2_BYTES BLOCK_CG_L2_BYTES RANDOM_CG_L2_FETCH_BYTES; do case "${!name}" in 0|64|128|256) ;; *) echo "$name must be 0,64,128,256" >&2; exit 2;; esac; done
[[ "$PAIR_LOAD_POLICY" == cg || "$PAIR_CG_L2_BYTES" == 0 ]] || { echo 'PAIR_CG_L2_BYTES requires pair policy cg' >&2; exit 2; }
[[ "$BLOCK_LOAD_POLICY" == cg || "$BLOCK_CG_L2_BYTES" == 0 ]] || { echo 'BLOCK_CG_L2_BYTES requires block policy cg' >&2; exit 2; }
[[ "$PAIR_LOAD_POLICY" == cg || "$BLOCK_LOAD_POLICY" == cg ]] || { echo 'Stage O requires at least one CG Count axis' >&2; exit 2; }
for w in "$SELF_WIDTH" "$MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 2;; esac; done
for d in "$SELF_DISTANCE" "$MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 2;; esac; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) exit 2;; esac; done
for g in "$SELF_GUARD" "$MATE_GUARD"; do case "$g" in branch|predicated) ;; *) exit 2;; esac; done
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || exit 2
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="pp${PAIR_LOAD_POLICY}${PAIR_CG_L2_BYTES}_bp${BLOCK_LOAD_POLICY}${BLOCK_CG_L2_BYTES}_base${BASE_CG_L2_BYTES}_mp${MATE_LOAD_POLICY}_$$"
ISO="${STAGEO_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-stage-o/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
N_BIN="$ISO/stagen.bin"; N_OUT="$ISO/stagen.build.out"; N_ERR="$ISO/stagen.build.err"
N=27 ARCH="$ARCH" OUT="$N_BIN" MATE_LOAD_POLICY="$MATE_LOAD_POLICY" PAIR_LOAD_POLICY="$PAIR_LOAD_POLICY" BLOCK_LOAD_POLICY="$BLOCK_LOAD_POLICY" STAGEN_CG_L2_FETCH_BYTES="$BASE_CG_L2_BYTES" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" \
  SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" SELF_GUARD="$SELF_GUARD" MATE_GUARD="$MATE_GUARD" RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$N_ERR" STAGEN_BUILD_DIR="$ISO/stagen-build" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh" >"$N_OUT" 2>"$ISO/stagen.driver.err"
[[ -x "$N_BIN" ]] || { echo 'Stage-O Stage-N base binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  source_before_stage_n=.* source_after_all=([^ ]+) upstream_bin=.*/\1/p' "$N_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'Stage-O Stage-N source_after_all missing' >&2; exit 3; }
grep -Fq "pair_load_policy=$PAIR_LOAD_POLICY block_load_policy=$BLOCK_LOAD_POLICY stagen_cg_l2_fetch_bytes=$BASE_CG_L2_BYTES" "$N_OUT" || { echo 'Stage-O Stage-N base policy drift' >&2; exit 3; }

NEXT="$ISO/stageo.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-cg-l2-policy.py" "$SRC" "$NEXT" "$PAIR_CG_L2_BYTES" "$BLOCK_CG_L2_BYTES" >"$ISO/stageo.transform.out"
grep -Fq "b300_mainrec_stageo_pair_block_cg_l2=1 pair_policy=$PAIR_LOAD_POLICY block_policy=$BLOCK_LOAD_POLICY pair_l2_bytes=$PAIR_CG_L2_BYTES block_l2_bytes=$BLOCK_CG_L2_BYTES base_l2_bytes=$BASE_CG_L2_BYTES" "$ISO/stageo.transform.out" || exit 3
: >"$BUILD_ERR"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'Stage-O binary missing' >&2; exit 5; }
echo "built $OUT"
echo "  stage_o_cg_l2=1 mate_load_policy=$MATE_LOAD_POLICY pair_load_policy=$PAIR_LOAD_POLICY block_load_policy=$BLOCK_LOAD_POLICY"
echo "  base_cg_l2_bytes=$BASE_CG_L2_BYTES pair_cg_l2_bytes=$PAIR_CG_L2_BYTES block_cg_l2_bytes=$BLOCK_CG_L2_BYTES"
echo "  self_geometry width=$SELF_WIDTH distance=$SELF_DISTANCE evict=$SELF_EVICT guard=$SELF_GUARD"
echo "  mate_geometry width=$MATE_WIDTH distance=$MATE_DISTANCE evict=$MATE_EVICT guard=$MATE_GUARD"
echo "  source_before_stage_o=$SRC source_after_all=$NEXT stage_n_base_bin=$N_BIN"
echo "  stage_o_scope=cg_l2_fetch_size_only pair_block_policy_preserved=1 self_load_unchanged=1 mate_load_unchanged=1"
echo "  ptxas_log=$BUILD_ERR stage_n_ptxas_log=$N_ERR"
