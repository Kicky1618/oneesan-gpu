#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'Stage-U builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_stageu_n27}")"
UPSTREAM_KIND="${UPSTREAM_KIND:-stages}"; ILP2_MATE_CG_L2_BYTES="${ILP2_MATE_CG_L2_BYTES:-128}"
CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"; BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$UPSTREAM_KIND" in stager|stages) ;; *) echo 'UPSTREAM_KIND must be stager or stages' >&2; exit 2;; esac
case "$ILP2_MATE_CG_L2_BYTES" in 0|64|128|256) ;; *) echo 'ILP2_MATE_CG_L2_BYTES must be 0,64,128,256' >&2; exit 2;; esac
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
[[ "$PTXAS_VERBOSE" == 0 || "$PTXAS_VERBOSE" == 1 ]] || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
TBUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh"
UGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-mate-cg-l2-policy.py"
[[ -s "$TBUILDER" && -s "$UGEN" ]] || exit 2
TAG="up${UPSTREAM_KIND}_l2${ILP2_MATE_CG_L2_BYTES}_$$"; ISO="${STAGEU_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-stage-u/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
T_BIN="$ISO/staget-cg.bin"; T_OUT="$ISO/staget-cg.build.out"; T_ERR="$ISO/staget-cg.build.err"
# Crucial Stage-U semantics: reconstruct a fresh Stage-T cg candidate directly from
# exact R/S lineage. Do not require or consume an accepted Stage-T winner.
env N=27 ARCH="$ARCH" OUT="$T_BIN" UPSTREAM_KIND="$UPSTREAM_KIND" ILP2_MATE_LOAD_POLICY=cg PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$T_ERR" \
  bash "$TBUILDER" >"$T_OUT" 2>"$ISO/staget-cg.driver.err"
[[ -x "$T_BIN" ]] || { echo 'Stage-U synthetic Stage-T cg binary missing' >&2; exit 3; }
grep -Fq "stage_t_ilp2_mate_policy=1 upstream_kind=$UPSTREAM_KIND" "$T_OUT" || { echo 'Stage-U failed to reconstruct Stage-T cg from immediate upstream' >&2; exit 3; }
grep -Fq 'ilp2_mate_load_policy=cg' "$T_OUT" || { echo 'Stage-U synthetic low mate policy is not cg' >&2; exit 3; }
SRC="$(sed -nE 's/^  .*source_after_all=([^ ]+).*/\1/p' "$T_OUT" | tail -n1)"; [[ -n "$SRC" && -f "$SRC" ]] || { echo 'Stage-U Stage-T source_after_all missing' >&2; exit 3; }
NEXT="$ISO/stageu.cu"; python3 "$UGEN" "$SRC" "$NEXT" "$ILP2_MATE_CG_L2_BYTES" >"$ISO/stageu.transform.out"
grep -Fq "b300_mainrec_stageu_ilp2_mate_cg_l2=1 l2_bytes=$ILP2_MATE_CG_L2_BYTES" "$ISO/stageu.transform.out" || exit 3
grep -Fq 'ilp8_byte_identical=1 count_loads_unchanged=1 semantics_unchanged=1' "$ISO/stageu.transform.out" || exit 3
: >"$BUILD_ERR"; PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v"); REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT"); DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'Stage-U binary missing' >&2; exit 5; }
echo "built $OUT"
echo "  stage_u_ilp2_mate_cg_l2=1 upstream_kind=$UPSTREAM_KIND ilp2_mate_policy=cg ilp2_mate_cg_l2_bytes=$ILP2_MATE_CG_L2_BYTES"
echo "  synthetic_staget_cg=1 accepted_staget_winner_required=0 immediate_control_lineage=R_or_S"
echo "  source_before_stage_u=$SRC source_after_all=$NEXT synthetic_staget_bin=$T_BIN"
echo "  stage_u_scope=ilp2_mate_reads_only ilp8_exact_upstream=1 count_policies_preserved=1 count_l2_preserved=1 mate_writes_unchanged=1"
echo "  ptxas_log=$BUILD_ERR synthetic_staget_ptxas_log=$T_ERR"
