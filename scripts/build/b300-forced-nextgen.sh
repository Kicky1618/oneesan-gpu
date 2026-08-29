#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'nextgen forced builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; RECURRENCE_ILP="${RECURRENCE_ILP:-2}"; RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0/1' >&2; exit 2; }
case "$RECURRENCE_ILP" in 2|4|8);;*)echo 'RECURRENCE_ILP must be 2,4,8' >&2;exit 2;;esac
for x in RANDOM_CG PREFETCH_L2 DUALMASK; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256);;*)echo 'RANDOM_CG_L2_FETCH_BYTES must be 0,64,128,256' >&2;exit 2;;esac
if (( RANDOM_CG_L2_FETCH_BYTES > 0 )) && [[ "$RANDOM_CG" != 1 ]]; then echo 'RANDOM_CG_L2_FETCH_BYTES>0 requires RANDOM_CG=1' >&2; exit 2; fi
case "$CLOSURE_BATCH" in 0|2|4);;*)echo 'CLOSURE_BATCH must be 0,2,4' >&2;exit 2;;esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || { echo 'MAXRREGCOUNT must be 0 or 32..255' >&2; exit 2; }
[[ "$PTXAS_VERBOSE" == 0 || "$PTXAS_VERBOSE" == 1 ]] || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="hd${HIGH_DROP_CHUNK}_ilp${RECURRENCE_ILP}_cg${RANDOM_CG}_cgl2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_dual${DUALMASK}_cb${CLOSURE_BATCH}_r${MAXRREGCOUNT}_$$"
ISO="${NEXTGEN_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
BASE_BIN="$ISO/base.bin"; BASE_OUT="$ISO/base.build.out"; BASE_ERR="$ISO/base.build.err"; FINAL_ERR="${BUILD_ERR:-${OUT}.build.err}"

# Canonical production recurrence source plus all established proofs/gates.
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK=0 BUILD_ERR="$BASE_ERR" CALIBRATED_BUILD_DIR="$ISO/basechain" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$ISO/base.driver.err"
[[ -x "$BASE_BIN" ]] || { echo 'canonical recurrence base binary missing' >&2; exit 3; }
grep -Fq 'calibrated_forced=1 main_recurrence=1' "$BASE_OUT"; grep -Fq "high_drop_chunk=$HIGH_DROP_CHUNK dualmask=0" "$BASE_OUT"
SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"; [[ -n "$SRC" && -f "$SRC" ]] || { echo 'canonical recurrence source missing' >&2; exit 3; }
cp "$BASE_ERR" "$FINAL_ERR"; CURRENT="$SRC"

python3 "$ONEESAN_ROOT/scripts/bench/b300-main-recurrence-ilp-partition-proof.py" >"$ISO/mainrec-ilp-proof.out"
if [[ "$RECURRENCE_ILP" != 2 ]]; then NEXT="$ISO/mainrec_ilp${RECURRENCE_ILP}.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-ilp.py" "$CURRENT" "$NEXT" "$RECURRENCE_ILP" >"$ISO/mainrec-ilp.transform.out"; grep -Fq "unified_main_recurrence_ilp=$RECURRENCE_ILP" "$ISO/mainrec-ilp.transform.out"; CURRENT="$NEXT"; fi
if [[ "$RANDOM_CG" == 1 ]]; then NEXT="$ISO/mainrec_random_cg.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py" "$CURRENT" "$NEXT" "$RANDOM_CG_L2_FETCH_BYTES" >"$ISO/mainrec-cg.transform.out"; grep -Fq 'mainrec_random_cg=1' "$ISO/mainrec-cg.transform.out"; grep -Fq "l2_prefetch_bytes=$RANDOM_CG_L2_FETCH_BYTES" "$ISO/mainrec-cg.transform.out"; CURRENT="$NEXT"; fi
if [[ "$PREFETCH_L2" == 1 ]]; then NEXT="$ISO/mainrec_prefetch_l2.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-prefetch-l2.py" "$CURRENT" "$NEXT" >"$ISO/mainrec-prefetch.transform.out"; grep -Fq 'mainrec_prefetch_l2=1' "$ISO/mainrec-prefetch.transform.out"; CURRENT="$NEXT"; fi
if [[ "$DUALMASK" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh" >"$ISO/dualmask-proof.out" 2>"$ISO/dualmask-proof.err"; NEXT="$ISO/block_dualmask.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$CURRENT" "$NEXT" >"$ISO/dualmask.transform.out"; grep -Fq 'block_pull_dualmask=1' "$ISO/dualmask.transform.out"; CURRENT="$NEXT"; fi
if [[ "$CLOSURE_BATCH" != 0 ]]; then NEXT="$ISO/block_closure_batch.cu"; python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$CURRENT" "$NEXT" >"$ISO/closure-batch.transform.out"; grep -Fq 'batch_macro=B300_BLOCK_CLOSURE_BATCH' "$ISO/closure-batch.transform.out"; CURRENT="$NEXT"; fi

grep -Fq 'high_rec_groups=' "$CURRENT"; grep -Fq 'b300_main_trit_get(x,p-14)' "$CURRENT"
if grep -Fq 'b300_main_trit_get(x,p-13)' "$CURRENT"; then echo 'stale p13 recurrence artifact in final source' >&2; exit 4; fi
if [[ "$RECURRENCE_ILP" != 2 ]]; then grep -Fq "base+=Code(${RECURRENCE_ILP})*grid" "$CURRENT"; fi
if [[ "$RANDOM_CG" == 1 ]]; then
  grep -Fq 'b300_mainrec_random_load_cg(in+pj0)' "$CURRENT"
  if (( RANDOM_CG_L2_FETCH_BYTES == 0 )); then grep -Fq 'ld.global.cg.u32' "$CURRENT"; else grep -Fq "ld.global.cg.L2::${RANDOM_CG_L2_FETCH_BYTES}B.u32" "$CURRENT"; fi
fi
if [[ "$PREFETCH_L2" == 1 ]]; then grep -Fq 'prefetch.global.L2' "$CURRENT"; fi
if [[ "$DUALMASK" == 1 ]]; then grep -Fq 'b300_block_endpoint_masks(d)' "$CURRENT"; fi
if [[ "$CLOSURE_BATCH" != 0 ]]; then grep -Fq 'B300_BLOCK_CLOSURE_BATCH' "$CURRENT"; fi

if [[ "$RECURRENCE_ILP" == 2 && "$RANDOM_CG" == 0 && "$RANDOM_CG_L2_FETCH_BYTES" == 0 && "$PREFETCH_L2" == 0 && "$DUALMASK" == 0 && "$CLOSURE_BATCH" == 0 && "$MAXRREGCOUNT" == 0 ]]; then
  cp "$BASE_BIN" "$OUT"; chmod +x "$OUT"
  echo "built $OUT"
  echo "  nextgen_forced=1 high_drop_chunk=$HIGH_DROP_CHUNK recurrence_ilp=2 random_cg=0 random_cg_l2_fetch_bytes=0 prefetch_l2=0 dualmask=0 closure_batch=0 maxrregcount=0"
  echo "  source_after_all=$CURRENT"
  echo "  production_chain_proof_gates=1 experimental_post_transforms=0"
  echo "  ptxas_log=$FINAL_ERR"
  exit 0
fi

PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$CURRENT" -o "$OUT" >"$ISO/final.compile.out" 2>>"$FINAL_ERR"
[[ -x "$OUT" ]] || { echo 'nextgen forced binary missing' >&2; exit 5; }
echo "built $OUT"
echo "  nextgen_forced=1 high_drop_chunk=$HIGH_DROP_CHUNK recurrence_ilp=$RECURRENCE_ILP random_cg=$RANDOM_CG random_cg_l2_fetch_bytes=$RANDOM_CG_L2_FETCH_BYTES prefetch_l2=$PREFETCH_L2 dualmask=$DUALMASK closure_batch=$CLOSURE_BATCH maxrregcount=$MAXRREGCOUNT"
echo "  source_after_all=$CURRENT"
echo "  production_chain_proof_gates=1 experimental_post_transforms=1 extra_state_bytes=0"
echo "  transform_order=production_recurrence,recurrence_ilp,random_cg_l2_fetch,prefetch_l2,dualmask,closure_batch,register_cap"
echo "  ptxas_log=$FINAL_ERR"
