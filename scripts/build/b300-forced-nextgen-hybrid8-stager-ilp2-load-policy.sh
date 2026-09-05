#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'Stage-R builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_stager_n27}")"
UPSTREAM_KIND="${UPSTREAM_KIND:-stageq}"
STAGEQ_UPSTREAM_KIND="${STAGEQ_UPSTREAM_KIND:-stagep}"
STAGEP_COUNT_UPSTREAM="${STAGEP_COUNT_UPSTREAM:-stagen}"
MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-default}"; PAIR_LOAD_POLICY="${PAIR_LOAD_POLICY:-default}"; BLOCK_LOAD_POLICY="${BLOCK_LOAD_POLICY:-default}"
ILP2_PAIR_LOAD_POLICY="${ILP2_PAIR_LOAD_POLICY:-$PAIR_LOAD_POLICY}"; ILP2_BLOCK_LOAD_POLICY="${ILP2_BLOCK_LOAD_POLICY:-$BLOCK_LOAD_POLICY}"
BASE_CG_L2_BYTES="${BASE_CG_L2_BYTES:-0}"; PAIR_CG_L2_BYTES="${PAIR_CG_L2_BYTES:-0}"; BLOCK_CG_L2_BYTES="${BLOCK_CG_L2_BYTES:-0}"; MATE_CG_L2_BYTES="${MATE_CG_L2_BYTES:-0}"
ILP8_PAIR_CG_L2_BYTES="${ILP8_PAIR_CG_L2_BYTES:-0}"; ILP8_BLOCK_CG_L2_BYTES="${ILP8_BLOCK_CG_L2_BYTES:-0}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
SELF_WIDTH="${SELF_WIDTH:-8}"; SELF_DISTANCE="${SELF_DISTANCE:-1}"; MATE_WIDTH="${MATE_WIDTH:-$SELF_WIDTH}"; MATE_DISTANCE="${MATE_DISTANCE:-$SELF_DISTANCE}"
SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"; SELF_GUARD="${SELF_GUARD:-branch}"; MATE_GUARD="${MATE_GUARD:-branch}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-$BASE_CG_L2_BYTES}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"; BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$UPSTREAM_KIND" in stagen|stageo|stagep|stageq) ;; *) echo 'UPSTREAM_KIND must be stagen, stageo, stagep, or stageq' >&2; exit 2;; esac
case "$STAGEQ_UPSTREAM_KIND" in stagen|stageo|stagep) ;; *) echo 'STAGEQ_UPSTREAM_KIND must be stagen, stageo, or stagep' >&2; exit 2;; esac
case "$STAGEP_COUNT_UPSTREAM" in stagen|stageo) ;; *) echo 'STAGEP_COUNT_UPSTREAM must be stagen or stageo' >&2; exit 2;; esac
for p in "$MATE_LOAD_POLICY" "$PAIR_LOAD_POLICY" "$BLOCK_LOAD_POLICY" "$ILP2_PAIR_LOAD_POLICY" "$ILP2_BLOCK_LOAD_POLICY"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-R load policy=$p" >&2; exit 2;; esac; done
for name in BASE_CG_L2_BYTES PAIR_CG_L2_BYTES BLOCK_CG_L2_BYTES MATE_CG_L2_BYTES ILP8_PAIR_CG_L2_BYTES ILP8_BLOCK_CG_L2_BYTES RANDOM_CG_L2_FETCH_BYTES; do case "${!name}" in 0|64|128|256) ;; *) echo "$name must be 0,64,128,256" >&2; exit 2;; esac; done
for w in "$SELF_WIDTH" "$MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 2;; esac; done
for d in "$SELF_DISTANCE" "$MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 2;; esac; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) exit 2;; esac; done
for g in "$SELF_GUARD" "$MATE_GUARD"; do case "$g" in branch|predicated) ;; *) exit 2;; esac; done
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do [[ "${!x}" == 0 || "${!x}" == 1 ]] || exit 2; done
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || exit 2
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

TAG="up${UPSTREAM_KIND}_qup${STAGEQ_UPSTREAM_KIND}_lp${ILP2_PAIR_LOAD_POLICY}_lb${ILP2_BLOCK_LOAD_POLICY}_$$"
ISO="${STAGER_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-stage-r/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
UP_BIN="$ISO/upstream.bin"; UP_OUT="$ISO/upstream.build.out"; UP_ERR="$ISO/upstream.build.err"
COMMON=(N=27 ARCH="$ARCH" OUT="$UP_BIN" MATE_LOAD_POLICY="$MATE_LOAD_POLICY" PAIR_LOAD_POLICY="$PAIR_LOAD_POLICY" BLOCK_LOAD_POLICY="$BLOCK_LOAD_POLICY" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" SELF_GUARD="$SELF_GUARD" MATE_GUARD="$MATE_GUARD" RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$UP_ERR")
case "$UPSTREAM_KIND" in
  stagen)
    env "${COMMON[@]}" STAGEN_CG_L2_FETCH_BYTES="$BASE_CG_L2_BYTES" STAGEN_BUILD_DIR="$ISO/stagen-build" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh" >"$UP_OUT" 2>"$ISO/upstream.driver.err"
    ;;
  stageo)
    env "${COMMON[@]}" BASE_CG_L2_BYTES="$BASE_CG_L2_BYTES" PAIR_CG_L2_BYTES="$PAIR_CG_L2_BYTES" BLOCK_CG_L2_BYTES="$BLOCK_CG_L2_BYTES" STAGEO_BUILD_DIR="$ISO/stageo-build" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageo-cg-l2-policy.sh" >"$UP_OUT" 2>"$ISO/upstream.driver.err"
    ;;
  stagep)
    env "${COMMON[@]}" COUNT_UPSTREAM="$STAGEP_COUNT_UPSTREAM" BASE_CG_L2_BYTES="$BASE_CG_L2_BYTES" PAIR_CG_L2_BYTES="$PAIR_CG_L2_BYTES" BLOCK_CG_L2_BYTES="$BLOCK_CG_L2_BYTES" MATE_CG_L2_BYTES="$MATE_CG_L2_BYTES" STAGEP_BUILD_DIR="$ISO/stagep-build" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stagep-mate-cg-l2-policy.sh" >"$UP_OUT" 2>"$ISO/upstream.driver.err"
    ;;
  stageq)
    env "${COMMON[@]}" UPSTREAM_KIND="$STAGEQ_UPSTREAM_KIND" STAGEP_COUNT_UPSTREAM="$STAGEP_COUNT_UPSTREAM" BASE_CG_L2_BYTES="$BASE_CG_L2_BYTES" PAIR_CG_L2_BYTES="$PAIR_CG_L2_BYTES" BLOCK_CG_L2_BYTES="$BLOCK_CG_L2_BYTES" MATE_CG_L2_BYTES="$MATE_CG_L2_BYTES" ILP8_PAIR_CG_L2_BYTES="$ILP8_PAIR_CG_L2_BYTES" ILP8_BLOCK_CG_L2_BYTES="$ILP8_BLOCK_CG_L2_BYTES" STAGEQ_BUILD_DIR="$ISO/stageq-build" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh" >"$UP_OUT" 2>"$ISO/upstream.driver.err"
    ;;
esac
[[ -x "$UP_BIN" ]] || { echo 'Stage-R upstream binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  .*source_after_all=([^ ]+).*/\1/p' "$UP_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'Stage-R upstream source_after_all missing' >&2; tail -n 40 "$UP_OUT" >&2 || true; exit 3; }
grep -Fq "pair_load_policy=$PAIR_LOAD_POLICY block_load_policy=$BLOCK_LOAD_POLICY" "$UP_OUT" || { echo 'Stage-R upstream Count policy drift' >&2; exit 3; }
if [[ "$UPSTREAM_KIND" == stageq ]]; then
  grep -Fq "ilp8_pair_cg_l2_bytes=$ILP8_PAIR_CG_L2_BYTES ilp8_block_cg_l2_bytes=$ILP8_BLOCK_CG_L2_BYTES" "$UP_OUT" || { echo 'Stage-R lost Stage-Q ILP8 L2 tuple' >&2; exit 3; }
fi

NEXT="$ISO/stager.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-load-policy.py" "$SRC" "$NEXT" "$ILP2_PAIR_LOAD_POLICY" "$ILP2_BLOCK_LOAD_POLICY" >"$ISO/stager.transform.out"
grep -Fq "b300_mainrec_stager_ilp2_pair_block_policy=1 pair_policy=$ILP2_PAIR_LOAD_POLICY block_policy=$ILP2_BLOCK_LOAD_POLICY" "$ISO/stager.transform.out" || exit 3
if [[ "$UPSTREAM_KIND" == stageq ]]; then grep -Fq 'stageq_preserved=1' "$ISO/stager.transform.out" || exit 3; fi
: >"$BUILD_ERR"; PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v"); REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'Stage-R binary missing' >&2; exit 5; }
echo "built $OUT"
echo "  stage_r_ilp2_load_policy=1 upstream_kind=$UPSTREAM_KIND stageq_upstream_kind=$STAGEQ_UPSTREAM_KIND"
echo "  high_pair_load_policy=$PAIR_LOAD_POLICY high_block_load_policy=$BLOCK_LOAD_POLICY ilp2_pair_load_policy=$ILP2_PAIR_LOAD_POLICY ilp2_block_load_policy=$ILP2_BLOCK_LOAD_POLICY"
echo "  ilp8_pair_cg_l2_bytes=$ILP8_PAIR_CG_L2_BYTES ilp8_block_cg_l2_bytes=$ILP8_BLOCK_CG_L2_BYTES mate_load_policy=$MATE_LOAD_POLICY mate_cg_l2_bytes=$MATE_CG_L2_BYTES"
echo "  source_before_stage_r=$SRC source_after_all=$NEXT upstream_bin=$UP_BIN"
echo "  stage_r_scope=ilp2_pair_block_load_policy_only ilp8_exact_upstream=1 self_policy_preserved=1 mate_policy_preserved=1 mate_writes_unchanged=1"
echo "  ptxas_log=$BUILD_ERR upstream_ptxas_log=$UP_ERR"
