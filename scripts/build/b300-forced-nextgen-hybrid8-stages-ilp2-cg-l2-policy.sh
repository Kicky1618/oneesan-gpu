#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'Stage-S builder targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; OUT="$(build_path "${OUT:-b300_forced_nextgen_hybrid8_stages_n27}")"
UPSTREAM_KIND="${UPSTREAM_KIND:-stageq}"; STAGEQ_UPSTREAM_KIND="${STAGEQ_UPSTREAM_KIND:-stagep}"; STAGEP_COUNT_UPSTREAM="${STAGEP_COUNT_UPSTREAM:-stagen}"
MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-default}"; PAIR_LOAD_POLICY="${PAIR_LOAD_POLICY:-default}"; BLOCK_LOAD_POLICY="${BLOCK_LOAD_POLICY:-default}"
ILP2_PAIR_LOAD_POLICY="${ILP2_PAIR_LOAD_POLICY:-$PAIR_LOAD_POLICY}"; ILP2_BLOCK_LOAD_POLICY="${ILP2_BLOCK_LOAD_POLICY:-$BLOCK_LOAD_POLICY}"
ILP2_PAIR_CG_L2_BYTES="${ILP2_PAIR_CG_L2_BYTES:-0}"; ILP2_BLOCK_CG_L2_BYTES="${ILP2_BLOCK_CG_L2_BYTES:-0}"
BASE_CG_L2_BYTES="${BASE_CG_L2_BYTES:-0}"; PAIR_CG_L2_BYTES="${PAIR_CG_L2_BYTES:-0}"; BLOCK_CG_L2_BYTES="${BLOCK_CG_L2_BYTES:-0}"; MATE_CG_L2_BYTES="${MATE_CG_L2_BYTES:-0}"; ILP8_PAIR_CG_L2_BYTES="${ILP8_PAIR_CG_L2_BYTES:-0}"; ILP8_BLOCK_CG_L2_BYTES="${ILP8_BLOCK_CG_L2_BYTES:-0}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"; SELF_WIDTH="${SELF_WIDTH:-8}"; SELF_DISTANCE="${SELF_DISTANCE:-1}"; MATE_WIDTH="${MATE_WIDTH:-$SELF_WIDTH}"; MATE_DISTANCE="${MATE_DISTANCE:-$SELF_DISTANCE}"; SELF_EVICT="${SELF_EVICT:-default}"; MATE_EVICT="${MATE_EVICT:-default}"; SELF_GUARD="${SELF_GUARD:-branch}"; MATE_GUARD="${MATE_GUARD:-branch}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-$BASE_CG_L2_BYTES}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"; BUILD_ERR="${BUILD_ERR:-${OUT}.build.err}"
case "$UPSTREAM_KIND" in stagen|stageo|stagep|stageq) ;; *) echo 'UPSTREAM_KIND must be stagen,stageo,stagep,stageq' >&2; exit 2;; esac
for p in "$MATE_LOAD_POLICY" "$PAIR_LOAD_POLICY" "$BLOCK_LOAD_POLICY" "$ILP2_PAIR_LOAD_POLICY" "$ILP2_BLOCK_LOAD_POLICY"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-S load policy=$p" >&2; exit 2;; esac; done
for name in BASE_CG_L2_BYTES PAIR_CG_L2_BYTES BLOCK_CG_L2_BYTES MATE_CG_L2_BYTES ILP8_PAIR_CG_L2_BYTES ILP8_BLOCK_CG_L2_BYTES ILP2_PAIR_CG_L2_BYTES ILP2_BLOCK_CG_L2_BYTES RANDOM_CG_L2_FETCH_BYTES; do case "${!name}" in 0|64|128|256) ;; *) echo "$name must be 0,64,128,256" >&2; exit 2;; esac; done
[[ "$ILP2_PAIR_LOAD_POLICY" == cg || "$ILP2_PAIR_CG_L2_BYTES" == 0 ]] || { echo 'ILP2_PAIR_CG_L2_BYTES must be 0 when ILP2 pair policy is not cg' >&2; exit 2; }
[[ "$ILP2_BLOCK_LOAD_POLICY" == cg || "$ILP2_BLOCK_CG_L2_BYTES" == 0 ]] || { echo 'ILP2_BLOCK_CG_L2_BYTES must be 0 when ILP2 block policy is not cg' >&2; exit 2; }
[[ "$ILP2_PAIR_LOAD_POLICY" == cg || "$ILP2_BLOCK_LOAD_POLICY" == cg ]] || { echo 'Stage S requires at least one ILP2 cg axis' >&2; exit 4; }
for w in "$SELF_WIDTH" "$MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 2;; esac; done
for d in "$SELF_DISTANCE" "$MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 2;; esac; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) exit 2;; esac; done
for g in "$SELF_GUARD" "$MATE_GUARD"; do case "$g" in branch|predicated) ;; *) exit 2;; esac; done
for x in HIGH_DROP_CHUNK RANDOM_CG PREFETCH_L2 DUALMASK PTXAS_VERBOSE; do [[ "${!x}" == 0 || "${!x}" == 1 ]] || exit 2; done
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || exit 2; [[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

# Stage R's high-state Count policy always comes from Stage N. Its effective L2 hint,
# however, comes from the maximal Count upstream actually reconstructed beneath R:
# Q > O (possibly through P) > N. Keep this provenance separate from the low-state S axis.
EFFECTIVE_HIGH_PAIR_L2=0; EFFECTIVE_HIGH_BLOCK_L2=0
if [[ "$PAIR_LOAD_POLICY" == cg ]]; then
  case "$UPSTREAM_KIND" in
    stageq) EFFECTIVE_HIGH_PAIR_L2="$ILP8_PAIR_CG_L2_BYTES" ;;
    stageo) EFFECTIVE_HIGH_PAIR_L2="$PAIR_CG_L2_BYTES" ;;
    stagep) [[ "$STAGEP_COUNT_UPSTREAM" == stageo ]] && EFFECTIVE_HIGH_PAIR_L2="$PAIR_CG_L2_BYTES" || EFFECTIVE_HIGH_PAIR_L2="$BASE_CG_L2_BYTES" ;;
    stagen) EFFECTIVE_HIGH_PAIR_L2="$BASE_CG_L2_BYTES" ;;
  esac
fi
if [[ "$BLOCK_LOAD_POLICY" == cg ]]; then
  case "$UPSTREAM_KIND" in
    stageq) EFFECTIVE_HIGH_BLOCK_L2="$ILP8_BLOCK_CG_L2_BYTES" ;;
    stageo) EFFECTIVE_HIGH_BLOCK_L2="$BLOCK_CG_L2_BYTES" ;;
    stagep) [[ "$STAGEP_COUNT_UPSTREAM" == stageo ]] && EFFECTIVE_HIGH_BLOCK_L2="$BLOCK_CG_L2_BYTES" || EFFECTIVE_HIGH_BLOCK_L2="$BASE_CG_L2_BYTES" ;;
    stagen) EFFECTIVE_HIGH_BLOCK_L2="$BASE_CG_L2_BYTES" ;;
  esac
fi

TAG="up${UPSTREAM_KIND}_rp${ILP2_PAIR_LOAD_POLICY}_rb${ILP2_BLOCK_LOAD_POLICY}_sl2${ILP2_PAIR_CG_L2_BYTES}-${ILP2_BLOCK_CG_L2_BYTES}_$$"
ISO="${STAGES_BUILD_DIR:-$ONEESAN_BUILD_DIR/nextgen-stage-s/$TAG}"; mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
UP_BIN="$ISO/stager.bin"; UP_OUT="$ISO/stager.build.out"; UP_ERR="$ISO/stager.build.err"
env N=27 ARCH="$ARCH" OUT="$UP_BIN" UPSTREAM_KIND="$UPSTREAM_KIND" STAGEQ_UPSTREAM_KIND="$STAGEQ_UPSTREAM_KIND" STAGEP_COUNT_UPSTREAM="$STAGEP_COUNT_UPSTREAM" MATE_LOAD_POLICY="$MATE_LOAD_POLICY" PAIR_LOAD_POLICY="$PAIR_LOAD_POLICY" BLOCK_LOAD_POLICY="$BLOCK_LOAD_POLICY" ILP2_PAIR_LOAD_POLICY="$ILP2_PAIR_LOAD_POLICY" ILP2_BLOCK_LOAD_POLICY="$ILP2_BLOCK_LOAD_POLICY" BASE_CG_L2_BYTES="$BASE_CG_L2_BYTES" PAIR_CG_L2_BYTES="$PAIR_CG_L2_BYTES" BLOCK_CG_L2_BYTES="$BLOCK_CG_L2_BYTES" MATE_CG_L2_BYTES="$MATE_CG_L2_BYTES" ILP8_PAIR_CG_L2_BYTES="$ILP8_PAIR_CG_L2_BYTES" ILP8_BLOCK_CG_L2_BYTES="$ILP8_BLOCK_CG_L2_BYTES" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" SELF_GUARD="$SELF_GUARD" MATE_GUARD="$MATE_GUARD" RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE="$PTXAS_VERBOSE" BUILD_ERR="$UP_ERR" STAGER_BUILD_DIR="$ISO/stager-build" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh" >"$UP_OUT" 2>"$ISO/stager.driver.err"
[[ -x "$UP_BIN" ]] || { echo 'Stage-S Stage-R upstream binary missing' >&2; exit 3; }
SRC="$(sed -nE 's/^  .*source_after_all=([^ ]+).*/\1/p' "$UP_OUT" | tail -n1)"; [[ -n "$SRC" && -f "$SRC" ]] || { echo 'Stage-S Stage-R source_after_all missing' >&2; exit 3; }
grep -Fq "ilp2_pair_load_policy=$ILP2_PAIR_LOAD_POLICY ilp2_block_load_policy=$ILP2_BLOCK_LOAD_POLICY" "$UP_OUT" || { echo 'Stage-S lost Stage-R ILP2 tuple' >&2; exit 3; }
grep -Fq "ilp8_pair_cg_l2_bytes=$ILP8_PAIR_CG_L2_BYTES ilp8_block_cg_l2_bytes=$ILP8_BLOCK_CG_L2_BYTES" "$UP_OUT" || { echo 'Stage-S lost Stage-Q builder tuple' >&2; exit 3; }
NEXT="$ISO/stages.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py" "$SRC" "$NEXT" "$ILP2_PAIR_CG_L2_BYTES" "$ILP2_BLOCK_CG_L2_BYTES" >"$ISO/stages.transform.out"
grep -Fq "b300_mainrec_stages_ilp2_pair_block_cg_l2=1 pair_policy=$ILP2_PAIR_LOAD_POLICY block_policy=$ILP2_BLOCK_LOAD_POLICY pair_l2_bytes=$ILP2_PAIR_CG_L2_BYTES block_l2_bytes=$ILP2_BLOCK_CG_L2_BYTES" "$ISO/stages.transform.out" || exit 3
grep -Fq "high_pair=$PAIR_LOAD_POLICY high_block=$BLOCK_LOAD_POLICY high_pair_l2=$EFFECTIVE_HIGH_PAIR_L2 high_block_l2=$EFFECTIVE_HIGH_BLOCK_L2" "$ISO/stages.transform.out" || { echo 'Stage-S transformed high-state provenance drift' >&2; exit 3; }
: >"$BUILD_ERR"; PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v"); REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13); [[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$NEXT" -o "$OUT" 2>>"$BUILD_ERR"
[[ -x "$OUT" ]] || { echo 'Stage-S binary missing' >&2; exit 5; }
echo "built $OUT"
echo "  stage_s_ilp2_cg_l2=1 upstream_kind=$UPSTREAM_KIND"
echo "  high_pair_load_policy=$PAIR_LOAD_POLICY high_block_load_policy=$BLOCK_LOAD_POLICY high_pair_cg_l2_bytes=$EFFECTIVE_HIGH_PAIR_L2 high_block_cg_l2_bytes=$EFFECTIVE_HIGH_BLOCK_L2"
echo "  ilp2_pair_load_policy=$ILP2_PAIR_LOAD_POLICY ilp2_block_load_policy=$ILP2_BLOCK_LOAD_POLICY ilp2_pair_cg_l2_bytes=$ILP2_PAIR_CG_L2_BYTES ilp2_block_cg_l2_bytes=$ILP2_BLOCK_CG_L2_BYTES"
echo "  source_before_stage_s=$SRC source_after_all=$NEXT upstream_bin=$UP_BIN"
echo "  stage_s_scope=ilp2_cg_l2_hint_only ilp8_exact_upstream=1 stage_r_policy_preserved=1"
echo "  ptxas_log=$BUILD_ERR upstream_ptxas_log=$UP_ERR"
