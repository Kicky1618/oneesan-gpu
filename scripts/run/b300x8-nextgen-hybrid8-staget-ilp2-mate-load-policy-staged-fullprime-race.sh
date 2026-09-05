#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-T promotion targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g8_winner.env}"; STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"; STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"; STAGEQ_PREPARE_ENV="${STAGEQ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27_prepared.env}"
STAGER_WINNER_ENV="${STAGER_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_staged_g8_winner.env}"; STAGER_PREPARE_ENV="${STAGER_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_fullprime_n27_prepared.env}"; STAGES_WINNER_ENV="${STAGES_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_staged_g8_winner.env}"; STAGES_PREPARE_ENV="${STAGES_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"; UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"
RUN_STAGED="${RUN_STAGED:-1}"; PREPARE_ONLY="${PREPARE_ONLY:-0}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; POLICY_LIST="${POLICY_LIST:-default cg cs}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_staget_ilp2_mate_staged_g${NGPU}}"; WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"; MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_staget_ilp2_mate_fullprime_n27}"; PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"; PROMOTION_ENV="${PROMOTION_ENV:-${RACE_PREFIX}_promotion.env}"
for x in RUN_STAGED PREPARE_ONLY SELECT_ONLY REBUILD_BUCKETS; do [[ "${!x}" == 0 || "${!x}" == 1 ]] || exit 2; done
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2; case "$UPSTREAM_KIND" in auto|stager|stages) ;; *) exit 2;; esac
for x in MOD TARGET_MIB MAX_WINDOW; do [[ "${!x}" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for f in "$PROFILE_FILE" "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$STAGER_WINNER_ENV" "$STAGER_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-T input=$f" >&2; exit 2; }; done
command -v sha256sum >/dev/null || exit 2
if [[ "$RUN_STAGED" == 1 ]]; then
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" STAGEQ_PREPARE_ENV="$STAGEQ_PREPARE_ENV" STAGER_WINNER_ENV="$STAGER_WINNER_ENV" STAGER_PREPARE_ENV="$STAGER_PREPARE_ENV" STAGES_WINNER_ENV="$STAGES_WINNER_ENV" STAGES_PREPARE_ENV="$STAGES_PREPARE_ENV" \
    UPSTREAM_KIND="$UPSTREAM_KIND" ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" MIN_SPEEDUP="$MIN_SPEEDUP" POLICY_LIST="$POLICY_LIST" PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-calibrate.sh"
fi
[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-T winner=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for k in B300_STAGET_STAGED_VALIDATED B300_STAGET_FINAL_ENABLED B300_STAGET_NGPU B300_STAGET_UPSTREAM_KIND B300_STAGET_STAGER_UPSTREAM_KIND B300_STAGET_LOW_PAIR_POLICY B300_STAGET_LOW_BLOCK_POLICY B300_STAGET_LOW_PAIR_L2_BYTES B300_STAGET_LOW_BLOCK_L2_BYTES B300_STAGET_HIGH_PAIR_POLICY B300_STAGET_HIGH_BLOCK_POLICY B300_STAGET_HIGH_PAIR_L2_BYTES B300_STAGET_HIGH_BLOCK_L2_BYTES B300_STAGET_HIGH_MATE_POLICY B300_STAGET_HIGH_MATE_L2_BYTES B300_STAGET_POLICY B300_STAGET_FINAL_BIN B300_STAGET_FINAL_THREADS B300_STAGET_FINAL_SPEEDUP B300_STAGET_FINAL_SPILL_FREE B300_STAGET_CONTROL_BIN B300_STAGET_CONTROL_THREADS B300_STAGET_FINAL_STAGE_ROWS B300_STAGET_FINAL_STAGE_RESIDUE B300_STAGET_UPSTREAM_MANIFEST; do [[ -n "${!k+x}" ]] || { echo "Stage-T winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGET_STAGED_VALIDATED" == 1 && "$B300_STAGET_FINAL_ENABLED" == 1 && "$B300_STAGET_FINAL_SPILL_FREE" == 1 && "$B300_STAGET_NGPU" == "$NGPU" ]] || { echo 'Stage T did not survive staged validation' >&2; exit 4; }
case "$B300_STAGET_POLICY" in cg|cs) ;; *) echo 'Stage T retained exact default mate policy' >&2; exit 4;; esac
python3 - "$B300_STAGET_FINAL_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('Stage-T speedup below threshold')
PY

# Bind and verify Stage R first. Stage S is an optional low-state Count/L2 refinement
# whose exact control must be this prepared Stage-R winner.
# shellcheck disable=SC1090
source "$STAGER_PREPARE_ENV"
[[ "${B300_STAGER_PREPARED:-0}" == 1 && "$B300_STAGER_PREPARED_MOD" == "$MOD" && "$B300_STAGER_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-T/Stage-R prepare mismatch' >&2; exit 3; }
[[ -x "$B300_STAGER_PREPARED_BIN" && -s "$B300_STAGER_PREPARED_MANIFEST" ]] || { echo 'Stage-T Stage-R prepared binary/manifest missing' >&2; exit 3; }
sha256sum -c "$B300_STAGER_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-R manifest mismatch before Stage T promotion' >&2; exit 3; }
[[ "$B300_STAGET_STAGER_UPSTREAM_KIND" == "$B300_STAGER_PREPARED_UPSTREAM_KIND" ]] || { echo 'Stage-T Stage-R upstream provenance drift' >&2; exit 3; }

# Resolve high-state mate provenance from the exact Stage-R upstream.
EXPECTED_HIGH_MATE_POLICY=''
EXPECTED_HIGH_MATE_L2=0
case "$B300_STAGER_PREPARED_UPSTREAM_KIND" in
  stageq)
    [[ -s "$STAGEQ_PREPARE_ENV" ]] || { echo 'Stage-T Stage-Q prepare missing for high mate provenance' >&2; exit 3; }
    # shellcheck disable=SC1090
    source "$STAGEQ_PREPARE_ENV"
    [[ "${B300_STAGEQ_PREPARED:-0}" == 1 && "$B300_STAGEQ_PREPARED_MOD" == "$MOD" && "$B300_STAGEQ_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-T/Stage-Q prepare mismatch' >&2; exit 3; }
    [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEQ_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-Q winner before Stage T' >&2; exit 3; }
    EXPECTED_HIGH_MATE_POLICY="$B300_STAGEQ_PREPARED_MATE_LOAD_POLICY"
    EXPECTED_HIGH_MATE_L2="$B300_STAGEQ_PREPARED_MATE_CG_L2_BYTES"
    ;;
  stagep)
    [[ -s "$STAGEP_PREPARE_ENV" ]] || { echo 'Stage-T Stage-P prepare missing for high mate provenance' >&2; exit 3; }
    # shellcheck disable=SC1090
    source "$STAGEP_PREPARE_ENV"
    [[ "${B300_STAGEP_PREPARED:-0}" == 1 && "$B300_STAGEP_PREPARED_MOD" == "$MOD" && "$B300_STAGEP_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-T/Stage-P prepare mismatch' >&2; exit 3; }
    [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEP_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-P winner before Stage T' >&2; exit 3; }
    EXPECTED_HIGH_MATE_POLICY="$B300_STAGEP_PREPARED_MATE_LOAD_POLICY"
    EXPECTED_HIGH_MATE_L2="$B300_STAGEP_PREPARED_MATE_L2_BYTES"
    ;;
  stageo)
    [[ -s "$STAGEO_PREPARE_ENV" ]] || { echo 'Stage-T Stage-O prepare missing for high mate provenance' >&2; exit 3; }
    # shellcheck disable=SC1090
    source "$STAGEO_PREPARE_ENV"
    [[ "${B300_STAGEO_PREPARED:-0}" == 1 && "$B300_STAGEO_PREPARED_MOD" == "$MOD" && "$B300_STAGEO_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-T/Stage-O prepare mismatch' >&2; exit 3; }
    [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEO_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-O winner before Stage T' >&2; exit 3; }
    EXPECTED_HIGH_MATE_POLICY="$B300_STAGEO_PREPARED_MATE_LOAD_POLICY"
    EXPECTED_HIGH_MATE_L2=0
    ;;
  stagen)
    [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-N winner before Stage T' >&2; exit 3; }
    EXPECTED_HIGH_MATE_POLICY="$B300_STAGEN_PREPARED_MATE_LOAD_POLICY"
    EXPECTED_HIGH_MATE_L2=0
    ;;
  *) echo 'bad Stage-R upstream before Stage T' >&2; exit 3;;
esac
case "$EXPECTED_HIGH_MATE_POLICY" in default|cg|cs) ;; *) echo 'bad expected Stage-T high mate policy' >&2; exit 3;; esac
case "$EXPECTED_HIGH_MATE_L2" in 0|64|128|256) ;; *) echo 'bad expected Stage-T high mate L2' >&2; exit 3;; esac
[[ "$EXPECTED_HIGH_MATE_POLICY" == cg || "$EXPECTED_HIGH_MATE_L2" == 0 ]] || { echo 'non-cg high mate policy cannot carry L2 hint' >&2; exit 3; }
[[ "$B300_STAGET_HIGH_MATE_POLICY" == "$EXPECTED_HIGH_MATE_POLICY" && "$B300_STAGET_HIGH_MATE_L2_BYTES" == "$EXPECTED_HIGH_MATE_L2" ]] || { echo 'Stage-T changed high mate provenance' >&2; exit 3; }

# Bind the exact prepared immediate upstream selected by T: S when accepted, otherwise R.
case "$B300_STAGET_UPSTREAM_KIND" in
  stages)
    [[ -s "$STAGES_PREPARE_ENV" ]] || { echo 'Stage-T selected Stage S but prepare env is missing' >&2; exit 3; }
    # shellcheck disable=SC1090
    source "$STAGES_PREPARE_ENV"
    [[ "${B300_STAGES_PREPARED:-0}" == 1 && "$B300_STAGES_PREPARED_MOD" == "$MOD" && "$B300_STAGES_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-T/Stage-S prepare mismatch' >&2; exit 3; }
    UP_BIN="$B300_STAGES_PREPARED_BIN"; UP_THREADS="$B300_STAGES_PREPARED_THREADS"; UP_MAN="$B300_STAGES_PREPARED_MANIFEST"; UP_LABEL="$B300_STAGES_PREPARED_LABEL"
    [[ "$B300_STAGES_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN" ]] || { echo 'Stage-S control is not exact Stage-R winner before Stage T' >&2; exit 3; }
    [[ "$B300_STAGET_STAGER_UPSTREAM_KIND" == "$B300_STAGES_PREPARED_STAGER_UPSTREAM_KIND" && "$B300_STAGET_LOW_PAIR_POLICY" == "$B300_STAGES_PREPARED_LOW_PAIR_POLICY" && "$B300_STAGET_LOW_BLOCK_POLICY" == "$B300_STAGES_PREPARED_LOW_BLOCK_POLICY" && "$B300_STAGET_LOW_PAIR_L2_BYTES" == "$B300_STAGES_PREPARED_PAIR_L2_BYTES" && "$B300_STAGET_LOW_BLOCK_L2_BYTES" == "$B300_STAGES_PREPARED_BLOCK_L2_BYTES" ]] || { echo 'Stage-T lost Stage-S/R low Count provenance' >&2; exit 3; }
    ;;
  stager)
    UP_BIN="$B300_STAGER_PREPARED_BIN"; UP_THREADS="$B300_STAGER_PREPARED_THREADS"; UP_MAN="$B300_STAGER_PREPARED_MANIFEST"; UP_LABEL="$B300_STAGER_PREPARED_LABEL"
    [[ "$B300_STAGET_LOW_PAIR_POLICY" == "$B300_STAGER_PREPARED_PAIR_POLICY" && "$B300_STAGET_LOW_BLOCK_POLICY" == "$B300_STAGER_PREPARED_BLOCK_POLICY" && "$B300_STAGET_LOW_PAIR_L2_BYTES" == 0 && "$B300_STAGET_LOW_BLOCK_L2_BYTES" == 0 ]] || { echo 'Stage-T lost Stage-R low Count provenance' >&2; exit 3; }
    ;;
  *) echo 'bad Stage-T immediate upstream' >&2; exit 3;;
esac
[[ -x "$UP_BIN" && -s "$UP_MAN" ]] || exit 3; sha256sum -c "$UP_MAN" >/dev/null || { echo 'Stage-T upstream manifest failed' >&2; exit 3; }
[[ "$B300_STAGET_CONTROL_BIN" == "$UP_BIN" ]] || { echo 'Stage-T control is not exact prepared immediate upstream binary' >&2; exit 3; }
# High Count provenance is always inherited from R.
[[ "$B300_STAGET_HIGH_PAIR_POLICY" == "$B300_STAGER_PREPARED_HIGH_PAIR_POLICY" && "$B300_STAGET_HIGH_BLOCK_POLICY" == "$B300_STAGER_PREPARED_HIGH_BLOCK_POLICY" && "$B300_STAGET_HIGH_PAIR_L2_BYTES" == "$B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES" && "$B300_STAGET_HIGH_BLOCK_L2_BYTES" == "$B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES" ]] || { echo 'Stage-T changed high Count provenance' >&2; exit 3; }
[[ -x "$B300_STAGET_FINAL_BIN" ]] || exit 3
if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"; files=("$WINNER_ENV" "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$STAGER_WINNER_ENV" "$STAGER_PREPARE_ENV" "$UP_MAN" "$B300_STAGET_FINAL_BIN" "$B300_STAGET_CONTROL_BIN"); [[ "$B300_STAGET_UPSTREAM_KIND" == stages ]] && files+=("$STAGES_WINNER_ENV" "$STAGES_PREPARE_ENV"); sha256sum "${files[@]}" >"$tmp"; mv "$tmp" "$MANIFEST"
else [[ -s "$MANIFEST" ]] || { echo 'missing Stage-T promotion manifest' >&2; exit 3; }; fi
sha256sum -c "$MANIFEST" >/dev/null || { echo 'Stage-T promotion fingerprint mismatch' >&2; exit 3; }
FINAL_SHA="$(sha256sum "$B300_STAGET_FINAL_BIN"|awk '{print $1}')"; CONTROL_SHA="$(sha256sum "$B300_STAGET_CONTROL_BIN"|awk '{print $1}')"; MANIFEST_SHA="$(sha256sum "$MANIFEST"|awk '{print $1}')"
label="staget_ilp2_mate_${B300_STAGET_POLICY}_up${B300_STAGET_UPSTREAM_KIND}"; control_label="staget_exact_${B300_STAGET_UPSTREAM_KIND}_control"
cat >"$PROMOTION_ENV" <<EOF
B300_STAGET_PROMOTION_VALIDATED=1
B300_STAGET_PROMOTION_MOD=$MOD
B300_STAGET_PROMOTION_NGPU=$NGPU
B300_STAGET_PROMOTION_UPSTREAM_KIND=$B300_STAGET_UPSTREAM_KIND
B300_STAGET_PROMOTION_POLICY=$B300_STAGET_POLICY
B300_STAGET_PROMOTION_BIN=$(printf '%q' "$B300_STAGET_FINAL_BIN")
B300_STAGET_PROMOTION_BIN_SHA256=$FINAL_SHA
B300_STAGET_PROMOTION_CONTROL_BIN=$(printf '%q' "$B300_STAGET_CONTROL_BIN")
B300_STAGET_PROMOTION_CONTROL_SHA256=$CONTROL_SHA
B300_STAGET_PROMOTION_SPEEDUP=$B300_STAGET_FINAL_SPEEDUP
B300_STAGET_PROMOTION_MANIFEST=$(printf '%q' "$MANIFEST")
B300_STAGET_PROMOTION_MANIFEST_SHA256=$MANIFEST_SHA
EOF
if [[ "$PREPARE_ONLY" == 1 ]]; then
  mkdir -p "$(dirname "$PREPARE_ENV")"; {
    printf 'B300_STAGET_PREPARED=1\n'; printf 'B300_STAGET_PREPARED_MOD=%q\n' "$MOD"; printf 'B300_STAGET_PREPARED_NGPU=%q\n' "$NGPU"; printf 'B300_STAGET_PREPARED_UPSTREAM_KIND=%q\n' "$B300_STAGET_UPSTREAM_KIND"; printf 'B300_STAGET_PREPARED_STAGER_UPSTREAM_KIND=%q\n' "$B300_STAGET_STAGER_UPSTREAM_KIND"
    printf 'B300_STAGET_PREPARED_LOW_PAIR_POLICY=%q\n' "$B300_STAGET_LOW_PAIR_POLICY"; printf 'B300_STAGET_PREPARED_LOW_BLOCK_POLICY=%q\n' "$B300_STAGET_LOW_BLOCK_POLICY"; printf 'B300_STAGET_PREPARED_LOW_PAIR_L2_BYTES=%q\n' "$B300_STAGET_LOW_PAIR_L2_BYTES"; printf 'B300_STAGET_PREPARED_LOW_BLOCK_L2_BYTES=%q\n' "$B300_STAGET_LOW_BLOCK_L2_BYTES"; printf 'B300_STAGET_PREPARED_HIGH_PAIR_POLICY=%q\n' "$B300_STAGET_HIGH_PAIR_POLICY"; printf 'B300_STAGET_PREPARED_HIGH_BLOCK_POLICY=%q\n' "$B300_STAGET_HIGH_BLOCK_POLICY"; printf 'B300_STAGET_PREPARED_HIGH_PAIR_L2_BYTES=%q\n' "$B300_STAGET_HIGH_PAIR_L2_BYTES"; printf 'B300_STAGET_PREPARED_HIGH_BLOCK_L2_BYTES=%q\n' "$B300_STAGET_HIGH_BLOCK_L2_BYTES"; printf 'B300_STAGET_PREPARED_HIGH_MATE_POLICY=%q\n' "$B300_STAGET_HIGH_MATE_POLICY"; printf 'B300_STAGET_PREPARED_HIGH_MATE_L2_BYTES=%q\n' "$B300_STAGET_HIGH_MATE_L2_BYTES"
    printf 'B300_STAGET_PREPARED_POLICY=%q\n' "$B300_STAGET_POLICY"; printf 'B300_STAGET_PREPARED_BIN=%q\n' "$B300_STAGET_FINAL_BIN"; printf 'B300_STAGET_PREPARED_LABEL=%q\n' "$label"; printf 'B300_STAGET_PREPARED_THREADS=%q\n' "$B300_STAGET_FINAL_THREADS"; printf 'B300_STAGET_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGET_CONTROL_BIN"; printf 'B300_STAGET_PREPARED_CONTROL_LABEL=%q\n' "$control_label"; printf 'B300_STAGET_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGET_CONTROL_THREADS"; printf 'B300_STAGET_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGET_FINAL_SPEEDUP"; printf 'B300_STAGET_PREPARED_FINAL_STAGE_ROWS=%q\n' "$B300_STAGET_FINAL_STAGE_ROWS"; printf 'B300_STAGET_PREPARED_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGET_FINAL_STAGE_RESIDUE"; printf 'B300_STAGET_PREPARED_UPSTREAM_MANIFEST=%q\n' "$UP_MAN"; printf 'B300_STAGET_PREPARED_MANIFEST=%q\n' "$MANIFEST"; printf 'B300_STAGET_PREPARED_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA"; printf 'B300_STAGET_PREPARED_PROMOTION_ENV=%q\n' "$PROMOTION_ENV"
  } >"$PREPARE_ENV"; cat "$PREPARE_ENV"; echo "STAGE T PREPARED upstream=$B300_STAGET_UPSTREAM_KIND policy=$B300_STAGET_POLICY speedup=${B300_STAGET_FINAL_SPEEDUP}x" >&2; exit 0
fi
[[ "$NGPU" == 8 ]] || { echo 'Stage-T complete-prime promotion requires NGPU=8; use PREPARE_ONLY=1 for screening' >&2; exit 2; }
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$MOD" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" FORCED_OVERRIDE_BIN="$B300_STAGET_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGET_FINAL_THREADS" FORCED_BASE_BIN="$B300_STAGET_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGET_CONTROL_THREADS" REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" RACE_PREFIX="$RACE_PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
