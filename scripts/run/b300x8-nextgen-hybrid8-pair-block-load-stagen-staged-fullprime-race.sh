#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-N promotion targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEL_WINNER_ENV="${STAGEL_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g8_winner.env}"
STAGEL_PREPARE_ENV="${STAGEL_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_fullprime_n27_prepared.env}"
STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_g8_winner.env}"
STAGEM_PREPARE_ENV="${STAGEM_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; RUN_STAGED="${RUN_STAGED:-1}"; PREPARE_ONLY="${PREPARE_ONLY:-0}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}}"; WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"; RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27}"; PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"
for x in RUN_STAGED PREPARE_ONLY SELECT_ONLY REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$UPSTREAM_KIND" in auto|stagel|stagem) ;; *) exit 2;; esac
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for x in MOD TARGET_MIB MAX_WINDOW; do [[ "${!x}" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for f in "$PROFILE_FILE" "$STAGE_F_ENV" "$STAGEL_WINNER_ENV" "$STAGEL_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-N promotion input=$f" >&2; exit 2; }; done
command -v sha256sum >/dev/null || exit 2

# Validate Stage L before any Stage-N GPU work. Stage M, when selected, is a
# strictly stronger upstream and has its own manifest chained to Stage L.
# shellcheck disable=SC1090
source "$STAGEL_PREPARE_ENV"
[[ "${B300_STAGEL_PREPARED:-0}" == 1 && "${B300_STAGEL_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEL_PREPARED_NGPU:-0}" == "$NGPU" ]] || { echo 'Stage-N/Stage-L prepare mismatch' >&2; exit 3; }
[[ -s "${B300_STAGEL_PREPARED_MANIFEST:-}" ]] || { echo 'Stage-L manifest missing for Stage N' >&2; exit 3; }
sha256sum -c "$B300_STAGEL_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-L manifest mismatch before Stage N' >&2; exit 3; }
RESOLVED_UPSTREAM=stagel; UP_PREPARE_ENV="$STAGEL_PREPARE_ENV"; UP_WINNER_ENV="$STAGEL_WINNER_ENV"; UP_MANIFEST="$B300_STAGEL_PREPARED_MANIFEST"; UP_BIN="$B300_STAGEL_PREPARED_BIN"; UP_LABEL="$B300_STAGEL_PREPARED_LABEL"; UP_THREADS="$B300_STAGEL_PREPARED_THREADS"; UP_MATE_POLICY=default
use_m=0
if [[ "$UPSTREAM_KIND" == stagem ]]; then use_m=1
elif [[ "$UPSTREAM_KIND" == auto && -s "$STAGEM_PREPARE_ENV" && -s "$STAGEM_WINNER_ENV" ]]; then
  grep -Fq 'B300_STAGEM_PREPARED=1' "$STAGEM_PREPARE_ENV" && use_m=1 || true
fi
if ((use_m)); then
  [[ -s "$STAGEM_PREPARE_ENV" && -s "$STAGEM_WINNER_ENV" ]] || { echo 'Stage-N requested Stage M but artifacts missing' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$STAGEM_PREPARE_ENV"
  [[ "${B300_STAGEM_PREPARED:-0}" == 1 && "${B300_STAGEM_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEM_PREPARED_NGPU:-0}" == "$NGPU" ]] || { echo 'Stage-N/Stage-M prepare mismatch' >&2; exit 3; }
  [[ -s "${B300_STAGEM_PREPARED_MANIFEST:-}" ]] || exit 3; sha256sum -c "$B300_STAGEM_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-M manifest mismatch before Stage N' >&2; exit 3; }
  [[ "${B300_STAGEM_PREPARED_CONTROL_BIN:-}" == "$B300_STAGEL_PREPARED_BIN" ]] || { echo 'Stage-M no longer controls against exact Stage L' >&2; exit 3; }
  case "$B300_STAGEM_PREPARED_POLICY" in cg|cs) ;; *) exit 3;; esac
  RESOLVED_UPSTREAM=stagem; UP_PREPARE_ENV="$STAGEM_PREPARE_ENV"; UP_WINNER_ENV="$STAGEM_WINNER_ENV"; UP_MANIFEST="$B300_STAGEM_PREPARED_MANIFEST"; UP_BIN="$B300_STAGEM_PREPARED_BIN"; UP_LABEL="$B300_STAGEM_PREPARED_LABEL"; UP_THREADS="$B300_STAGEM_PREPARED_THREADS"; UP_MATE_POLICY="$B300_STAGEM_PREPARED_POLICY"
elif [[ "$UPSTREAM_KIND" == stagem ]]; then exit 3
fi
[[ -x "$UP_BIN" ]] || exit 3

if [[ "$RUN_STAGED" == 1 ]]; then
  STAGE_F_ENV="$STAGE_F_ENV" STAGEL_GUARD_ENV="$STAGEL_WINNER_ENV" STAGEM_WINNER_ENV="$STAGEM_WINNER_ENV" UPSTREAM_KIND="$RESOLVED_UPSTREAM" ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" MIN_SPEEDUP="$MIN_SPEEDUP" PAIR_POLICY_LIST="$PAIR_POLICY_LIST" BLOCK_POLICY_LIST="$BLOCK_POLICY_LIST" PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-pair-block-load-policy-staged-calibrate.sh"
fi
[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-N winner=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_NGPU B300_STAGEN_UPSTREAM_KIND B300_STAGEN_MATE_LOAD_POLICY B300_STAGEN_BASE_COUNT_POLICY B300_STAGEN_PAIR_POLICY B300_STAGEN_BLOCK_POLICY B300_STAGEN_FINAL_BIN B300_STAGEN_FINAL_THREADS B300_STAGEN_FINAL_SPEEDUP B300_STAGEN_FINAL_SPILL_FREE B300_STAGEN_CONTROL_BIN B300_STAGEN_CONTROL_THREADS B300_STAGEN_FINAL_STAGE_ROWS B300_STAGEN_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-N winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_FINAL_SPILL_FREE" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage N did not survive staged validation' >&2; exit 4; }
[[ "$B300_STAGEN_UPSTREAM_KIND" == "$RESOLVED_UPSTREAM" && "$B300_STAGEN_MATE_LOAD_POLICY" == "$UP_MATE_POLICY" ]] || { echo 'Stage-N upstream drift' >&2; exit 3; }
[[ "$B300_STAGEN_CONTROL_BIN" == "$UP_BIN" ]] || { echo 'Stage-N control is not exact prepared upstream' >&2; exit 3; }
for p in "$B300_STAGEN_PAIR_POLICY" "$B300_STAGEN_BLOCK_POLICY" "$B300_STAGEN_BASE_COUNT_POLICY"; do case "$p" in default|cg|cs) ;; *) exit 3;; esac; done
[[ "$B300_STAGEN_PAIR_POLICY" != "$B300_STAGEN_BASE_COUNT_POLICY" || "$B300_STAGEN_BLOCK_POLICY" != "$B300_STAGEN_BASE_COUNT_POLICY" ]] || { echo 'Stage-N promoted symmetric inherited baseline' >&2; exit 4; }
python3 - "$B300_STAGEN_FINAL_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('Stage-N speedup below threshold')
PY
[[ -x "$B300_STAGEN_FINAL_BIN" ]] || exit 3

if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"
  inputs=("$WINNER_ENV" "$STAGE_F_ENV" "$STAGEL_WINNER_ENV" "$STAGEL_PREPARE_ENV" "$B300_STAGEL_PREPARED_MANIFEST")
  if [[ "$RESOLVED_UPSTREAM" == stagem ]]; then inputs+=("$STAGEM_WINNER_ENV" "$STAGEM_PREPARE_ENV" "$B300_STAGEM_PREPARED_MANIFEST"); fi
  inputs+=("$B300_STAGEN_FINAL_BIN" "$B300_STAGEN_CONTROL_BIN")
  sha256sum "${inputs[@]}" >"$tmp"; mv "$tmp" "$MANIFEST"
else
  [[ -s "$MANIFEST" ]] || { echo 'missing Stage-N promotion manifest' >&2; exit 3; }
fi
sha256sum -c "$MANIFEST" >/dev/null || { echo 'Stage-N promotion fingerprint mismatch' >&2; exit 3; }
FINAL_SHA="$(sha256sum "$B300_STAGEN_FINAL_BIN" | awk '{print $1}')"; CONTROL_SHA="$(sha256sum "$B300_STAGEN_CONTROL_BIN" | awk '{print $1}')"; MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"; UP_MANIFEST_SHA="$(sha256sum "$UP_MANIFEST" | awk '{print $1}')"
label="stagen_pair_${B300_STAGEN_PAIR_POLICY}_block_${B300_STAGEN_BLOCK_POLICY}_mp${UP_MATE_POLICY}"; control_label="stagen_${RESOLVED_UPSTREAM}_control"
mkdir -p "$(dirname "$PREPARE_ENV")"
{
  printf 'B300_STAGEN_PREPARED=1\n'; printf 'B300_STAGEN_PREPARED_MOD=%q\n' "$MOD"; printf 'B300_STAGEN_PREPARED_NGPU=%q\n' "$NGPU"; printf 'B300_STAGEN_PREPARED_UPSTREAM_KIND=%q\n' "$RESOLVED_UPSTREAM"; printf 'B300_STAGEN_PREPARED_UPSTREAM_PREPARE_ENV=%q\n' "$UP_PREPARE_ENV"; printf 'B300_STAGEN_PREPARED_UPSTREAM_WINNER_ENV=%q\n' "$UP_WINNER_ENV"; printf 'B300_STAGEN_PREPARED_UPSTREAM_MANIFEST=%q\n' "$UP_MANIFEST"; printf 'B300_STAGEN_PREPARED_UPSTREAM_MANIFEST_SHA256=%q\n' "$UP_MANIFEST_SHA";
  printf 'B300_STAGEN_PREPARED_MATE_LOAD_POLICY=%q\n' "$UP_MATE_POLICY"; printf 'B300_STAGEN_PREPARED_BASE_COUNT_POLICY=%q\n' "$B300_STAGEN_BASE_COUNT_POLICY"; printf 'B300_STAGEN_PREPARED_PAIR_POLICY=%q\n' "$B300_STAGEN_PAIR_POLICY"; printf 'B300_STAGEN_PREPARED_BLOCK_POLICY=%q\n' "$B300_STAGEN_BLOCK_POLICY"; printf 'B300_STAGEN_PREPARED_BIN=%q\n' "$B300_STAGEN_FINAL_BIN"; printf 'B300_STAGEN_PREPARED_BIN_SHA256=%q\n' "$FINAL_SHA"; printf 'B300_STAGEN_PREPARED_LABEL=%q\n' "$label"; printf 'B300_STAGEN_PREPARED_THREADS=%q\n' "$B300_STAGEN_FINAL_THREADS";
  printf 'B300_STAGEN_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGEN_CONTROL_BIN"; printf 'B300_STAGEN_PREPARED_CONTROL_BIN_SHA256=%q\n' "$CONTROL_SHA"; printf 'B300_STAGEN_PREPARED_CONTROL_LABEL=%q\n' "$control_label"; printf 'B300_STAGEN_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGEN_CONTROL_THREADS"; printf 'B300_STAGEN_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGEN_FINAL_SPEEDUP"; printf 'B300_STAGEN_PREPARED_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEN_FINAL_STAGE_ROWS"; printf 'B300_STAGEN_PREPARED_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEN_FINAL_STAGE_RESIDUE"; printf 'B300_STAGEN_PREPARED_MANIFEST=%q\n' "$MANIFEST"; printf 'B300_STAGEN_PREPARED_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA";
} >"$PREPARE_ENV"
if [[ "$PREPARE_ONLY" == 1 ]]; then cat "$PREPARE_ENV"; echo "STAGE N PREPARED upstream=$RESOLVED_UPSTREAM pair=$B300_STAGEN_PAIR_POLICY block=$B300_STAGEN_BLOCK_POLICY speedup=${B300_STAGEN_FINAL_SPEEDUP}x env=$PREPARE_ENV" >&2; exit 0; fi
[[ "$NGPU" == 8 ]] || { echo 'Stage-N complete-prime promotion requires NGPU=8; use PREPARE_ONLY=1 for screening' >&2; exit 2; }
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$MOD" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" FORCED_OVERRIDE_BIN="$B300_STAGEN_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGEN_FINAL_THREADS" FORCED_BASE_BIN="$B300_STAGEN_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGEN_CONTROL_THREADS" REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
