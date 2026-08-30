#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-Q promotion targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g8_winner.env}"; STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_WINNER_ENV="${STAGEO_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_staged_g8_winner.env}"; STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"
STAGEP_WINNER_ENV="${STAGEP_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_staged_g8_winner.env}"; STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
RUN_STAGED="${RUN_STAGED:-1}"; PREPARE_ONLY="${PREPARE_ONLY:-0}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"; BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_staged_g${NGPU}}"; WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"; MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27}"; PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"
for x in RUN_STAGED PREPARE_ONLY SELECT_ONLY REBUILD_BUCKETS; do [[ "${!x}" == 0 || "${!x}" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagen|stageo|stagep) ;; *) echo 'bad Stage-Q UPSTREAM_KIND' >&2; exit 2;; esac
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for f in "$PROFILE_FILE" "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-Q promotion input=$f" >&2; exit 2; }; done
command -v sha256sum >/dev/null || exit 2
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

if [[ "$RUN_STAGED" == 1 ]]; then
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" \
    STAGEO_WINNER_ENV="$STAGEO_WINNER_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_WINNER_ENV="$STAGEP_WINNER_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" \
    UPSTREAM_KIND="$UPSTREAM_KIND" ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" MIN_SPEEDUP="$MIN_SPEEDUP" \
    PAIR_L2_LIST="$PAIR_L2_LIST" BLOCK_L2_LIST="$BLOCK_L2_LIST" PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-calibrate.sh"
fi
[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-Q winner env=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
req=(B300_STAGEQ_STAGED_VALIDATED B300_STAGEQ_FINAL_ENABLED B300_STAGEQ_NGPU B300_STAGEQ_UPSTREAM_KIND B300_STAGEQ_STAGEP_COUNT_UPSTREAM B300_STAGEQ_PAIR_POLICY B300_STAGEQ_BLOCK_POLICY B300_STAGEQ_MATE_LOAD_POLICY B300_STAGEQ_BASE_CG_L2_BYTES B300_STAGEQ_BUILDER_PAIR_CG_L2_BYTES B300_STAGEQ_BUILDER_BLOCK_CG_L2_BYTES B300_STAGEQ_MATE_CG_L2_BYTES B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES B300_STAGEQ_PAIR_L2_BYTES B300_STAGEQ_BLOCK_L2_BYTES B300_STAGEQ_FINAL_BIN B300_STAGEQ_FINAL_THREADS B300_STAGEQ_FINAL_SPEEDUP B300_STAGEQ_FINAL_SPILL_FREE B300_STAGEQ_CONTROL_BIN B300_STAGEQ_CONTROL_THREADS B300_STAGEQ_FINAL_STAGE_ROWS B300_STAGEQ_FINAL_STAGE_RESIDUE B300_STAGEQ_UPSTREAM_ROWS B300_STAGEQ_UPSTREAM_RESIDUE B300_STAGEQ_UPSTREAM_MANIFEST)
for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-Q winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEQ_STAGED_VALIDATED" == 1 && "$B300_STAGEQ_FINAL_ENABLED" == 1 && "$B300_STAGEQ_FINAL_SPILL_FREE" == 1 && "$B300_STAGEQ_NGPU" == "$NGPU" ]] || { echo 'Stage Q did not survive staged validation' >&2; exit 4; }
[[ "$B300_STAGEQ_PAIR_L2_BYTES" != "$B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES" || "$B300_STAGEQ_BLOCK_L2_BYTES" != "$B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES" ]] || { echo 'Stage-Q winner equals exact upstream L2 tuple' >&2; exit 4; }
python3 - "$B300_STAGEQ_FINAL_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('Stage-Q speedup below threshold')
PY
[[ -x "$B300_STAGEQ_FINAL_BIN" && -x "$B300_STAGEQ_CONTROL_BIN" ]] || { echo 'Stage-Q final/control binary missing' >&2; exit 3; }
[[ -s "$B300_STAGEQ_UPSTREAM_MANIFEST" ]] || { echo 'Stage-Q upstream manifest missing' >&2; exit 3; }
sha256sum -c "$B300_STAGEQ_UPSTREAM_MANIFEST" >/dev/null || { echo 'Stage-Q upstream manifest mismatch' >&2; exit 3; }

RESOLVED="$B300_STAGEQ_UPSTREAM_KIND"; UP_PREPARE_ENV=''; UP_WINNER_ENV=''; UP_MANIFEST=''; UP_BIN=''; UP_THREADS=''
case "$RESOLVED" in
  stagen)
    # shellcheck disable=SC1090
    source "$STAGEN_PREPARE_ENV"
    [[ "${B300_STAGEN_PREPARED:-0}" == 1 && "$B300_STAGEN_PREPARED_MOD" == "$MOD" && "$B300_STAGEN_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-Q/Stage-N prepare drift' >&2; exit 3; }
    [[ "$B300_STAGEQ_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" && "$B300_STAGEQ_UPSTREAM_MANIFEST" == "$B300_STAGEN_PREPARED_MANIFEST" ]] || { echo 'Stage-Q control is not exact Stage N' >&2; exit 3; }
    [[ "$B300_STAGEQ_PAIR_POLICY" == "$B300_STAGEN_PREPARED_PAIR_POLICY" && "$B300_STAGEQ_BLOCK_POLICY" == "$B300_STAGEN_PREPARED_BLOCK_POLICY" && "$B300_STAGEQ_MATE_LOAD_POLICY" == "$B300_STAGEN_PREPARED_MATE_LOAD_POLICY" ]] || { echo 'Stage-Q/Stage-N policy drift' >&2; exit 3; }
    UP_PREPARE_ENV="$STAGEN_PREPARE_ENV"; UP_WINNER_ENV="$STAGEN_WINNER_ENV"; UP_MANIFEST="$B300_STAGEN_PREPARED_MANIFEST"; UP_BIN="$B300_STAGEN_PREPARED_BIN"; UP_THREADS="$B300_STAGEN_PREPARED_THREADS"
    ;;
  stageo)
    [[ -s "$STAGEO_PREPARE_ENV" && -s "$STAGEO_WINNER_ENV" ]] || { echo 'Stage-Q Stage-O provenance missing' >&2; exit 3; }
    # shellcheck disable=SC1090
    source "$STAGEO_PREPARE_ENV"
    [[ "${B300_STAGEO_PREPARED:-0}" == 1 && "$B300_STAGEO_PREPARED_MOD" == "$MOD" && "$B300_STAGEO_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-Q/Stage-O prepare drift' >&2; exit 3; }
    [[ "$B300_STAGEQ_CONTROL_BIN" == "$B300_STAGEO_PREPARED_BIN" && "$B300_STAGEQ_UPSTREAM_MANIFEST" == "$B300_STAGEO_PREPARED_MANIFEST" ]] || { echo 'Stage-Q control is not exact Stage O' >&2; exit 3; }
    [[ "$B300_STAGEQ_PAIR_POLICY" == "$B300_STAGEO_PREPARED_PAIR_POLICY" && "$B300_STAGEQ_BLOCK_POLICY" == "$B300_STAGEO_PREPARED_BLOCK_POLICY" && "$B300_STAGEQ_MATE_LOAD_POLICY" == "$B300_STAGEO_PREPARED_MATE_LOAD_POLICY" ]] || { echo 'Stage-Q/Stage-O policy drift' >&2; exit 3; }
    [[ "$B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES" == "$B300_STAGEO_PREPARED_PAIR_L2_BYTES" && "$B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES" == "$B300_STAGEO_PREPARED_BLOCK_L2_BYTES" ]] || { echo 'Stage-Q/Stage-O Count L2 drift' >&2; exit 3; }
    UP_PREPARE_ENV="$STAGEO_PREPARE_ENV"; UP_WINNER_ENV="$STAGEO_WINNER_ENV"; UP_MANIFEST="$B300_STAGEO_PREPARED_MANIFEST"; UP_BIN="$B300_STAGEO_PREPARED_BIN"; UP_THREADS="$B300_STAGEO_PREPARED_THREADS"
    ;;
  stagep)
    [[ -s "$STAGEP_PREPARE_ENV" && -s "$STAGEP_WINNER_ENV" ]] || { echo 'Stage-Q Stage-P provenance missing' >&2; exit 3; }
    # shellcheck disable=SC1090
    source "$STAGEP_PREPARE_ENV"
    [[ "${B300_STAGEP_PREPARED:-0}" == 1 && "$B300_STAGEP_PREPARED_MOD" == "$MOD" && "$B300_STAGEP_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-Q/Stage-P prepare drift' >&2; exit 3; }
    [[ "$B300_STAGEQ_CONTROL_BIN" == "$B300_STAGEP_PREPARED_BIN" && "$B300_STAGEQ_UPSTREAM_MANIFEST" == "$B300_STAGEP_PREPARED_MANIFEST" ]] || { echo 'Stage-Q control is not exact Stage P' >&2; exit 3; }
    [[ "$B300_STAGEQ_STAGEP_COUNT_UPSTREAM" == "$B300_STAGEP_PREPARED_COUNT_UPSTREAM" && "$B300_STAGEQ_PAIR_POLICY" == "$B300_STAGEP_PREPARED_PAIR_POLICY" && "$B300_STAGEQ_BLOCK_POLICY" == "$B300_STAGEP_PREPARED_BLOCK_POLICY" && "$B300_STAGEQ_MATE_LOAD_POLICY" == "$B300_STAGEP_PREPARED_MATE_LOAD_POLICY" && "$B300_STAGEQ_MATE_CG_L2_BYTES" == "$B300_STAGEP_PREPARED_MATE_L2_BYTES" ]] || { echo 'Stage-Q/Stage-P policy drift' >&2; exit 3; }
    UP_PREPARE_ENV="$STAGEP_PREPARE_ENV"; UP_WINNER_ENV="$STAGEP_WINNER_ENV"; UP_MANIFEST="$B300_STAGEP_PREPARED_MANIFEST"; UP_BIN="$B300_STAGEP_PREPARED_BIN"; UP_THREADS="$B300_STAGEP_PREPARED_THREADS"
    ;;
  *) echo "bad resolved Stage-Q upstream=$RESOLVED" >&2; exit 3;;
esac
[[ "$B300_STAGEQ_CONTROL_BIN" == "$UP_BIN" ]] || exit 3
sha256sum -c "$UP_MANIFEST" >/dev/null || { echo 'Stage-Q exact upstream fingerprint mismatch' >&2; exit 3; }

if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"
  inputs=("$WINNER_ENV" "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$B300_STAGEN_PREPARED_MANIFEST")
  if [[ "$RESOLVED" == stageo || ( "$RESOLVED" == stagep && "$B300_STAGEQ_STAGEP_COUNT_UPSTREAM" == stageo ) ]]; then
    inputs+=("$STAGEO_WINNER_ENV" "$STAGEO_PREPARE_ENV")
  fi
  [[ "$RESOLVED" == stagep ]] && inputs+=("$STAGEP_WINNER_ENV" "$STAGEP_PREPARE_ENV")
  inputs+=("$UP_MANIFEST" "$B300_STAGEQ_FINAL_BIN" "$B300_STAGEQ_CONTROL_BIN")
  uniq=(); for f in "${inputs[@]}"; do seen=0; for old in "${uniq[@]}"; do [[ "$old" == "$f" ]] && seen=1; done; ((seen)) || uniq+=("$f"); done
  sha256sum "${uniq[@]}" >"$tmp"; mv "$tmp" "$MANIFEST"
else
  [[ -s "$MANIFEST" ]] || { echo 'Stage-Q manifest required when RUN_STAGED=0' >&2; exit 3; }
fi
sha256sum -c "$MANIFEST" >/dev/null || { echo 'Stage-Q promotion fingerprint mismatch' >&2; exit 3; }
FINAL_SHA="$(sha256sum "$B300_STAGEQ_FINAL_BIN"|awk '{print $1}')"; CONTROL_SHA="$(sha256sum "$B300_STAGEQ_CONTROL_BIN"|awk '{print $1}')"; MANIFEST_SHA="$(sha256sum "$MANIFEST"|awk '{print $1}')"; UP_MANIFEST_SHA="$(sha256sum "$UP_MANIFEST"|awk '{print $1}')"
label="stageq_ilp8_count_l2_p${B300_STAGEQ_PAIR_L2_BYTES}_b${B300_STAGEQ_BLOCK_L2_BYTES}_${RESOLVED}"; control_label="stageq_${RESOLVED}_control"
mkdir -p "$(dirname "$PREPARE_ENV")"
{
  printf 'B300_STAGEQ_PREPARED=1\n'; printf 'B300_STAGEQ_PREPARED_MOD=%q\n' "$MOD"; printf 'B300_STAGEQ_PREPARED_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEQ_PREPARED_UPSTREAM_KIND=%q\n' "$RESOLVED"; printf 'B300_STAGEQ_PREPARED_STAGEP_COUNT_UPSTREAM=%q\n' "$B300_STAGEQ_STAGEP_COUNT_UPSTREAM"
  printf 'B300_STAGEQ_PREPARED_UPSTREAM_PREPARE_ENV=%q\n' "$UP_PREPARE_ENV"; printf 'B300_STAGEQ_PREPARED_UPSTREAM_WINNER_ENV=%q\n' "$UP_WINNER_ENV"; printf 'B300_STAGEQ_PREPARED_UPSTREAM_MANIFEST=%q\n' "$UP_MANIFEST"; printf 'B300_STAGEQ_PREPARED_UPSTREAM_MANIFEST_SHA256=%q\n' "$UP_MANIFEST_SHA"
  printf 'B300_STAGEQ_PREPARED_PAIR_POLICY=%q\n' "$B300_STAGEQ_PAIR_POLICY"; printf 'B300_STAGEQ_PREPARED_BLOCK_POLICY=%q\n' "$B300_STAGEQ_BLOCK_POLICY"; printf 'B300_STAGEQ_PREPARED_MATE_LOAD_POLICY=%q\n' "$B300_STAGEQ_MATE_LOAD_POLICY"
  printf 'B300_STAGEQ_PREPARED_BASE_CG_L2_BYTES=%q\n' "$B300_STAGEQ_BASE_CG_L2_BYTES"; printf 'B300_STAGEQ_PREPARED_BUILDER_PAIR_CG_L2_BYTES=%q\n' "$B300_STAGEQ_BUILDER_PAIR_CG_L2_BYTES"; printf 'B300_STAGEQ_PREPARED_BUILDER_BLOCK_CG_L2_BYTES=%q\n' "$B300_STAGEQ_BUILDER_BLOCK_CG_L2_BYTES"; printf 'B300_STAGEQ_PREPARED_MATE_CG_L2_BYTES=%q\n' "$B300_STAGEQ_MATE_CG_L2_BYTES"
  printf 'B300_STAGEQ_PREPARED_UPSTREAM_PAIR_L2_BYTES=%q\n' "$B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES"; printf 'B300_STAGEQ_PREPARED_UPSTREAM_BLOCK_L2_BYTES=%q\n' "$B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES"; printf 'B300_STAGEQ_PREPARED_PAIR_L2_BYTES=%q\n' "$B300_STAGEQ_PAIR_L2_BYTES"; printf 'B300_STAGEQ_PREPARED_BLOCK_L2_BYTES=%q\n' "$B300_STAGEQ_BLOCK_L2_BYTES"
  printf 'B300_STAGEQ_PREPARED_BIN=%q\n' "$B300_STAGEQ_FINAL_BIN"; printf 'B300_STAGEQ_PREPARED_BIN_SHA256=%q\n' "$FINAL_SHA"; printf 'B300_STAGEQ_PREPARED_LABEL=%q\n' "$label"; printf 'B300_STAGEQ_PREPARED_THREADS=%q\n' "$B300_STAGEQ_FINAL_THREADS"
  printf 'B300_STAGEQ_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGEQ_CONTROL_BIN"; printf 'B300_STAGEQ_PREPARED_CONTROL_BIN_SHA256=%q\n' "$CONTROL_SHA"; printf 'B300_STAGEQ_PREPARED_CONTROL_LABEL=%q\n' "$control_label"; printf 'B300_STAGEQ_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGEQ_CONTROL_THREADS"
  printf 'B300_STAGEQ_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGEQ_FINAL_SPEEDUP"; printf 'B300_STAGEQ_PREPARED_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEQ_FINAL_STAGE_ROWS"; printf 'B300_STAGEQ_PREPARED_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEQ_FINAL_STAGE_RESIDUE"
  printf 'B300_STAGEQ_PREPARED_MANIFEST=%q\n' "$MANIFEST"; printf 'B300_STAGEQ_PREPARED_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA"
} >"$PREPARE_ENV"
if [[ "$PREPARE_ONLY" == 1 ]]; then
  cat "$PREPARE_ENV"
  echo "STAGE Q PREPARED upstream=$RESOLVED pair_l2=$B300_STAGEQ_PAIR_L2_BYTES block_l2=$B300_STAGEQ_BLOCK_L2_BYTES speedup=${B300_STAGEQ_FINAL_SPEEDUP}x env=$PREPARE_ENV" >&2
  exit 0
fi
[[ "$NGPU" == 8 ]] || { echo 'Stage-Q complete-prime promotion requires NGPU=8; use PREPARE_ONLY=1 for screening' >&2; exit 2; }
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$MOD" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_STAGEQ_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGEQ_FINAL_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEQ_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGEQ_CONTROL_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
