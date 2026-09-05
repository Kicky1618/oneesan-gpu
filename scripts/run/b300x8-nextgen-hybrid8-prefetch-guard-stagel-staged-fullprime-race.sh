#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-L guard promotion targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
UPSTREAM_PREPARE_ENV="${UPSTREAM_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_fullprime_n27_prepared.env}"
UPSTREAM_WINNER_ENV="${UPSTREAM_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_staged_winner.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
RUN_STAGED="${RUN_STAGED:-1}"; PREPARE_ONLY="${PREPARE_ONLY:-0}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; GUARD_LIST="${GUARD_LIST:-bb pb bp pp}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g${NGPU}}"
WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_fullprime_n27}"
PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"

for x in RUN_STAGED PREPARE_ONLY SELECT_ONLY REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for x in MOD TARGET_MIB MAX_WINDOW; do v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for f in "$PROFILE_FILE" "$STAGE_F_ENV" "$UPSTREAM_PREPARE_ENV" "$UPSTREAM_WINNER_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-L input=$f" >&2; exit 2; }; done
command -v sha256sum >/dev/null || exit 2

# Bind Stage L to the exact prepared upstream before any GPU work. Both Stage J
# and Stage K expose their modulus and manifest; accepting neither is fail-closed.
# shellcheck disable=SC1090
source "$UPSTREAM_PREPARE_ENV"
UPSTREAM_KIND=""; UP_MANIFEST=""; UP_MOD=""
if [[ "${B300_STAGEK_PREPARED:-0}" == 1 ]]; then
  UPSTREAM_KIND=stagek
  UP_MOD="${B300_STAGEK_PREPARED_MOD:-}"
  UP_MANIFEST="${B300_STAGEK_PREPARED_MANIFEST:-}"
  SW="$B300_STAGEK_PREPARED_SELF_WIDTH"; SD="$B300_STAGEK_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEK_PREPARED_SELF_EVICT"
  MW="$B300_STAGEK_PREPARED_MATE_WIDTH"; MD="$B300_STAGEK_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEK_PREPARED_MATE_EVICT"
elif [[ "${B300_STAGEJ_PREPARED:-0}" == 1 ]]; then
  UPSTREAM_KIND=stagej
  UP_MOD="${B300_STAGEJ_PREPARED_MOD:-}"
  UP_MANIFEST="${B300_STAGEJ_PREPARED_MANIFEST:-}"
  SW="$B300_STAGEJ_PREPARED_SELF_WIDTH"; SD="$B300_STAGEJ_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEJ_PREPARED_SELF_EVICT"
  MW="$B300_STAGEJ_PREPARED_MATE_WIDTH"; MD="$B300_STAGEJ_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEJ_PREPARED_MATE_EVICT"
else
  echo 'Stage-L upstream prepare is neither Stage J nor Stage K' >&2; exit 3
fi
[[ "$UP_MOD" == "$MOD" ]] || { echo "Stage-L/upstream modulus mismatch stagel=$MOD upstream=${UP_MOD:-missing}" >&2; exit 3; }
[[ -s "$UP_MANIFEST" ]] || { echo 'Stage-L upstream manifest missing' >&2; exit 3; }
sha256sum -c "$UP_MANIFEST" >/dev/null || { echo 'Stage-L upstream manifest failed verification' >&2; exit 3; }
UP_MANIFEST_SHA="$(sha256sum "$UP_MANIFEST" | awk '{print $1}')"
for w in "$SW" "$MW"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$SD" "$MD"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for e in "$SE" "$ME"; do case "$e" in default|normal|last) ;; *) exit 3;; esac; done

if [[ "$RUN_STAGED" == 1 ]]; then
  STAGE_F_ENV="$STAGE_F_ENV" UPSTREAM_PREPARE_ENV="$UPSTREAM_PREPARE_ENV" UPSTREAM_WINNER_ENV="$UPSTREAM_WINNER_ENV" \
    ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" \
    MIN_SPEEDUP="$MIN_SPEEDUP" GUARD_LIST="$GUARD_LIST" PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-prefetch-guard-staged-calibrate.sh"
fi
[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-L winner=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for k in B300_STAGEL_STAGED_VALIDATED B300_STAGEL_FINAL_ENABLED B300_STAGEL_NGPU B300_STAGEL_UPSTREAM_KIND \
  B300_STAGEL_FINAL_PROFILE B300_STAGEL_FINAL_SELF_GUARD B300_STAGEL_FINAL_MATE_GUARD B300_STAGEL_FINAL_BIN \
  B300_STAGEL_FINAL_THREADS B300_STAGEL_FINAL_SPEEDUP B300_STAGEL_FINAL_SPILL_FREE B300_STAGEL_CONTROL_BIN \
  B300_STAGEL_CONTROL_THREADS B300_STAGEL_FINAL_STAGE_ROWS B300_STAGEL_FINAL_STAGE_RESIDUE \
  B300_STAGEL_STAGE_F_ENV B300_STAGEL_UPSTREAM_PREPARE_ENV B300_STAGEL_UPSTREAM_WINNER_ENV; do
  [[ -n "${!k+x}" ]] || { echo "Stage-L winner missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEL_STAGED_VALIDATED" == 1 && "$B300_STAGEL_FINAL_ENABLED" == 1 && "$B300_STAGEL_FINAL_SPILL_FREE" == 1 ]] || { echo 'Stage L did not survive staged validation' >&2; exit 4; }
[[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || { echo 'Stage-L NGPU drift' >&2; exit 3; }
[[ "$B300_STAGEL_UPSTREAM_KIND" == "$UPSTREAM_KIND" ]] || { echo 'Stage-L upstream kind drift' >&2; exit 3; }
case "$B300_STAGEL_FINAL_PROFILE" in pb|bp|pp) ;; *) echo 'Stage-L final profile must improve on bb' >&2; exit 4;; esac
for g in "$B300_STAGEL_FINAL_SELF_GUARD" "$B300_STAGEL_FINAL_MATE_GUARD"; do case "$g" in branch|predicated) ;; *) exit 3;; esac; done
[[ -x "$B300_STAGEL_FINAL_BIN" && -x "$B300_STAGEL_CONTROL_BIN" ]] || exit 3
[[ "$B300_STAGEL_STAGE_F_ENV" == "$STAGE_F_ENV" && "$B300_STAGEL_UPSTREAM_PREPARE_ENV" == "$UPSTREAM_PREPARE_ENV" && "$B300_STAGEL_UPSTREAM_WINNER_ENV" == "$UPSTREAM_WINNER_ENV" ]] || { echo 'Stage-L input provenance drift' >&2; exit 3; }
python3 - "$B300_STAGEL_FINAL_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('Stage-L speedup gate failed')
PY

if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"
  sha256sum "$WINNER_ENV" "$STAGE_F_ENV" "$UPSTREAM_PREPARE_ENV" "$UPSTREAM_WINNER_ENV" "$UP_MANIFEST" \
    "$B300_STAGEL_FINAL_BIN" "$B300_STAGEL_CONTROL_BIN" >"$tmp"
  mv "$tmp" "$MANIFEST"
else
  [[ -s "$MANIFEST" ]] || { echo 'missing Stage-L promotion manifest' >&2; exit 3; }
fi
sha256sum -c "$MANIFEST" >/dev/null || { echo 'Stage-L promotion fingerprint mismatch' >&2; exit 3; }
FINAL_SHA="$(sha256sum "$B300_STAGEL_FINAL_BIN" | awk '{print $1}')"; CONTROL_SHA="$(sha256sum "$B300_STAGEL_CONTROL_BIN" | awk '{print $1}')"; MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
label="prefetch_guard_${B300_STAGEL_FINAL_PROFILE}_sw${SW}d${SD}_mw${MW}d${MD}_sev${SE}_mev${ME}"
control_label="prefetch_guard_bb_sw${SW}d${SD}_mw${MW}d${MD}_sev${SE}_mev${ME}"

if [[ "$PREPARE_ONLY" == 1 ]]; then
  mkdir -p "$(dirname "$PREPARE_ENV")"
  {
    printf 'B300_STAGEL_PREPARED=1\n'
    printf 'B300_STAGEL_PREPARED_MOD=%q\n' "$MOD"
    printf 'B300_STAGEL_PREPARED_NGPU=%q\n' "$NGPU"
    printf 'B300_STAGEL_PREPARED_UPSTREAM_KIND=%q\n' "$UPSTREAM_KIND"
    printf 'B300_STAGEL_PREPARED_SELF_WIDTH=%q\n' "$SW"
    printf 'B300_STAGEL_PREPARED_SELF_DISTANCE=%q\n' "$SD"
    printf 'B300_STAGEL_PREPARED_SELF_EVICT=%q\n' "$SE"
    printf 'B300_STAGEL_PREPARED_MATE_WIDTH=%q\n' "$MW"
    printf 'B300_STAGEL_PREPARED_MATE_DISTANCE=%q\n' "$MD"
    printf 'B300_STAGEL_PREPARED_MATE_EVICT=%q\n' "$ME"
    printf 'B300_STAGEL_PREPARED_PROFILE=%q\n' "$B300_STAGEL_FINAL_PROFILE"
    printf 'B300_STAGEL_PREPARED_SELF_GUARD=%q\n' "$B300_STAGEL_FINAL_SELF_GUARD"
    printf 'B300_STAGEL_PREPARED_MATE_GUARD=%q\n' "$B300_STAGEL_FINAL_MATE_GUARD"
    printf 'B300_STAGEL_PREPARED_BIN=%q\n' "$B300_STAGEL_FINAL_BIN"
    printf 'B300_STAGEL_PREPARED_LABEL=%q\n' "$label"
    printf 'B300_STAGEL_PREPARED_THREADS=%q\n' "$B300_STAGEL_FINAL_THREADS"
    printf 'B300_STAGEL_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGEL_CONTROL_BIN"
    printf 'B300_STAGEL_PREPARED_CONTROL_LABEL=%q\n' "$control_label"
    printf 'B300_STAGEL_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGEL_CONTROL_THREADS"
    printf 'B300_STAGEL_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGEL_FINAL_SPEEDUP"
    printf 'B300_STAGEL_PREPARED_UPSTREAM_MANIFEST=%q\n' "$UP_MANIFEST"
    printf 'B300_STAGEL_PREPARED_UPSTREAM_MANIFEST_SHA256=%q\n' "$UP_MANIFEST_SHA"
    printf 'B300_STAGEL_PREPARED_MANIFEST=%q\n' "$MANIFEST"
    printf 'B300_STAGEL_PREPARED_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA"
    printf 'B300_STAGEL_PREPARED_FINAL_BIN_SHA256=%q\n' "$FINAL_SHA"
    printf 'B300_STAGEL_PREPARED_CONTROL_BIN_SHA256=%q\n' "$CONTROL_SHA"
    printf 'B300_STAGEL_PREPARED_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEL_FINAL_STAGE_ROWS"
    printf 'B300_STAGEL_PREPARED_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEL_FINAL_STAGE_RESIDUE"
  } >"$PREPARE_ENV"
  cat "$PREPARE_ENV"
  echo "STAGE L PREPARED mod=$MOD ngpu=$NGPU profile=$B300_STAGEL_FINAL_PROFILE guards=$B300_STAGEL_FINAL_SELF_GUARD/$B300_STAGEL_FINAL_MATE_GUARD upstream=$UPSTREAM_KIND speedup=${B300_STAGEL_FINAL_SPEEDUP}x" >&2
  exit 0
fi

[[ "$NGPU" == 8 ]] || { echo 'Stage-L complete-prime promotion requires NGPU=8; use PREPARE_ONLY=1 for local screening' >&2; exit 2; }
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$MOD" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_STAGEL_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGEL_FINAL_THREADS" \
  FORCED_BASE_BIN="$B300_STAGEL_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGEL_CONTROL_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
