#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-O promotion targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g8_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
RUN_STAGED="${RUN_STAGED:-1}"; PREPARE_ONLY="${PREPARE_ONLY:-0}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"; BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_staged_g${NGPU}}"; WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"; RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27}"; PREPARE_ENV="${PREPARE_ENV:-${RACE_PREFIX}_prepared.env}"
for x in RUN_STAGED PREPARE_ONLY SELECT_ONLY REBUILD_BUCKETS; do [[ "${!x}" == 0 || "${!x}" == 1 ]] || exit 2; done
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for x in MOD TARGET_MIB MAX_WINDOW; do [[ "${!x}" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for f in "$PROFILE_FILE" "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-O promotion input=$f" >&2; exit 2; }; done
command -v sha256sum >/dev/null || exit 2

# Bind Stage O to the exact prepared Stage-N artifact before any GPU work.
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
for k in B300_STAGEN_PREPARED B300_STAGEN_PREPARED_MOD B300_STAGEN_PREPARED_NGPU B300_STAGEN_PREPARED_PAIR_POLICY B300_STAGEN_PREPARED_BLOCK_POLICY B300_STAGEN_PREPARED_MATE_LOAD_POLICY B300_STAGEN_PREPARED_BIN B300_STAGEN_PREPARED_THREADS B300_STAGEN_PREPARED_MANIFEST; do
  [[ -n "${!k+x}" ]] || { echo "Stage-N prepare missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEN_PREPARED" == 1 && "$B300_STAGEN_PREPARED_MOD" == "$MOD" && "$B300_STAGEN_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-O/Stage-N prepare mismatch' >&2; exit 3; }
[[ -x "$B300_STAGEN_PREPARED_BIN" && -s "$B300_STAGEN_PREPARED_MANIFEST" ]] || exit 3
sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-N manifest mismatch before Stage O promotion' >&2; exit 3; }
N_BIN="$B300_STAGEN_PREPARED_BIN"; N_THREADS="$B300_STAGEN_PREPARED_THREADS"; N_MANIFEST="$B300_STAGEN_PREPARED_MANIFEST"
N_PAIR="$B300_STAGEN_PREPARED_PAIR_POLICY"; N_BLOCK="$B300_STAGEN_PREPARED_BLOCK_POLICY"; N_MATE="$B300_STAGEN_PREPARED_MATE_LOAD_POLICY"
[[ "$N_PAIR" == cg || "$N_BLOCK" == cg ]] || { echo 'Stage O not applicable: prepared Stage N has no CG Count axis' >&2; exit 4; }

if [[ "$RUN_STAGED" == 1 ]]; then
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" MIN_SPEEDUP="$MIN_SPEEDUP" PAIR_L2_LIST="$PAIR_L2_LIST" BLOCK_L2_LIST="$BLOCK_L2_LIST" PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageo-cg-l2-staged-calibrate.sh"
fi
[[ -s "$WINNER_ENV" ]] || { echo "missing Stage-O winner=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for k in B300_STAGEO_STAGED_VALIDATED B300_STAGEO_FINAL_ENABLED B300_STAGEO_NGPU B300_STAGEO_PAIR_POLICY B300_STAGEO_BLOCK_POLICY B300_STAGEO_MATE_LOAD_POLICY B300_STAGEO_BASE_CG_L2_BYTES B300_STAGEO_BASE_PAIR_L2_BYTES B300_STAGEO_BASE_BLOCK_L2_BYTES B300_STAGEO_PAIR_L2_BYTES B300_STAGEO_BLOCK_L2_BYTES B300_STAGEO_FINAL_BIN B300_STAGEO_FINAL_THREADS B300_STAGEO_FINAL_SPEEDUP B300_STAGEO_FINAL_SPILL_FREE B300_STAGEO_CONTROL_BIN B300_STAGEO_CONTROL_THREADS B300_STAGEO_FINAL_STAGE_ROWS B300_STAGEO_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-O winner missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEO_STAGED_VALIDATED" == 1 && "$B300_STAGEO_FINAL_ENABLED" == 1 && "$B300_STAGEO_FINAL_SPILL_FREE" == 1 && "$B300_STAGEO_NGPU" == "$NGPU" ]] || { echo 'Stage O did not survive staged validation' >&2; exit 4; }
[[ "$B300_STAGEO_PAIR_POLICY" == "$N_PAIR" && "$B300_STAGEO_BLOCK_POLICY" == "$N_BLOCK" && "$B300_STAGEO_MATE_LOAD_POLICY" == "$N_MATE" ]] || { echo 'Stage-O changed Stage-N load policies' >&2; exit 3; }
[[ "$B300_STAGEO_CONTROL_BIN" == "$N_BIN" ]] || { echo 'Stage-O control is not exact prepared Stage N' >&2; exit 3; }
for b in "$B300_STAGEO_BASE_CG_L2_BYTES" "$B300_STAGEO_BASE_PAIR_L2_BYTES" "$B300_STAGEO_BASE_BLOCK_L2_BYTES" "$B300_STAGEO_PAIR_L2_BYTES" "$B300_STAGEO_BLOCK_L2_BYTES"; do case "$b" in 0|64|128|256) ;; *) exit 3;; esac; done
[[ "$B300_STAGEO_PAIR_L2_BYTES" != "$B300_STAGEO_BASE_PAIR_L2_BYTES" || "$B300_STAGEO_BLOCK_L2_BYTES" != "$B300_STAGEO_BASE_BLOCK_L2_BYTES" ]] || { echo 'Stage-O promoted inherited L2 baseline unchanged' >&2; exit 4; }
[[ "$N_PAIR" == cg || "$B300_STAGEO_PAIR_L2_BYTES" == 0 ]] || exit 3
[[ "$N_BLOCK" == cg || "$B300_STAGEO_BLOCK_L2_BYTES" == 0 ]] || exit 3
python3 - "$B300_STAGEO_FINAL_SPEEDUP" "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('Stage-O speedup below threshold')
PY
[[ -x "$B300_STAGEO_FINAL_BIN" ]] || exit 3

if [[ "$RUN_STAGED" == 1 ]]; then
  tmp="${MANIFEST}.tmp"; mkdir -p "$(dirname "$MANIFEST")"
  sha256sum "$WINNER_ENV" "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$N_MANIFEST" "$B300_STAGEO_FINAL_BIN" "$B300_STAGEO_CONTROL_BIN" >"$tmp"; mv "$tmp" "$MANIFEST"
else
  [[ -s "$MANIFEST" ]] || { echo 'missing Stage-O promotion manifest' >&2; exit 3; }
fi
sha256sum -c "$MANIFEST" >/dev/null || { echo 'Stage-O promotion fingerprint mismatch' >&2; exit 3; }
FINAL_SHA="$(sha256sum "$B300_STAGEO_FINAL_BIN" | awk '{print $1}')"; CONTROL_SHA="$(sha256sum "$B300_STAGEO_CONTROL_BIN" | awk '{print $1}')"; MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"; N_MANIFEST_SHA="$(sha256sum "$N_MANIFEST" | awk '{print $1}')"
label="stageo_pairl2_${B300_STAGEO_PAIR_L2_BYTES}_blockl2_${B300_STAGEO_BLOCK_L2_BYTES}"; control_label=stageo_stagen_control
mkdir -p "$(dirname "$PREPARE_ENV")"
{
  printf 'B300_STAGEO_PREPARED=1\n'; printf 'B300_STAGEO_PREPARED_MOD=%q\n' "$MOD"; printf 'B300_STAGEO_PREPARED_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEO_PREPARED_STAGEN_WINNER_ENV=%q\n' "$STAGEN_WINNER_ENV"; printf 'B300_STAGEO_PREPARED_STAGEN_PREPARE_ENV=%q\n' "$STAGEN_PREPARE_ENV"; printf 'B300_STAGEO_PREPARED_STAGEN_MANIFEST=%q\n' "$N_MANIFEST"; printf 'B300_STAGEO_PREPARED_STAGEN_MANIFEST_SHA256=%q\n' "$N_MANIFEST_SHA"
  printf 'B300_STAGEO_PREPARED_PAIR_POLICY=%q\n' "$N_PAIR"; printf 'B300_STAGEO_PREPARED_BLOCK_POLICY=%q\n' "$N_BLOCK"; printf 'B300_STAGEO_PREPARED_MATE_LOAD_POLICY=%q\n' "$N_MATE"
  printf 'B300_STAGEO_PREPARED_BASE_CG_L2_BYTES=%q\n' "$B300_STAGEO_BASE_CG_L2_BYTES"; printf 'B300_STAGEO_PREPARED_BASE_PAIR_L2_BYTES=%q\n' "$B300_STAGEO_BASE_PAIR_L2_BYTES"; printf 'B300_STAGEO_PREPARED_BASE_BLOCK_L2_BYTES=%q\n' "$B300_STAGEO_BASE_BLOCK_L2_BYTES"
  printf 'B300_STAGEO_PREPARED_PAIR_L2_BYTES=%q\n' "$B300_STAGEO_PAIR_L2_BYTES"; printf 'B300_STAGEO_PREPARED_BLOCK_L2_BYTES=%q\n' "$B300_STAGEO_BLOCK_L2_BYTES"
  printf 'B300_STAGEO_PREPARED_BIN=%q\n' "$B300_STAGEO_FINAL_BIN"; printf 'B300_STAGEO_PREPARED_BIN_SHA256=%q\n' "$FINAL_SHA"; printf 'B300_STAGEO_PREPARED_LABEL=%q\n' "$label"; printf 'B300_STAGEO_PREPARED_THREADS=%q\n' "$B300_STAGEO_FINAL_THREADS"
  printf 'B300_STAGEO_PREPARED_CONTROL_BIN=%q\n' "$B300_STAGEO_CONTROL_BIN"; printf 'B300_STAGEO_PREPARED_CONTROL_BIN_SHA256=%q\n' "$CONTROL_SHA"; printf 'B300_STAGEO_PREPARED_CONTROL_LABEL=%q\n' "$control_label"; printf 'B300_STAGEO_PREPARED_CONTROL_THREADS=%q\n' "$B300_STAGEO_CONTROL_THREADS"
  printf 'B300_STAGEO_PREPARED_STAGED_SPEEDUP=%q\n' "$B300_STAGEO_FINAL_SPEEDUP"; printf 'B300_STAGEO_PREPARED_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEO_FINAL_STAGE_ROWS"; printf 'B300_STAGEO_PREPARED_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEO_FINAL_STAGE_RESIDUE"
  printf 'B300_STAGEO_PREPARED_MANIFEST=%q\n' "$MANIFEST"; printf 'B300_STAGEO_PREPARED_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA"
} >"$PREPARE_ENV"
if [[ "$PREPARE_ONLY" == 1 ]]; then cat "$PREPARE_ENV"; echo "STAGE O PREPARED pair_l2=$B300_STAGEO_PAIR_L2_BYTES block_l2=$B300_STAGEO_BLOCK_L2_BYTES speedup=${B300_STAGEO_FINAL_SPEEDUP}x env=$PREPARE_ENV" >&2; exit 0; fi
[[ "$NGPU" == 8 ]] || { echo 'Stage-O complete-prime promotion requires NGPU=8; use PREPARE_ONLY=1 for screening' >&2; exit 2; }
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$MOD" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" FORCED_OVERRIDE_BIN="$B300_STAGEO_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_STAGEO_FINAL_THREADS" FORCED_BASE_BIN="$B300_STAGEO_CONTROL_BIN" FORCED_BASE_LABEL="$control_label" FORCED_BASE_THREADS="$B300_STAGEO_CONTROL_THREADS" REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
