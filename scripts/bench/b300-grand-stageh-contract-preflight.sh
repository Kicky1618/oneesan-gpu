#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-firstpass.sh"
STAGED="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
VERIFY="$ONEESAN_ROOT/scripts/run/b300x8-grand-verify-exact.sh"
for f in "$FIRST" "$STAGED" "$PROMOTE" "$VERIFY"; do
  [[ -f "$f" ]] || { echo "missing Stage-H dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'B300_STAGEH_STAGED_VALIDATED' \
  'B300_STAGEH_FINAL_ENABLED' \
  'B300_STAGEH_FINAL_SPILL_FREE' \
  'B300_STAGEH_INPUT_STAGE_F_ENV' \
  'sha256sum -c "$MANIFEST"' \
  'B300_STAGEH_PROMOTION_BIN_SHA256' \
  'B300_STAGEH_PREPARED_BIN' \
  'B300_STAGEH_PREPARED_CONTROL_BIN'; do
  grep -Fq "$s" "$STAGED" || { echo "Stage-H staged marker missing: $s" >&2; exit 3; }
done

for s in \
  'WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"' \
  'WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX"' \
  'WORK_ROOT="$WORK_ROOT"' \
  'BEST_WORK="$WORK_ROOT/b300_exact_singlepass_' \
  'FINAL_RESULT_SHA="$(sha256sum "$FINAL_RESULT"' \
  "{'schema':3,'binary_sha256':bsha,'profile_sha256':psha}" \
  'B300_GRAND_SELECTED_SCHEMA=1' \
  'B300_GRAND_SELECTED_VALIDATED=1' \
  'B300_GRAND_SELECTED_PROFILE_SHA256=' \
  'B300_GRAND_SELECTED_BINARY_SHA256=' \
  'B300_GRAND_SELECTED_SMOKE_PRIME=' \
  'B300_GRAND_SELECTED_RACE_RESULT_SHA256=' \
  'B300_GRAND_SELECTED_FIRSTPASS_META=' \
  'B300_GRAND_STAGEH_SELECTED_RACE_RESULT_SHA256=' \
  'normalized_contract=1'; do
  grep -Fq "$s" "$FIRST" || { echo "Stage-H normalized contract marker missing: $s" >&2; exit 3; }
done

for s in \
  'selected binary fingerprint mismatch' \
  'selected profile fingerprint mismatch' \
  'single-pass TSV fingerprint mismatch' \
  'checkpoint fingerprint mismatch'; do
  grep -Fq "$s" "$PROMOTE" || { echo "hardened promotion marker missing for Stage H: $s" >&2; exit 3; }
done
for s in \
  'verify_b300_exact_result.py' \
  'B300 GRAND VERIFY COMPLETE'; do
  grep -Fq "$s" "$VERIFY" || { echo "exact verifier wrapper marker missing: $s" >&2; exit 3; }
done

echo 'b300-grand-stageh-contract-preflight OK staged_manifest=1 normalized_selection=1 checkpoint_schema3=1 race_fingerprint=1 work_root=1 hardened_promotion=1 independent_verifier=1 gpu_work=0'
