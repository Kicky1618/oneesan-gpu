#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-firstpass.sh"
LEGACY_STAGED="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh"
STAGEJ="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
VERIFY="$ONEESAN_ROOT/scripts/run/b300x8-grand-verify-exact.sh"
for f in "$GRAND" "$FIRST" "$LEGACY_STAGED" "$STAGEJ" "$PROMOTE" "$VERIFY"; do
  [[ -f "$f" ]] || { echo "missing Stage-H compatibility dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

# The old Stage-H runner remains the legacy fallback for artifacts produced
# before mate geometry was integrated into the grand selector.
for s in \
  'B300_STAGEH_STAGED_VALIDATED' \
  'B300_STAGEH_FINAL_ENABLED' \
  'B300_STAGEH_FINAL_SPILL_FREE' \
  'B300_STAGEH_INPUT_STAGE_F_ENV' \
  'sha256sum -c "$MANIFEST"' \
  'B300_STAGEH_PROMOTION_BIN_SHA256' \
  'B300_STAGEH_PREPARED_BIN' \
  'B300_STAGEH_PREPARED_CONTROL_BIN'; do
  grep -Fq "$s" "$LEGACY_STAGED" || { echo "legacy Stage-H staged marker missing: $s" >&2; exit 3; }
done

# Current integrated path is Stage J. Historical B300_GRAND_STAGEH_* fields
# are compatibility aliases of the selected Stage-J mate geometry.
for s in \
  'RUN_STAGEJ="${RUN_STAGEJ:-${RUN_STAGEH:-1}}"' \
  'STAGEJ_MIN_SPEEDUP="${STAGEJ_MIN_SPEEDUP:-${STAGEH_MIN_SPEEDUP:-1.002}}"' \
  'INPUT_ENV="$HYBRID_NS_WINNER_ENV"' \
  'B300_STAGEJ_PREPARED_BIN' \
  'MODE=stagej_mategeo_grand' \
  'B300_GRAND_STAGEJ_OK' \
  'B300_GRAND_STAGEH_OK' \
  'B300_GRAND_STAGEH_WIDTH' \
  'B300_GRAND_STAGEH_DISTANCE'; do
  grep -Fq "$s" "$GRAND" || { echo "integrated Stage-J/Stage-H alias marker missing: $s" >&2; exit 3; }
done

for s in \
  'B300_STAGEJ_STAGED_VALIDATED' \
  'B300_STAGEJ_FINAL_ENABLED' \
  'B300_STAGEJ_PREPARED' \
  'B300_STAGEJ_PREPARED_SELF_WIDTH' \
  'B300_STAGEJ_PREPARED_MATE_WIDTH' \
  'B300_STAGEJ_PREPARED_BIN' \
  'B300_STAGEJ_PREPARED_CONTROL_BIN'; do
  grep -Fq "$s" "$STAGEJ" || { echo "Stage-J staged marker missing: $s" >&2; exit 3; }
done

# Stage-H first-pass is now a compatibility normalizer. It must consume the
# integrated aliases when present and retain the legacy staged fallback.
for s in \
  'WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"' \
  'WORK_ROOT="$WORK_ROOT" PREFIX="$BASE_PREFIX"' \
  'BASE_GRAND_ENV="${BASE_GRAND_ENV:-${BASE_PREFIX}.race_grand.env}"' \
  "grep -q '^B300_GRAND_STAGEH_OK=' \"\$BASE_GRAND_ENV\"" \
  'B300_GRAND_STAGEH_REASON=integrated_in_grand' \
  'B300_GRAND_STAGEH_SELECTED_SCHEMA=1' \
  'B300_GRAND_STAGEH_SELECTED_VALIDATED=1' \
  'B300_GRAND_STAGEH_SELECTED_RACE_RESULT_SHA256=' \
  'B300_GRAND_SELECTED_SCHEMA=1' \
  'B300_GRAND_SELECTED_VALIDATED=1' \
  'B300_GRAND_SELECTED_PROFILE_SHA256=' \
  'B300_GRAND_SELECTED_BINARY_SHA256=' \
  'B300_GRAND_SELECTED_SMOKE_PRIME=' \
  'B300_GRAND_SELECTED_RACE_RESULT_SHA256=' \
  'B300_GRAND_SELECTED_FIRSTPASS_META=' \
  'normalized_contract=1' \
  'b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh'; do
  grep -Fq "$s" "$FIRST" || { echo "Stage-H compatibility first-pass marker missing: $s" >&2; exit 3; }
done

for s in \
  'selected binary fingerprint mismatch' \
  'selected profile fingerprint mismatch' \
  'single-pass TSV fingerprint mismatch' \
  'checkpoint fingerprint mismatch'; do
  grep -Fq "$s" "$PROMOTE" || { echo "hardened promotion marker missing for Stage-H compatibility: $s" >&2; exit 3; }
done
for s in 'verify_b300_exact_result.py' 'B300 GRAND VERIFY COMPLETE'; do
  grep -Fq "$s" "$VERIFY" || { echo "exact verifier wrapper marker missing: $s" >&2; exit 3; }
done

echo 'b300-grand-stageh-contract-preflight OK legacy_staged=1 integrated_stagej=1 stageh_aliases=1 no_duplicate_fullprime=1 normalized_selection=1 checkpoint_schema3=1 race_fingerprint=1 work_root=1 hardened_promotion=1 independent_verifier=1 gpu_work=0'
