#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-stageg-firstpass.sh"
CONT="$ONEESAN_ROOT/scripts/run/b300x8-grand-stageg-continue.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
for f in "$FIRST" "$CONT" "$PROMOTE"; do
  [[ -f "$f" ]] || { echo "missing Stage-G dependency=$f" >&2; exit 2; }
  bash -n "$f"
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
  'B300_GRAND_STAGEG_SELECTED_RACE_RESULT_SHA256=' \
  'normalized_contract=1'; do
  grep -Fq "$s" "$FIRST" || { echo "Stage-G normalized contract marker missing: $s" >&2; exit 3; }
done

for s in \
  'if [[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 ]]' \
  'delegating hardened exact promotion' \
  'b300x8-grand-promote-exact.sh' \
  'WARNING: continuing from legacy Stage-G contract with reduced provenance checks'; do
  grep -Fq "$s" "$CONT" || { echo "Stage-G continuation marker missing: $s" >&2; exit 3; }
done

# New geometry-complete Stage F must remain a no-op in legacy Stage G. Otherwise
# the guarded first-pass would duplicate an already-searched B300 geometry.
for s in \
  'B300_HYBRID8_NEXTSELF_FINAL_DISTANCE+x' \
  'B300_HYBRID8_NEXTSELF_SEARCH_DISTANCES+x' \
  'B300_GRAND_STAGEG_REASON=stage_f_geometry_complete'; do
  grep -Fq "$s" "$FIRST" || { echo "Stage-G no-duplicate marker missing: $s" >&2; exit 3; }
done

echo 'b300-grand-stageg-contract-preflight OK normalized_selection=1 checkpoint_schema3=1 race_fingerprint=1 work_root=1 hardened_delegation=1 legacy_fallback=1 geometry_duplicate_guard=1 gpu_work=0'
