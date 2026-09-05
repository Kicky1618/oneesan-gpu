#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-stagej-firstpass-v2.sh"
PROMOTE_J="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh"
PROMOTE_EXACT="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
VERIFY="$ONEESAN_ROOT/scripts/run/b300x8-grand-verify-exact.sh"
RACE="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
for f in "$FIRST" "$PROMOTE_J" "$PROMOTE_EXACT" "$VERIFY" "$RACE"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done

for s in \
  'B300_STAGEJ_PREPARED=1' \
  'B300_STAGEJ_PROMOTION_VALIDATED=1' \
  'STAGEJ_BUILD_DIR="${STAGEJ_BUILD_DIR:-$ONEESAN_BUILD_DIR/stagej-mate-geometry}"' \
  'ONEESAN_BUILD_DIR="$STAGEJ_BUILD_DIR"' \
  "s/B300_STAGEI_/B300_STAGEJ_/g" \
  'MOD="$MOD" TARGET_MIB="$TARGET_MIB"' \
  'B300_STAGEJ_PREPARED_MOD=' \
  'sha256sum -c "$MANIFEST"' \
  'namespace_isolated=1'; do
  grep -Fq "$s" "$PROMOTE_J" || { echo "Stage-J promotion marker missing: $s" >&2; exit 3; }
done

for s in \
  'b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh' \
  'B300_STAGEJ_PREPARED:-0' \
  'B300_STAGEJ_PREPARED_MOD' \
  'B300_STAGEJ_PREPARED_MANIFEST' \
  'sha256sum -c "$B300_STAGEJ_PREPARED_MANIFEST"' \
  'FORCED_OVERRIDE_BIN="$J_BIN"' \
  'FORCED_BASE_BIN="$J_CTL"' \
  'FORCED_EXTRA_BIN="$EXTRA_BIN"' \
  'SELECT_ONLY=1' \
  "{'schema':3,'binary_sha256':b,'profile_sha256':p}" \
  'B300_GRAND_SELECTED_SCHEMA=1' \
  'B300_GRAND_SELECTED_VALIDATED=1' \
  'B300_GRAND_STAGEJ_SELECTED_SCHEMA=2' \
  'B300_GRAND_STAGEJ_MANIFEST=' \
  'namespace_isolated=1'; do
  grep -Fq "$s" "$FIRST" || { echo "Stage-J v2 marker missing: $s" >&2; exit 3; }
done

# The only Stage-I namespace allowed in the public v2 first-pass is the official
# self-eviction stage. Geometry/prepared binaries must never use Stage-I names.
if grep -Eq 'B300_STAGEI_(PREPARED_BIN|PREPARED_CONTROL_BIN|ROWS|RESIDUE|MATE_|CONTROL_|FINAL_)' "$FIRST"; then
  echo 'Stage-J v2 leaked legacy geometry Stage-I namespace' >&2; exit 4
fi
grep -Fq 'B300_STAGEI_PREPARED:-0' "$FIRST"
grep -Fq 'B300_STAGEI_PREPARED_HINT' "$FIRST"

# Exact continuation and verifier remain the single hardened post-selection path.
grep -Fq 'B300_GRAND_SELECTED_RACE_RESULT_SHA256' "$PROMOTE_EXACT"
grep -Fq 'solver_fingerprint' "$PROMOTE_EXACT"
grep -Fq 'verify_b300_exact_result.py' "$VERIFY"

echo 'b300-grand-stagej-v2-contract-preflight OK stage_i=self_eviction_only stage_j=independent_mate_geometry namespace_isolated=1 build_dir_isolated=1 modulus_pinned=1 manifest_gate=1 complete_prime_race=1 select_only=1 exact_promotion=shared verifier=shared gpu_work=0'
