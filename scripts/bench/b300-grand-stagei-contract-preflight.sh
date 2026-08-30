#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-stagei-firstpass.sh"
PROMOTE_I="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-staged-fullprime-race.sh"
PROMOTE_EXACT="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
VERIFY="$ONEESAN_ROOT/scripts/run/b300x8-grand-verify-exact.sh"
RACE="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
for f in "$FIRST" "$PROMOTE_I" "$PROMOTE_EXACT" "$VERIFY" "$RACE"; do
  [[ -f "$f" ]] || { echo "missing Stage-I dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'existing grand first-pass (includes integrated Stage H)' \
  'STAGE_F_ENV="${STAGE_F_ENV:-${BASE_PREFIX}.hybrid8-nextself_winner.env}"' \
  'b300x8-nextgen-hybrid8-nextmate-geometry-staged-fullprime-race.sh' \
  'PREPARE_ONLY=1' \
  'B300_STAGEI_PREPARED=1' \
  'BASE_RUNTIME" == forced' \
  'FORCED_OVERRIDE_BIN="$B300_STAGEI_PREPARED_BIN"' \
  'FORCED_BASE_BIN="$B300_STAGEI_PREPARED_CONTROL_BIN"' \
  'FORCED_EXTRA_BIN="$EXTRA_BIN"' \
  'SELECT_ONLY=1' \
  'Stage-I grand final residue mismatch' \
  "{'schema':3,'binary_sha256':bsha,'profile_sha256':psha}" \
  'B300_GRAND_SELECTED_SCHEMA=1' \
  'B300_GRAND_SELECTED_VALIDATED=1' \
  'B300_GRAND_SELECTED_PROFILE_SHA256=' \
  'B300_GRAND_SELECTED_BINARY_SHA256=' \
  'B300_GRAND_SELECTED_RACE_RESULT_SHA256=' \
  'B300_GRAND_SELECTED_FIRSTPASS_META=' \
  'B300_GRAND_STAGEI_SELECTED_VALIDATED=1' \
  'B300_GRAND_STAGEI_SELF_WIDTH=' \
  'B300_GRAND_STAGEI_SELF_DISTANCE=' \
  'B300_GRAND_STAGEI_MATE_WIDTH=' \
  'B300_GRAND_STAGEI_MATE_DISTANCE=' \
  'normalized_contract=1'; do
  grep -Fq "$s" "$FIRST" || { echo "grand Stage-I marker missing: $s" >&2; exit 3; }
done

# The previous grand candidate may consume at most one extra forced slot. Warp
# and orbit winners are already supplied by the profiled race and must not be
# duplicated as FORCED_EXTRA.
python3 - "$FIRST" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
block=re.search(r'EXTRA_BIN=""; EXTRA_LABEL=""; EXTRA_THREADS=256(.*?)echo "=== Stage I grand final race',s,re.S)
if not block: raise SystemExit('Stage-I extra-candidate block missing')
b=block.group(1)
if 'BASE_RUNTIME" == forced' not in b: raise SystemExit('previous grand candidate is not forced-only')
if b.count('EXTRA_BIN="$BASE_BIN"') != 1: raise SystemExit('previous grand forced candidate assignment must be unique')
if 'BASE_RUNTIME" == warp' in b or 'BASE_RUNTIME" == orbit' in b:
    raise SystemExit('profiled previous winner must not consume forced extra slot')
print('stagei_candidate_budget=OK forced_slots<=3 profiled_slots=2')
PY

for s in \
  'sha256sum -c "$MANIFEST"' \
  'B300_STAGEI_PROMOTION_INPUT_STAGE_F_ENV_SHA256' \
  'B300_STAGEI_PREPARED_BIN=' \
  'B300_STAGEI_PREPARED_CONTROL_BIN='; do
  grep -Fq "$s" "$PROMOTE_I" || { echo "Stage-I promotion marker missing: $s" >&2; exit 3; }
done
for s in \
  'selected binary fingerprint mismatch' \
  'selected profile fingerprint mismatch' \
  'single-pass TSV fingerprint mismatch' \
  'checkpoint fingerprint mismatch'; do
  grep -Fq "$s" "$PROMOTE_EXACT" || { echo "exact promotion marker missing: $s" >&2; exit 3; }
done
grep -Fq 'verify_b300_exact_result.py' "$VERIFY" || { echo 'independent verifier wrapper missing' >&2; exit 3; }

# External race must retain the exact residue agreement gate for primary/base/
# extra plus profiled candidates before any Stage-I winner can be normalized.
for s in 'FATAL single-pass residue mismatch' 'SELECT_ONLY=1: selected'; do
  grep -Fq "$s" "$RACE" || { echo "external race gate missing: $s" >&2; exit 3; }
done

echo 'b300-grand-stagei-contract-preflight OK base_grand_preserved=1 stagei_prepare=1 forced_candidate_budget<=3 profiled_candidates=2 complete_prime_residue_gate=1 checkpoint_schema3=1 normalized_selection=1 hardened_exact_promotion=1 independent_verifier=1 gpu_work=0'
