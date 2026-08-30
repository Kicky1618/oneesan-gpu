#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-evict-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-evict-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh"
GRAND="$ONEESAN_ROOT/scripts/run/b300x8-grand-stagek-firstpass.sh"
CONTINUE="$ONEESAN_ROOT/scripts/run/b300x8-grand-stagek-continue.sh"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh"
STAGEJ="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh"
for f in "$SWEEP" "$STAGED" "$PROMOTE" "$GRAND" "$CONTINUE" "$BUILDER" "$STAGEJ"; do [[ -f "$f" ]] || { echo "missing Stage-K dependency=$f" >&2; exit 2; }; bash -n "$f"; done
for s in \
  'EVICT_LIST="${EVICT_LIST:-default normal last}"' \
  'b300_stagek_exact_match=1' \
  'B300_STAGEK_BASE_SPILL_FREE' \
  'B300_STAGEK_SPILL_FREE' \
  'forward_high_s' \
  'reverse_high_s' \
  'b300-forced-nextgen-hybrid8-self-mate-geometry.sh'; do grep -Fq "$s" "$SWEEP" || { echo "Stage-K sweep marker missing: $s" >&2; exit 3; }; done
for s in \
  'B300_STAGEJ_FINAL_STAGE_RESIDUE' \
  'B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE' \
  'B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE' \
  'B300_STAGEK_EVICT" != "$SELECTED_EV' \
  'B300_STAGEK_FINAL_MATE_EVICT'; do grep -Fq "$s" "$STAGED" || { echo "Stage-K staged marker missing: $s" >&2; exit 3; }; done
for s in \
  'B300_STAGEJ_PREPARED_MOD' \
  'Stage-K/Stage-J modulus mismatch' \
  'sha256sum -c "$B300_STAGEJ_PREPARED_MANIFEST"' \
  'sha256sum "$WINNER_ENV" "$STAGE_F_ENV" "$STAGEJ_WINNER_ENV" "$STAGEJ_PREPARE_ENV" "$B300_STAGEJ_PREPARED_MANIFEST" "$B300_STAGEK_FINAL_BIN" "$B300_STAGEK_CONTROL_BIN"' \
  'B300_STAGEK_PROMOTION_MOD=' \
  'B300_STAGEK_PROMOTION_STAGEJ_MANIFEST_SHA256=' \
  'B300_STAGEK_PREPARED_MOD=' \
  'B300_STAGEK_PREPARED_STAGEJ_MANIFEST=' \
  'B300_STAGEK_PREPARED=1' \
  'B300_STAGEK_PROMOTION_MATE_EVICT' \
  'Stage-K promotion fingerprint mismatch'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-K promotion marker missing: $s" >&2; exit 3; }; done
for s in \
  'b300x8-grand-firstpass.sh' \
  'b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh' \
  'b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh' \
  'One complete-prime race only' \
  'FORCED_EXTRA2_BIN="$E2_BIN"' \
  'B300_GRAND_STAGEK_PROMOTED' \
  'solver_fingerprint' \
  "{'schema':3,'binary_sha256':bsha,'profile_sha256':psha}"; do grep -Fq "$s" "$GRAND" || { echo "Stage-K grand marker missing: $s" >&2; exit 3; }; done
for s in 'B300_GRAND_SELECTED_VALIDATED' 'B300_GRAND_SELECTED_PROFILE_SHA256' 'selected schema-3 checkpoint fingerprint mismatch'; do grep -Fq "$s" "$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-continue.sh" || { echo "continue contract marker missing: $s" >&2; exit 3; }; done
python3 - "$GRAND" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
# Stage J and K are PREPARE_ONLY before the final single-pass race. This keeps
# the newly added refinement chain to one extra complete prime.
if s.count('PREPARE_ONLY=1') < 2: raise SystemExit('Stage J/K must both stage with PREPARE_ONLY=1')
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1: raise SystemExit('Stage-K grand must have exactly one explicit final complete-prime race')
if 'K_RC==4' not in s or 'retaining Stage J candidate' not in s: raise SystemExit('Stage-K rejection fallback missing')
print('stagek_single_fullprime_contract=OK')
PY
echo 'b300_stagek_preflight=OK bash_syntax=OK exact=OK spill=OK high_s=OK staged_lock=OK stagej_modulus=OK stagej_manifest=OK manifest=OK stagej_fallback=OK single_fullprime=OK normalized_continue=OK gpu_work=0 actions_triggered=0'
