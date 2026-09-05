#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SOURCE_PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-predicated-prefetch-source-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-guard-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-prefetch-guard-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-prefetch-guard-stagel-staged-fullprime-race.sh"
GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
STAGEJ="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh"
STAGEK="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh"
RACE="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
for f in "$SOURCE_PROOF" "$SWEEP" "$STAGED" "$PROMOTE" "$GRAND" "$STAGEJ" "$STAGEK" "$RACE"; do [[ -f "$f" ]] || { echo "missing Stage-L guard dependency=$f" >&2; exit 2; }; bash -n "$f"; done
bash "$SOURCE_PROOF"
for s in \
  'MOD="${MOD:-4294967291}"' \
  'B300_STAGEK_PREPARED_MOD' \
  'B300_STAGEJ_PREPARED_MOD' \
  'Stage-L/upstream modulus mismatch' \
  'sha256sum -c "$UP_MANIFEST"' \
  'B300_STAGEL_PREPARED_MOD' \
  'B300_STAGEL_PREPARED_UPSTREAM_MANIFEST_SHA256' \
  "Stage-L complete-prime promotion requires NGPU=8" \
  'SMOKE_PRIME="$MOD"'; do
  grep -Fq "$s" "$PROMOTE" || { echo "Stage-L promotion marker missing: $s" >&2; exit 3; }
done
for s in \
  'B300_STAGEL_STAGED_VALIDATED' \
  'B300_STAGEL_UPSTREAM_KIND' \
  'B300_STAGEL_FINAL_SELF_GUARD' \
  'B300_STAGEL_FINAL_MATE_GUARD' \
  'B300_STAGEL_UPSTREAM_PREPARE_ENV' \
  'B300_STAGEL_UPSTREAM_WINNER_ENV' \
  'b300-nextgen-hybrid8-guard-sweep.sh'; do
  grep -Fq "$s" "$STAGED" || { echo "Stage-L staged marker missing: $s" >&2; exit 3; }
done
for s in \
  'RUN_STAGEK="${RUN_STAGEK:-1}"' \
  'RUN_STAGEL="${RUN_STAGEL:-1}"' \
  'STAGEL_UPSTREAM_KIND=stagek' \
  'STAGEL_UPSTREAM_KIND=stagej' \
  'PREPARE_ONLY=1' \
  'Stage-L guard policy rejected; retaining Stage J/K' \
  'MODE=stagel_guard_grand' \
  'MODE=stagel_guard_joint'; do
  grep -Fq "$s" "$GRAND" || { echo "grand Stage-L marker missing: $s" >&2; exit 3; }
done
python3 - "$GRAND" "$PROMOTE" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); p=Path(sys.argv[2]).read_text()
# All Stage J/K/L promotions are staged/prepare-only inside the grand selector;
# the selector itself owns the only complete-prime race of this candidate set.
for needle in (
    'b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh',
    'b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh',
    'b300x8-nextgen-hybrid8-prefetch-guard-stagel-staged-fullprime-race.sh',
):
    pos=s.find(needle)
    if pos < 0: raise SystemExit('missing staged refinement runner '+needle)
    prefix=s[max(0,pos-900):pos]
    if 'PREPARE_ONLY=1' not in prefix: raise SystemExit('refinement runner is not prepare-only: '+needle)
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('grand selector must contain exactly one explicit final single-pass race')
if 'SMOKE_PRIME="$MOD"' not in p:
    raise SystemExit('Stage-L standalone full-prime must use staged modulus')
print('stagel_guard_promotion_contract=OK')
PY
echo 'b300_stagel_guard_promotion_preflight=OK source_predicate=OK exact=OK spill=OK stagej_mod=OK stagek_mod=OK upstream_manifest=OK ngpu=OK smoke_prime=OK staged_fallback=OK single_fullprime=OK gpu_work=0 actions_triggered=0'
