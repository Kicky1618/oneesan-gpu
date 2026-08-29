#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
NEXTSELF="$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh"
HYBRID="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh"
RACE="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
JOINT="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"

for f in "$GRAND" "$NEXTSELF" "$HYBRID" "$RACE" "$JOINT"; do
  [[ -f "$f" ]] || { echo "missing grand-selector dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'B300_NEXTSELF_PREPARED=1' \
  'B300_NEXTSELF_PREPARED_BIN' \
  'B300_NEXTSELF_PREPARED_CONTROL_BIN' \
  'B300_NEXTSELF_PREPARED_CONTROL_THREADS'; do
  grep -Fq "$s" "$NEXTSELF" || { echo "next-self prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'B300_HYBRID8_PREPARED=1' \
  'B300_HYBRID8_PREPARED_BIN' \
  'B300_HYBRID8_PREPARED_BASE_BIN' \
  'B300_HYBRID8_PREPARED_MANIFEST' \
  'sha256sum -c "$MANIFEST"'; do
  grep -Fq "$s" "$HYBRID" || { echo "hybrid8 prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'PREPARE_ONLY=1 PREPARE_ENV="$JOINT_PREPARE_ENV"' \
  'PREPARE_ONLY=1' \
  'NEXTSELF_RC == 4' \
  'HYBRID_RC == 4' \
  'MODE=nextself_hybrid8_joint' \
  'P_BIN="$B300_NEXTSELF_PREPARED_BIN"' \
  'B_BIN="$B300_NEXTSELF_PREPARED_CONTROL_BIN"' \
  'E1_BIN="$B300_HYBRID8_PREPARED_BIN"' \
  'E2_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"' \
  'E3_BIN="$JOINT_PRIMARY_BIN"' \
  'MODE=joint_fallback' \
  'FORCED_EXTRA3_BIN="$E3_BIN"' \
  'B300_GRAND_DROPPED_JOINT_BASE_WHEN_BOTH'; do
  grep -Fq "$s" "$GRAND" || { echo "grand selector marker missing: $s" >&2; exit 3; }
done

for s in \
  'FORCED_EXTRA3_BIN="${FORCED_EXTRA3_BIN:-}"' \
  'HAS_FORCED_EXTRA3=0' \
  'HAS_FORCED_EXTRA3=1' \
  'smoke_forced forced_extra3' \
  'EXPECTED_OK=$((3+HAS_FORCED_BASE+HAS_FORCED_EXTRA+HAS_FORCED_EXTRA2+HAS_FORCED_EXTRA3))' \
  'FATAL single-pass residue mismatch'; do
  grep -Fq "$s" "$RACE" || { echo "external race extra3 marker missing: $s" >&2; exit 3; }
done

for s in \
  'B300_JOINT_PREPARED=1' \
  'PROFILE_FILE=%q' \
  'SMOKE_PRIME=%q' \
  'FORCED_TARGET_MIB=%q' \
  'MAX_WINDOW=%q'; do
  grep -Fq "$s" "$JOINT" || { echo "joint prepare marker missing: $s" >&2; exit 3; }
done

python3 - "$GRAND" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
# Both-transform mode must consume exactly all five forced slots once.
block=re.search(r'if \(\( NEXTSELF_OK && HYBRID_OK \)\); then(.*?)elif \(\( NEXTSELF_OK \)\); then',s,re.S)
if not block: raise SystemExit('both-transform candidate block missing')
b=block.group(1)
for name in ('P_BIN','B_BIN','E1_BIN','E2_BIN','E3_BIN'):
    if len(re.findall(rf'\b{name}=',b)) != 1:
        raise SystemExit(f'{name} must be assigned exactly once in both-transform mode')
# The low-priority joint base must not silently steal a slot in both mode.
if 'JOINT_BASE_BIN' in b:
    raise SystemExit('joint base unexpectedly occupies both-transform candidate block')
print('grand_candidate_budget=OK forced_slots=5 profiled_slots=2 total=7')
PY

echo 'b300_nextgen_grand_selector_preflight=OK bash_syntax=OK nextself_prepare=OK hybrid8_prepare=OK fingerprint=OK staged_reject_fallback=OK forced_extra3=OK candidate_budget=7 gpu_work=0 actions_triggered=0'
