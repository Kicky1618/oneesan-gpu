#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
STAGEK="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh"
STAGEK_PREFLIGHT="$ONEESAN_ROOT/scripts/bench/b300-stagek-preflight.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
VERIFY="$ONEESAN_ROOT/scripts/run/b300x8-grand-verify-exact.sh"

for f in "$FIRST" "$GRAND" "$STAGEK" "$STAGEK_PREFLIGHT" "$PROMOTE" "$VERIFY"; do
  [[ -f "$f" ]] || { echo "missing integrated Stage-K dependency=$f" >&2; exit 2; }
  case "$f" in *.sh) bash -n "$f" ;; esac
done

# Reuse the deeper Stage-J/K implementation proof first.
bash "$STAGEK_PREFLIGHT"

# The canonical first-pass must expose Stage K directly and promise exactly one
# complete-prime selection race for the whole grand pipeline.
for s in \
  'RUN_STAGEK="${RUN_STAGEK:-1}"' \
  'STAGEK_MIN_SPEEDUP="${STAGEK_MIN_SPEEDUP:-1.002}"' \
  'MATE_EVICT_LIST="${MATE_EVICT_LIST:-default normal last}"' \
  'run_stagek=%s' \
  'stagek_min_speedup=%s' \
  'stagek_mate_evict_list=%s' \
  'complete_prime_races_expected=1' \
  'RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK"' \
  'b300x8-joint-nextself-hybrid8-select.sh'; do
  grep -Fq "$s" "$FIRST" || { echo "canonical first-pass Stage-K marker missing: $s" >&2; exit 3; }
done

# Stage I, J and K must all be staged before the selector's one external race.
for s in \
  'RUN_STAGEK="${RUN_STAGEK:-1}"' \
  'STAGEK_MIN_SPEEDUP="${STAGEK_MIN_SPEEDUP:-1.002}"' \
  'Stage K: refine only mate-prefetch eviction priority after Stage J geometry is fixed.' \
  'PREPARE_ONLY=1' \
  'MODE=stagek_mateevict_grand' \
  'MODE=stagek_mateevict_joint' \
  'B300_STAGEK_PREPARED_BIN' \
  'B300_STAGEK_PREPARED_CONTROL_BIN' \
  'B300_GRAND_STAGEK_OK' \
  'B300_GRAND_STAGEK_MATE_EVICT'; do
  grep -Fq "$s" "$GRAND" || { echo "integrated selector Stage-K marker missing: $s" >&2; exit 3; }
done
python3 - "$GRAND" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
# All optimization stages prepare candidates; the selector itself owns the
# only complete-prime race.
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('grand selector must contain exactly one external complete-prime race')
if 'Stage-K mate eviction rejected; retaining Stage J' not in s:
    raise SystemExit('Stage-K rejection must fall back to Stage J')
if s.find('# Stage K:') > s.find('b300x8-race-external-forced-profiled-once.sh'):
    raise SystemExit('Stage K must be prepared before complete-prime race')
print('integrated_stagek_single_prime=OK')
PY

# The post-selection path remains the shared hardened CRT continuation and
# independent verifier; Stage K must not invent a parallel exact pipeline.
for s in 'B300_GRAND_SELECTED_VALIDATED' 'B300_GRAND_SELECTED_BINARY_SHA256' 'B300_GRAND_SELECTED_RACE_RESULT_SHA256'; do
  grep -Fq "$s" "$PROMOTE" || { echo "shared promotion marker missing: $s" >&2; exit 3; }
done
grep -Fq 'verify_b300_exact_result.py' "$VERIFY" || exit 3

echo 'b300-grand-stagek-contract-preflight OK canonical_firstpass=1 stage_i_before_j_before_k=1 stagek_integrated=1 single_complete_prime=1 stagej_fallback=1 schema3_promotion=1 independent_verifier=1 gpu_work=0'
