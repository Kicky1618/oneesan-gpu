#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

race="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
joint="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"
meta="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-select.sh"
next="$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh"
stage="$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-nextself-staged-ab.sh"

for f in "$race" "$joint" "$meta" "$next" "$stage"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'FORCED_EXTRA_BIN=' \
  'FORCED_EXTRA2_BIN=' \
  'HAS_FORCED_EXTRA=' \
  'HAS_FORCED_EXTRA2=' \
  'smoke_forced forced_extra ' \
  'smoke_forced forced_extra2 ' \
  '3+HAS_FORCED_BASE+HAS_FORCED_EXTRA+HAS_FORCED_EXTRA2' \
  'BEST_BIN" == "$FORCED_EXTRA_BIN' \
  'BEST_BIN" == "$FORCED_EXTRA2_BIN'; do
  grep -Fq "$s" "$race" || { echo "single-pass extra-candidate marker missing: $s" >&2; exit 3; }
done

for s in \
  'PREPARE_ONLY=' \
  'PREPARE_ENV=' \
  'B300_JOINT_PREPARED=1' \
  'FORCED_OVERRIDE_BIN=%q' \
  'FORCED_BASE_BIN=%q' \
  'PROFILE_FILE=%q'; do
  grep -Fq "$s" "$joint" || { echo "joint prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'PREPARE_ONLY=1' \
  'B300_JOINT_PREPARED' \
  'b300x8-ilp8-nextself-staged-ab.sh' \
  'B300_NEXTSELF_STAGED_VALIDATED' \
  'FORCED_EXTRA_BIN="$JOINT_PRIMARY_BIN"' \
  'FORCED_EXTRA2_BIN="$JOINT_BASE_BIN"' \
  'RUN_STAGED=0' \
  'b300x8-ilp8-nextself-staged-fullprime-race.sh' \
  'staged candidate rejected; run calibrated joint/bucket race' \
  'b300x8-race-external-forced-profiled-once.sh'; do
  grep -Fq "$s" "$meta" || { echo "joint next-self marker missing: $s" >&2; exit 4; }
done

grep -Fq 'FORCED_OVERRIDE_BIN="$NEXTSELF_ADAPTER"' "$next" || { echo 'next-self promotion lost primary adapter' >&2; exit 4; }
grep -Fq 'FORCED_BASE_BIN="$CONTROL_ADAPTER"' "$next" || { echo 'next-self promotion lost control adapter' >&2; exit 4; }
grep -Fq 'B300_NEXTSELF_STAGED_VALIDATED=1' "$stage" || { echo 'next-self staged gate marker missing' >&2; exit 4; }

echo 'b300_joint_nextself_preflight=OK extras=2 prepare_only=OK staged_gate=OK fallback=OK unified_fullprime=OK gpu_work=0 actions_triggered=0'
