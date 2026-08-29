#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

meta="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextgen-hybrid8-select.sh"
joint="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"
hybrid="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh"
hybrid_stage="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-staged-calibrate.sh"
next="$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh"
next_stage="$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-nextself-staged-ab.sh"
race="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"

for f in "$meta" "$joint" "$hybrid" "$hybrid_stage" "$next" "$next_stage" "$race"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'PREPARE_ONLY=1' \
  'B300_JOINT_PREPARED' \
  'b300-nextgen-hybrid8-staged-calibrate.sh' \
  'B300_HYBRID8_STAGED_VALIDATED' \
  'B300_HYBRID8_FINAL_ENABLED' \
  'HYBRID_MANIFEST="${HYBRID_MANIFEST:-${HYBRID_WINNER_ENV%.env}_promotion-inputs.sha256}"' \
  'sha256sum "$HYBRID_WINNER_ENV" "$B300_HYBRID8_FINAL_BIN" "$B300_HYBRID8_BASE_BIN"' \
  'sha256sum -c "$HYBRID_MANIFEST"' \
  'MANIFEST="$HYBRID_MANIFEST"' \
  'b300x8-ilp8-nextself-staged-ab.sh' \
  'B300_NEXTSELF_STAGED_VALIDATED' \
  'NEXTSELF_MANIFEST="${NEXTSELF_MANIFEST:-${NEXTSELF_WINNER_ENV%.env}_promotion-inputs.sha256}"' \
  'sha256sum "$NEXTSELF_WINNER_ENV" "$B300_NEXTSELF_CONTROL_BIN" "$B300_NEXTSELF_BIN"' \
  'sha256sum -c "$NEXTSELF_MANIFEST"' \
  'PREPARE_ONLY=1 WINNER_ENV="$NEXTSELF_WINNER_ENV"' \
  'B300_NEXTSELF_PREPARED' \
  'JOINT STAGED SUMMARY' \
  'FORCED_EXTRA_BIN="$JOINT_PRIMARY_BIN"' \
  'FORCED_EXTRA2_BIN="$JOINT_BASE_BIN"' \
  'FORCED_EXTRA3_BIN="$NEXTSELF_RACE_BIN"' \
  'b300x8-nextgen-hybrid8-staged-fullprime-race.sh' \
  'promote prepared next-self into unified full-prime race' \
  'staged transforms rejected; run calibrated joint/bucket race' \
  'b300x8-race-external-forced-profiled-once.sh'; do
  grep -Fq "$s" "$meta" || { echo "joint nextgen marker missing: $s" >&2; exit 3; }
done
if grep -Fq 'nextself_binary_sha256=' "$meta"; then
  echo 'joint selector still carries obsolete hand-written next-self adapter' >&2
  exit 3
fi

for s in \
  'B300_JOINT_PREPARED=1' \
  'FORCED_OVERRIDE_BIN=%q' \
  'FORCED_BASE_BIN=%q' \
  'PROFILE_FILE=%q'; do
  grep -Fq "$s" "$joint" || { echo "joint prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'B300_HYBRID8_STAGED_VALIDATED' \
  'B300_HYBRID8_FINAL_ENABLED' \
  'B300_HYBRID8_FINAL_BIN' \
  'B300_HYBRID8_BASE_BIN' \
  'B300_HYBRID8_FINAL_STAGE_ROWS' \
  'B300_HYBRID8_FINAL_STAGE_RESIDUE' \
  'MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"' \
  'staged hybrid8 promotion fingerprint mismatch'; do
  grep -Fq "$s" "$hybrid" || { echo "hybrid promotion marker missing: $s" >&2; exit 4; }
done

for s in \
  'b300_nextgen_cgl2_calibrate_exact_gates' \
  'b300_nextgen_hybrid8_exact_intermediate_match' \
  'if [[ "$rows" == "$SEARCH_ROWS" && "$stage_res" != "$CORE_RES" ]]' \
  'B300_HYBRID8_FINAL_STAGE_ROWS' \
  'B300_HYBRID8_FINAL_STAGE_RESIDUE'; do
  grep -Fq "$s" "$hybrid_stage" || { echo "hybrid staged marker missing: $s" >&2; exit 4; }
done

grep -Fq 'B300_NEXTSELF_STAGED_VALIDATED=1' "$next_stage" || { echo 'next-self staged exact gate marker missing' >&2; exit 4; }
for s in \
  'PREPARE_ONLY=' \
  'B300_NEXTSELF_PREPARED=1' \
  'B300_NEXTSELF_PREPARED_BIN=' \
  'sub(/^backend=[^ ]+ /,"backend=gridfp-b300-hbm32-forced2window-opt-batch ")'; do
  grep -Fq "$s" "$next" || { echo "next-self prepare marker missing: $s" >&2; exit 4; }
done

for s in \
  'FORCED_EXTRA_BIN=' \
  'FORCED_EXTRA2_BIN=' \
  'FORCED_EXTRA3_BIN=' \
  'HAS_FORCED_EXTRA3=' \
  'smoke_forced forced_extra3 ' \
  '3+HAS_FORCED_BASE+HAS_FORCED_EXTRA+HAS_FORCED_EXTRA2+HAS_FORCED_EXTRA3' \
  'BEST_BIN" == "$FORCED_EXTRA3_BIN'; do
  grep -Fq "$s" "$race" || { echo "single-pass extra marker missing: $s" >&2; exit 4; }
done

echo 'b300_joint_nextgen_hybrid8_preflight=OK prepare=OK staged_abcd=OK hybrid8_gate=OK row_scoped_residue=OK hybrid8_fingerprint=OK nextself_fingerprint=OK nextself_prepare=OK backend_normalization=OK fallback=OK extras=3 unified_fullprime_max7=OK gpu_work=0 actions_triggered=0'
