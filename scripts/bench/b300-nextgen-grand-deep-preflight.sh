#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

checks=(
  scripts/bench/b300-nextgen-preflight.sh
  scripts/bench/b300-mainrec-hybrid8-nextself-transform-preflight.sh
  scripts/bench/b300-mainrec-hybrid8-nextself-distance-preflight.sh
  scripts/bench/b300-hybrid8-nextself-stagef-crosscheck-preflight.sh
  scripts/bench/b300-nextgen-grand-selector-preflight.sh
  scripts/bench/b300-grand-selector-contract-preflight.sh
  scripts/bench/b300-mainrec-hybrid8-next-mate-preflight.sh
  scripts/bench/b300-stageh-nextmate-preflight.sh
)
for rel in "${checks[@]}"; do
  f="$ONEESAN_ROOT/$rel"
  [[ -f "$f" ]] || { echo "deep preflight missing $rel" >&2; exit 2; }
  bash -n "$f"
  echo "=== deep preflight: $rel ===" >&2
  bash "$f"
done

for rel in \
  scripts/run/b300x8-grand-firstpass.sh \
  scripts/run/b300x8-grand-stageh-firstpass.sh \
  scripts/run/b300x8-grand-stageh-continue.sh \
  scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh; do
  f="$ONEESAN_ROOT/$rel"; [[ -f "$f" ]] || { echo "deep preflight missing $rel" >&2; exit 2; }; bash -n "$f"
done

grep -Fq 'SELECT_ONLY=1' "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
grep -Fq 'B300_GRAND_STAGEH_SELECTED_VALIDATED=1' "$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-firstpass.sh"
grep -Fq "fp.get('schema')!=3" "$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-continue.sh"
grep -Fq 'sha256sum -c "$MANIFEST"' "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh"

echo 'b300_nextgen_grand_deep_preflight=OK geometry=OK stage_e_f_crosscheck=OK distance=OK next_mate=OK stage_h=OK grand_contract=OK schema3_continue=OK gpu_work=0 actions_triggered=0'
