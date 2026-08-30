#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
GRAND_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagem.sh"
GEN_GRAND="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stagem.py"
BASE_FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
FIRST_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stagem.sh"
GEN_FIRST="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stagem.py"
PROMOTE_M="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh"
PRE_M="$ONEESAN_ROOT/scripts/bench/b300-stagem-preflight.sh"
for f in "$BASE_GRAND" "$GRAND_WRAP" "$GEN_GRAND" "$BASE_FIRST" "$FIRST_WRAP" "$GEN_FIRST" "$PROMOTE_M" "$PRE_M"; do
  [[ -f "$f" ]] || { echo "missing Stage-M grand dependency=$f" >&2; exit 2; }
  case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac
done
bash "$PRE_M"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagem-grand.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
STAGEM_SELECTOR_BUILD_DIR="$tmp/selector" PATCH_ONLY=1 bash "$GRAND_WRAP" 27 >"$tmp/selector.env"
# shellcheck disable=SC1090
source "$tmp/selector.env"
[[ "${B300_STAGEM_SELECTOR_PATCHED:-0}" == 1 && -s "$B300_STAGEM_SELECTOR_GENERATED" ]] || exit 3
GRAND="$B300_STAGEM_SELECTOR_GENERATED"
bash -n "$GRAND"

STAGEM_FIRSTPASS_BUILD_DIR="$tmp/firstpass" PATCH_ONLY=1 bash "$FIRST_WRAP" 27 >"$tmp/firstpass.env"
# shellcheck disable=SC1090
source "$tmp/firstpass.env"
[[ "${B300_STAGEM_FIRSTPASS_PATCHED:-0}" == 1 && -s "$B300_STAGEM_FIRSTPASS_GENERATED" ]] || exit 3
FIRST="$B300_STAGEM_FIRSTPASS_GENERATED"
bash -n "$FIRST"

need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-M grand marker missing in $f: $s" >&2; exit 3; }; }
for s in \
  'STAGEM_PREFIX=' \
  'STAGEM_WINNER_ENV=' \
  'STAGEM_PREPARE_ENV=' \
  'RUN_STAGEM=' \
  'STAGEM_MIN_SPEEDUP=' \
  'STAGEM_POLICY_LIST=' \
  'b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh' \
  'PREPARE_ONLY=1' \
  'B300_STAGEM_PREPARED' \
  'B300_STAGEM_PREPARED_BIN' \
  'B300_STAGEM_PREPARED_CONTROL_BIN' \
  'Stage-M mate-load policy rejected; retaining Stage L' \
  'MODE=stagem_mateload_grand' \
  'MODE=stagem_mateload_joint' \
  'B300_GRAND_STAGEM_OK' \
  'B300_GRAND_STAGEM_POLICY' \
  'B300_GRAND_STAGEM_INTEGRATED=1'; do need "$GRAND" "$s"; done

python3 - "$GRAND" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('generated grand selector must own exactly one complete-prime race')
mi=s.find('b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh')
ri=s.find('b300x8-race-external-forced-profiled-once.sh')
if mi < 0 or ri < 0 or mi >= ri:
    raise SystemExit('Stage M must be prepared before the one complete-prime race')
for marker in ('P_BIN="$B300_STAGEM_PREPARED_BIN"','B_BIN="$B300_STAGEM_PREPARED_CONTROL_BIN"'):
    if marker not in s: raise SystemExit('Stage-M candidate replacement missing: '+marker)
if 'STAGEM_OK && NEXTSELF_OK' not in s or 'elif ((STAGEM_OK))' not in s:
    raise SystemExit('Stage-M grand/joint branch pair missing')
if '[[ "$B300_STAGEM_PREPARED_CONTROL_BIN" == "$B300_STAGEL_PREPARED_BIN" ]]' not in s:
    raise SystemExit('Stage-M exact Stage-L control binding missing')
print('stagem_single_prime_contract=OK')
PY

for s in \
  'RUN_STAGEL=' 'RUN_STAGEM=' \
  'stagel_min_speedup=' 'stagel_guard_list=' \
  'stagem_min_speedup=' 'stagem_policy_list=' \
  'b300x8-joint-nextself-hybrid8-select-stagem.sh' \
  'B300_GRAND_SELECTED_STAGEL_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEM_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEM_POLICY'; do need "$FIRST" "$s"; done

python3 - "$FIRST" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stagem.sh' not in s:
    raise SystemExit('generated firstpass does not call Stage-M selector entrypoint')
if 'b300x8-joint-nextself-hybrid8-select.sh" 27' in s:
    raise SystemExit('generated firstpass still calls raw selector directly')
if "B300_GRAND_SELECTED_SCHEMA=3" not in s:
    raise SystemExit('Stage-M firstpass must preserve exact-promotion schema 3 ABI')
print('stagem_firstpass_contract=OK')
PY

echo 'b300-grand-stagem-contract-preflight OK generated_selector=1 generated_firstpass=1 stage_m_after_l=1 prepare_only=1 fallback_l=1 forced_slots_replaced=1 single_complete_prime=1 selected_schema3=1 gpu_work=0'
