#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-staget.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-staget.py"
SELECTOR="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-staget.sh"
for f in "$WRAP" "$GEN" "$SELECTOR"; do [[ -f "$f" ]] || exit 2; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-staget-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGET_FIRSTPASS_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"
# shellcheck disable=SC1090
source "$tmp/env"
[[ "${B300_STAGET_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGET_FIRSTPASS_GENERATED:-}" ]] || exit 3
F="$B300_STAGET_FIRSTPASS_GENERATED"; bash -n "$F"
need(){ grep -Fq "$2" "$1" || { echo "Stage-T firstpass marker missing: $2" >&2; exit 3; }; }
for s in \
  'RUN_STAGET=' 'STAGET_MIN_SPEEDUP=' 'STAGET_POLICY_LIST=' \
  'b300-staget-promotion-preflight.sh' 'b300-grand-staget-contract-preflight.sh' 'b300x8-joint-nextself-hybrid8-select-staget.sh' \
  'B300_GRAND_SELECTED_STAGET_ENABLED' 'B300_GRAND_SELECTED_STAGET_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGET_UPSTREAM_KIND' 'B300_GRAND_SELECTED_STAGET_STAGER_UPSTREAM_KIND' \
  'B300_GRAND_SELECTED_STAGET_LOW_PAIR_POLICY' 'B300_GRAND_SELECTED_STAGET_LOW_BLOCK_POLICY' \
  'B300_GRAND_SELECTED_STAGET_LOW_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGET_LOW_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_STAGET_HIGH_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_STAGET_HIGH_MATE_POLICY' 'B300_GRAND_SELECTED_STAGET_HIGH_MATE_L2_BYTES' \
  'B300_GRAND_SELECTED_STAGET_POLICY' 'B300_GRAND_SELECTED_STAGET_STAGED_SPEEDUP' \
  'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1' 'B300_GRAND_SELECTED_SCHEMA=3'; do need "$F" "$s"; done
python3 - "$F" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stages.sh" 27' in s:
    raise SystemExit('Stage-T firstpass still invokes Stage-S selector directly')
if s.count("printf 'B300_GRAND_SELECTED_SCHEMA=3") != 1:
    raise SystemExit('Stage-T firstpass must preserve selected schema 3 ABI')
if s.count("printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1") != 1:
    raise SystemExit('Stage-T single-prime selected marker drift')
if 'B300_GRAND_STAGET_INTEGRATED' not in s:
    raise SystemExit('Stage-T summary integration gate missing')
if s.find('B300_GRAND_SELECTED_STAGET_ACCEPTED') > s.find("printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1"):
    raise SystemExit('Stage-T selected provenance must precede stable prime marker')
if 'RUN_STAGER RUN_STAGES RUN_STAGET; do' not in s:
    raise SystemExit('Stage-T boolean validation chain missing')
print('staget_firstpass_contract_structure=OK')
PY
echo 'b300-grand-staget-firstpass-preflight OK selector=staget selected_schema3=1 t_provenance=1 priority=T,S,R complete_prime_races=1 gpu_work=0'
