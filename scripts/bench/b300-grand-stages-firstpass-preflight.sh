#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stages.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stages.py"
SELECTOR="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stages.sh"
for f in "$WRAP" "$GEN" "$SELECTOR"; do [[ -f "$f" ]] || exit 2; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stages-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGES_FIRSTPASS_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"
# shellcheck disable=SC1090
source "$tmp/env"
[[ "${B300_STAGES_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGES_FIRSTPASS_GENERATED:-}" ]] || exit 3
F="$B300_STAGES_FIRSTPASS_GENERATED"; bash -n "$F"
need(){ grep -Fq "$2" "$1" || { echo "Stage-S firstpass marker missing: $2" >&2; exit 3; }; }
for s in \
  'RUN_STAGES=' 'STAGES_MIN_SPEEDUP=' 'STAGES_PAIR_L2_LIST=' 'STAGES_BLOCK_L2_LIST=' \
  'b300-grand-stages-contract-preflight.sh' 'b300x8-joint-nextself-hybrid8-select-stages.sh' \
  'B300_GRAND_SELECTED_STAGES_ENABLED' 'B300_GRAND_SELECTED_STAGES_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND' 'B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY' 'B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY' \
  'B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1' 'B300_GRAND_SELECTED_SCHEMA=3'; do need "$F" "$s"; done
python3 - "$F" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stager.sh" 27' in s:
    raise SystemExit('Stage-S firstpass still invokes Stage-R selector directly')
if s.count("printf 'B300_GRAND_SELECTED_SCHEMA=3") != 1:
    raise SystemExit('Stage-S firstpass must preserve selected schema 3 ABI')
if s.count("printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1") != 1:
    raise SystemExit('Stage-S single-prime selected marker drift')
if 'B300_GRAND_STAGES_INTEGRATED' not in s:
    raise SystemExit('Stage-S summary integration gate missing')
if s.find('B300_GRAND_SELECTED_STAGES_ACCEPTED') > s.find("printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1"):
    raise SystemExit('Stage-S selected provenance must precede stable prime marker')
print('stages_firstpass_contract_structure=OK')
PY
echo 'b300-grand-stages-firstpass-preflight OK selector=stages selected_schema3=1 s_provenance=1 complete_prime_races=1 gpu_work=0'
