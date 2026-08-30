#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stageq.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stageq.py"
SELECTOR="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageq.sh"
for f in "$WRAP" "$GEN" "$SELECTOR"; do [[ -f "$f" ]] || { echo "missing Stage-Q firstpass dependency=$f" >&2; exit 2; }; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageq-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEQ_FIRSTPASS_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"
# shellcheck disable=SC1090
source "$tmp/env"
[[ "${B300_STAGEQ_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGEQ_FIRSTPASS_GENERATED:-}" ]] || exit 3
F="$B300_STAGEQ_FIRSTPASS_GENERATED"; bash -n "$F"
need(){ grep -Fq "$2" "$1" || { echo "Stage-Q firstpass marker missing: $2" >&2; exit 3; }; }
for s in \
  'RUN_STAGEQ=' 'STAGEQ_MIN_SPEEDUP=' 'STAGEQ_PAIR_L2_LIST=' 'STAGEQ_BLOCK_L2_LIST=' \
  'b300-grand-stageq-contract-preflight.sh' 'b300x8-joint-nextself-hybrid8-select-stageq.sh' \
  'B300_GRAND_SELECTED_STAGEQ_ENABLED' 'B300_GRAND_SELECTED_STAGEQ_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND' 'B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1' 'B300_GRAND_SELECTED_SCHEMA=3'; do need "$F" "$s"; done
python3 - "$F" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stagep.sh" 27' in s:
    raise SystemExit('Stage-Q firstpass still invokes Stage-P selector directly')
if s.count("printf 'B300_GRAND_SELECTED_SCHEMA=3") != 1:
    raise SystemExit('Stage-Q firstpass must preserve selected schema 3 ABI')
if s.count("printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1") != 1:
    raise SystemExit('Stage-Q firstpass single-prime selected marker drift')
if 'B300_GRAND_STAGEQ_INTEGRATED' not in s:
    raise SystemExit('Stage-Q summary integration gate missing')
print('stageq_firstpass_contract_structure=OK')
PY
echo 'b300-grand-stageq-firstpass-preflight OK selector=stageq selected_schema3=1 q_provenance=1 complete_prime_races=1 gpu_work=0'
