#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stager.sh"; GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stager.py"; SELECTOR="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stager.sh"
for f in "$WRAP" "$GEN" "$SELECTOR"; do [[ -f "$f" ]] || exit 2; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stager-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGER_FIRSTPASS_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"; source "$tmp/env"
[[ "${B300_STAGER_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGER_FIRSTPASS_GENERATED:-}" ]] || exit 3
F="$B300_STAGER_FIRSTPASS_GENERATED"; bash -n "$F"
need(){ grep -Fq "$2" "$1" || { echo "Stage-R firstpass marker missing: $2" >&2; exit 3; }; }
for s in 'RUN_STAGER=' 'STAGER_MIN_SPEEDUP=' 'STAGER_PAIR_POLICY_LIST=' 'STAGER_BLOCK_POLICY_LIST=' 'b300-grand-stager-contract-preflight.sh' 'b300x8-joint-nextself-hybrid8-select-stager.sh' 'B300_GRAND_SELECTED_STAGER_ENABLED' 'B300_GRAND_SELECTED_STAGER_ACCEPTED' 'B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND' 'B300_GRAND_SELECTED_STAGER_PAIR_POLICY' 'B300_GRAND_SELECTED_STAGER_BLOCK_POLICY' 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1' 'B300_GRAND_SELECTED_SCHEMA=3'; do need "$F" "$s"; done
python3 - "$F" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stageq.sh" 27' in s: raise SystemExit('Stage-R firstpass still invokes Stage-Q selector directly')
if s.count("printf 'B300_GRAND_SELECTED_SCHEMA=3") != 1: raise SystemExit('Stage-R firstpass must preserve selected schema 3 ABI')
if s.count("printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1") != 1: raise SystemExit('Stage-R single-prime selected marker drift')
if 'B300_GRAND_STAGER_INTEGRATED' not in s: raise SystemExit('Stage-R summary integration gate missing')
print('stager_firstpass_contract_structure=OK')
PY
echo 'b300-grand-stager-firstpass-preflight OK selector=stager selected_schema3=1 r_provenance=1 complete_prime_races=1 gpu_work=0'
