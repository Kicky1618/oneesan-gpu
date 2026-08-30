#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagen.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stagen.py"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh"
for f in "$BASE" "$WRAP" "$GEN" "$PROMOTE"; do
  [[ -f "$f" ]] || { echo "missing Stage-N grand dependency=$f" >&2; exit 2; }
  case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagen-grand.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
STAGEN_SELECTOR_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/selector.env"
# shellcheck disable=SC1090
source "$tmp/selector.env"
[[ "${B300_STAGEN_SELECTOR_PATCHED:-0}" == 1 && -s "$B300_STAGEN_SELECTOR_GENERATED" ]] || exit 3
SEL="$B300_STAGEN_SELECTOR_GENERATED"
bash -n "$SEL"

need(){ local s="$1"; grep -Fq "$s" "$SEL" || { echo "Stage-N grand marker missing: $s" >&2; exit 3; }; }
for s in \
  'STAGEN_PREFIX=' 'STAGEN_WINNER_ENV=' 'STAGEN_PREPARE_ENV=' 'STAGEN_RACE_PREFIX=' \
  'RUN_STAGEN=' 'STAGEN_MIN_SPEEDUP=' 'STAGEN_PAIR_POLICY_LIST=' 'STAGEN_BLOCK_POLICY_LIST=' \
  'b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh' \
  'UPSTREAM_KIND="$STAGEN_UPSTREAM_KIND"' 'PREPARE_ONLY=1' \
  'Stage-N control is not exact Stage-M winner' 'Stage-N control is not exact Stage-L winner' \
  'Stage-N pair/block load policy rejected; retaining Stage M/L' \
  'MODE=stagen_pairblock_grand' 'MODE=stagen_pairblock_joint' \
  'B300_GRAND_STAGEN_OK' 'B300_GRAND_STAGEN_UPSTREAM_KIND' \
  'B300_GRAND_STAGEN_PAIR_POLICY' 'B300_GRAND_STAGEN_BLOCK_POLICY' \
  'B300_GRAND_STAGEN_INTEGRATED=1' 'B300_GRAND_COMPLETE_PRIME_RACES=1'; do
  need "$s"
done

python3 - "$SEL" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-N selector must retain exactly one complete-prime race')
ni=s.find('b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh')
ri=s.find('b300x8-race-external-forced-profiled-once.sh')
if ni < 0 or ri < 0 or ni >= ri:
    raise SystemExit('Stage N must prepare before complete-prime race')
if s.find('STAGEN_OK=0') < s.find('STAGEM_OK=0'):
    raise SystemExit('Stage N must be downstream of Stage M')

def block(pattern,label):
    m=re.search(pattern,s,re.S)
    if not m: raise SystemExit(label+' branch missing')
    return m.group(1)

def require_mapping(b,label):
    req={
      'P_BIN':'B300_STAGEN_PREPARED_BIN',
      'B_BIN':'B300_STAGEN_PREPARED_CONTROL_BIN',
      'E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN',
      'E2_BIN':'B300_NEXTSELF_PREPARED_BIN',
      'E3_BIN':'JOINT_PRIMARY_BIN',
    }
    for slot,var in req.items():
        if not re.search(rf'\b{slot}="\${var}"',b):
            raise SystemExit(f'{label}: mapping mismatch {slot}->{var}')
    for bad in ('B300_NEXTSELF_PREPARED_CONTROL_BIN','B300_STAGEM_PREPARED_BIN','B300_STAGEL_PREPARED_BIN','JOINT_BASE_BIN'):
        if bad in b: raise SystemExit(f'{label}: redundant candidate {bad}')

g=block(r'if \(\(STAGEN_OK && NEXTSELF_OK\)\); then(.*?)elif \(\(STAGEN_OK\)\); then','Stage-N grand')
require_mapping(g,'Stage-N grand')
j=block(r'elif \(\(STAGEN_OK\)\); then(.*?)elif \(\(STAGEM_OK && NEXTSELF_OK\)\); then','Stage-N joint')
for slot,var in {'P_BIN':'B300_STAGEN_PREPARED_BIN','B_BIN':'B300_STAGEN_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'JOINT_PRIMARY_BIN'}.items():
    if not re.search(rf'\b{slot}="\${var}"',j): raise SystemExit(f'Stage-N joint mapping mismatch {slot}->{var}')
if 'STAGEN_UPSTREAM_KIND=stagem' not in s or 'STAGEN_UPSTREAM_KIND=stagel' not in s:
    raise SystemExit('Stage-N M/L upstream fallback missing')
print('stagen_grand_structure=OK candidate_budget=7 single_prime=1 fallback_ml=1')
PY

echo "b300-grand-stagen-contract-preflight OK selector_native=${B300_STAGEN_SELECTOR_NATIVE:-0} stage_n_after_m_l=1 exact_upstream_control=1 candidate_budget=7 single_complete_prime=1 gpu_work=0"
