#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
FIRSTPASS="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
CONTRACT="$ONEESAN_ROOT/scripts/bench/b300-grand-selector-contract-preflight.sh"
RACE="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
JOINT="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"
STAGEK="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh"

for f in "$GRAND" "$FIRSTPASS" "$CONTRACT" "$RACE" "$JOINT" "$STAGEK"; do
  [[ -f "$f" ]] || { echo "missing grand-selector dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

need(){
  local file="$1" marker="$2" label="$3"
  grep -Fq "$marker" "$file" || { echo "$label marker missing: $marker" >&2; exit 3; }
}

# Exit 4 is a candidate rejection. Any other nonzero status is fatal.
for s in \
  'PREPARE_ONLY=1 PREPARE_ENV="$JOINT_PREPARE_ENV"' \
  'NEXTSELF_RC == 4' \
  'HYBRID_RC == 4' \
  'HYBRID_NS_RC == 4' \
  'EVICT_RC == 4' \
  'STAGEJ_RC == 4' \
  'STAGEK_RC == 4'; do
  need "$GRAND" "$s" grand-selector
 done

# Stage I tunes self eviction, Stage J tunes independent mate geometry, and
# Stage K holds both geometries fixed while refining only mate eviction.
for s in \
  'RUN_STAGEI="${RUN_STAGEI:-1}"' \
  'RUN_STAGEJ="${RUN_STAGEJ:-${RUN_STAGEH:-1}}"' \
  'RUN_STAGEK="${RUN_STAGEK:-1}"' \
  'STAGEI_MIN_SPEEDUP="${STAGEI_MIN_SPEEDUP:-1.002}"' \
  'STAGEJ_MIN_SPEEDUP="${STAGEJ_MIN_SPEEDUP:-${STAGEH_MIN_SPEEDUP:-1.002}}"' \
  'STAGEK_MIN_SPEEDUP="${STAGEK_MIN_SPEEDUP:-1.002}"' \
  'MATE_EVICT_LIST="${MATE_EVICT_LIST:-default normal last}"' \
  'b300x8-nextgen-hybrid8-selfevict-prepare.sh' \
  'b300x8-nextgen-hybrid8-nextmate-geometry-stagej-staged-fullprime-race.sh' \
  'b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh' \
  'Stage-I self-eviction geometry drift' \
  'Stage-J self geometry drift' \
  'Stage-J self eviction drift' \
  'Stage-K self geometry drift' \
  'Stage-K mate geometry drift' \
  'Stage-K self eviction drift' \
  'Stage-K baseline mate eviction drift' \
  'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1' \
  'B300_GRAND_STAGEJ_INTEGRATED=1' \
  'B300_GRAND_STAGEK_INTEGRATED=1'; do
  need "$GRAND" "$s" stage-i-j-k
 done

for s in \
  'B300_GRAND_STAGEJ_OK' \
  'B300_GRAND_STAGEJ_MATE_WIDTH' \
  'B300_GRAND_STAGEJ_MATE_DISTANCE' \
  'B300_GRAND_STAGEK_OK' \
  'B300_GRAND_STAGEK_BASE_MATE_EVICT' \
  'B300_GRAND_STAGEK_MATE_EVICT' \
  'B300_GRAND_STAGEK_SEARCH_EVICTS' \
  'B300_GRAND_STAGEH_OK' \
  'B300_GRAND_STAGEH_WIDTH' \
  'B300_GRAND_STAGEH_DISTANCE' \
  'B300_GRAND_HYBRID8_NEXTSELF_WIDTH' \
  'B300_GRAND_HYBRID8_NEXTSELF_DISTANCE'; do
  need "$GRAND" "$s" summary
 done

for s in \
  'MODE=stagek_mateevict_grand' \
  'MODE=stagek_mateevict_joint' \
  'MODE=stagej_mategeo_grand' \
  'MODE=stagej_mategeo_joint' \
  'MODE=stagei_selfevict_grand' \
  'MODE=stagei_selfevict_joint' \
  'MODE=hybrid8_nextself_composed_grand' \
  'MODE=hybrid8_nextself_composed_joint' \
  'MODE=nextself_hybrid8_joint' \
  'MODE=joint_fallback' \
  'FORCED_EXTRA3_BIN="$E3_BIN"'; do
  need "$GRAND" "$s" candidate-mode
 done

for s in \
  'B300_STAGEK_PREPARED=1' \
  'B300_STAGEK_PREPARED_SELF_WIDTH' \
  'B300_STAGEK_PREPARED_MATE_WIDTH' \
  'B300_STAGEK_PREPARED_BASE_MATE_EVICT' \
  'B300_STAGEK_PREPARED_MATE_EVICT' \
  'Stage-J promotion manifest mismatch before Stage K' \
  'sha256sum -c "$MANIFEST"'; do
  need "$STAGEK" "$s" stage-k-runner
 done

need "$CONTRACT" 'b300_grand_selector_contract_preflight=OK' grand-functional-contract
for s in 'B300_JOINT_PREPARED=1' 'PROFILE_FILE=%q' 'SMOKE_PRIME=%q' 'FORCED_TARGET_MIB=%q' 'MAX_WINDOW=%q'; do
  need "$JOINT" "$s" joint-prepare
 done
for s in \
  'FORCED_EXTRA3_BIN="${FORCED_EXTRA3_BIN:-}"' \
  'HAS_FORCED_EXTRA3=1' \
  'smoke_forced forced_extra3' \
  'FATAL single-pass residue mismatch'; do
  need "$RACE" "$s" external-race
 done
need "$FIRSTPASS" 'SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS"' firstpass
need "$FIRSTPASS" 'b300x8-grand-firstpass OK' firstpass
if grep -Eq 'SELECT_ONLY=0|SELECT_ONLY="?0"?' "$FIRSTPASS"; then
  echo 'guarded grand first-pass contains a SELECT_ONLY=0 path' >&2
  exit 3
fi

# Five forced-like slots + profiled warp/orbit = seven candidates. Stronger
# stages replace the first two slots; they must not add a second complete-prime
# race or waste slots on controls already subsumed by the primary.
python3 - "$GRAND" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()

def block(pattern,label):
    m=re.search(pattern,s,re.S)
    if not m: raise SystemExit(f'{label} candidate block missing')
    b=m.group(1)
    for name in ('P_BIN','B_BIN','E1_BIN','E2_BIN','E3_BIN'):
        if len(re.findall(rf'\b{name}=',b)) != 1:
            raise SystemExit(f'{label}: {name} must be assigned exactly once')
    return b

def mapping(b,label,required,forbidden=()):
    for slot,candidate in required.items():
        if not re.search(rf'\b{slot}="\${candidate}"',b):
            raise SystemExit(f'{label}: mapping mismatch {slot}->{candidate}')
    for bad in forbidden:
        if bad in b:
            raise SystemExit(f'{label}: forbidden candidate {bad} occupies budget')

k=block(r'if \(\( STAGEK_OK && NEXTSELF_OK \)\); then(.*?)elif \(\( STAGEK_OK \)\); then','stagek+nextself')
mapping(k,'stagek+nextself',{
    'P_BIN':'B300_STAGEK_PREPARED_BIN',
    'B_BIN':'B300_STAGEK_PREPARED_CONTROL_BIN',
    'E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN',
    'E2_BIN':'B300_NEXTSELF_PREPARED_BIN',
    'E3_BIN':'JOINT_PRIMARY_BIN',
},('B300_NEXTSELF_PREPARED_CONTROL_BIN','JOINT_BASE_BIN','B300_STAGEJ_PREPARED_BIN'))

j=block(r'elif \(\( STAGEJ_OK && NEXTSELF_OK \)\); then(.*?)elif \(\( STAGEJ_OK \)\); then','stagej+nextself')
mapping(j,'stagej+nextself',{
    'P_BIN':'B300_STAGEJ_PREPARED_BIN',
    'B_BIN':'B300_STAGEJ_PREPARED_CONTROL_BIN',
    'E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN',
    'E2_BIN':'B300_NEXTSELF_PREPARED_BIN',
    'E3_BIN':'JOINT_PRIMARY_BIN',
},('B300_NEXTSELF_PREPARED_CONTROL_BIN','JOINT_BASE_BIN'))

e=block(r'elif \(\( EVICT_OK && NEXTSELF_OK \)\); then(.*?)elif \(\( EVICT_OK \)\); then','stagei+nextself')
mapping(e,'stagei+nextself',{
    'P_BIN':'B300_EVICT_BIN',
    'B_BIN':'B300_EVICT_CONTROL_BIN',
    'E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN',
    'E2_BIN':'B300_NEXTSELF_PREPARED_BIN',
    'E3_BIN':'JOINT_PRIMARY_BIN',
},('B300_NEXTSELF_PREPARED_CONTROL_BIN','JOINT_BASE_BIN'))

f=block(r'elif \(\( HYBRID_NS_OK && NEXTSELF_OK \)\); then(.*?)elif \(\( HYBRID_NS_OK \)\); then','stagef+nextself')
mapping(f,'stagef+nextself',{
    'P_BIN':'B300_HYBRID8_NEXTSELF_PREPARED_BIN',
    'B_BIN':'B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_BIN',
    'E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN',
    'E2_BIN':'B300_NEXTSELF_PREPARED_BIN',
    'E3_BIN':'JOINT_PRIMARY_BIN',
},('B300_NEXTSELF_PREPARED_CONTROL_BIN','JOINT_BASE_BIN'))

sep=block(r'elif \(\( NEXTSELF_OK && HYBRID_OK \)\); then(.*?)elif \(\( NEXTSELF_OK \)\); then','separate-nextself+hybrid8')
if 'JOINT_BASE_BIN' in sep:
    raise SystemExit('separate transform branch unexpectedly includes joint base')

print('grand_candidate_budget=OK forced_slots=5 profiled_slots=2 total=7 stagek_mapping=OK stagej_mapping=OK stagei_mapping=OK stagef_mapping=OK')
PY

echo 'b300_nextgen_grand_selector_preflight=OK bash_syntax=OK firstpass_guard=OK rejection_contract=OK stagei_selfevict=OK stagej_mate_geometry=OK stagek_mate_eviction=OK compatibility_aliases=OK candidate_budget=7 gpu_work=0 actions_triggered=0'
