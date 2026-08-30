#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
FIRSTPASS="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
CONTRACT="$ONEESAN_ROOT/scripts/bench/b300-grand-selector-contract-preflight.sh"
NEXTSELF="$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh"
HYBRID="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh"
HYBRID_NS_GEOMETRY="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-geometry-sweep.sh"
HYBRID_NS_STAGE="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-staged-calibrate.sh"
HYBRID_NS_RUN="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-staged-fullprime-race.sh"
HYBRID_NS_PREFLIGHT="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-nextself-transform-preflight.sh"
STAGEH_RUN="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh"
RACE="$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh"
JOINT="$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh"

for f in "$GRAND" "$FIRSTPASS" "$CONTRACT" "$NEXTSELF" "$HYBRID" "$HYBRID_NS_GEOMETRY" "$HYBRID_NS_STAGE" "$HYBRID_NS_RUN" "$HYBRID_NS_PREFLIGHT" "$STAGEH_RUN" "$RACE" "$JOINT"; do
  [[ -f "$f" ]] || { echo "missing grand-selector dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'B300_NEXTSELF_PREPARED=1' \
  'B300_NEXTSELF_PREPARED_BIN' \
  'B300_NEXTSELF_PREPARED_CONTROL_BIN' \
  'B300_NEXTSELF_PREPARED_CONTROL_THREADS'; do
  grep -Fq "$s" "$NEXTSELF" || { echo "next-self prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'B300_HYBRID8_PREPARED=1' \
  'B300_HYBRID8_PREPARED_BIN' \
  'B300_HYBRID8_PREPARED_BASE_BIN' \
  'B300_HYBRID8_PREPARED_MANIFEST' \
  'sha256sum -c "$MANIFEST"'; do
  grep -Fq "$s" "$HYBRID" || { echo "hybrid8 prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'WIDTH_LIST="${WIDTH_LIST:-1 2 4 8}"' \
  'DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"' \
  'b300_nextgen_hybrid8_nextself_geometry_sweep=1' \
  'B300_HYBRID8_NEXTSELF_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_BEST_DISTANCE' \
  'geometry control lacks known spill-free ILP2/ILP8 ptxas'; do
  grep -Fq "$s" "$HYBRID_NS_GEOMETRY" || { echo "geometry sweep marker missing: $s" >&2; exit 3; }
done

for s in \
  'B300_HYBRID8_NEXTSELF_STAGED_VALIDATED' \
  'B300_HYBRID8_NEXTSELF_FINAL_ENABLED' \
  'B300_HYBRID8_NEXTSELF_FINAL_WIDTH' \
  'B300_HYBRID8_NEXTSELF_FINAL_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_FINAL_BIN' \
  'B300_HYBRID8_NEXTSELF_CONTROL_BIN' \
  'B300_HYBRID8_NEXTSELF_FINAL_SPILL_FREE=1' \
  'B300_HYBRID8_NEXTSELF_CONTROL_SPILL_FREE=1' \
  'B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS' \
  'B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE' \
  'b300-nextgen-hybrid8-nextself-geometry-sweep.sh' \
  'FATAL Stage-F geometry changed during validation' \
  'B300_HYBRID8_NEXTSELF_SEARCH_DISTANCES' \
  'stage_e_crosscheck=1' \
  'geometry_locked=1'; do
  grep -Fq "$s" "$HYBRID_NS_STAGE" || { echo "Stage F geometry marker missing: $s" >&2; exit 3; }
done

for s in \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'sha256sum "$WINNER_ENV" "$B300_HYBRID8_NEXTSELF_STAGE_E_ENV" "$B300_HYBRID8_NEXTSELF_FINAL_BIN" "$B300_HYBRID8_NEXTSELF_CONTROL_BIN"' \
  'sha256sum -c "$MANIFEST"' \
  'B300_HYBRID8_NEXTSELF_PROMOTION_WIDTH' \
  'B300_HYBRID8_NEXTSELF_PROMOTION_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_PREPARED=1' \
  'B300_HYBRID8_NEXTSELF_PREPARED_WIDTH' \
  'B300_HYBRID8_NEXTSELF_PREPARED_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_PREPARED_BIN' \
  'B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_BIN' \
  'B300_HYBRID8_NEXTSELF_PREPARED_MANIFEST' \
  'nextgen_hybrid8_nextself_w${B300_HYBRID8_NEXTSELF_FINAL_WIDTH}_d${B300_HYBRID8_NEXTSELF_FINAL_DISTANCE}_t${B300_HYBRID8_NEXTSELF_THRESHOLD}'; do
  grep -Fq "$s" "$HYBRID_NS_RUN" || { echo "Stage F geometry prepare marker missing: $s" >&2; exit 3; }
done

for s in \
  'B300_STAGEH_STAGED_VALIDATED' \
  'B300_STAGEH_FINAL_ENABLED' \
  'B300_STAGEH_FINAL_SPILL_FREE' \
  'B300_STAGEH_PREPARED=1' \
  'B300_STAGEH_PREPARED_WIDTH' \
  'B300_STAGEH_PREPARED_DISTANCE' \
  'B300_STAGEH_PREPARED_BIN' \
  'B300_STAGEH_PREPARED_CONTROL_BIN' \
  'sha256sum -c "$MANIFEST"'; do
  grep -Fq "$s" "$STAGEH_RUN" || { echo "Stage H prepare marker missing: $s" >&2; exit 3; }
done

grep -Fq 'b300-mainrec-hybrid8-nextself-transform-preflight OK' "$HYBRID_NS_PREFLIGHT" || { echo 'hybrid8 next-self transform preflight marker missing' >&2; exit 3; }
grep -Fq 'b300_grand_selector_contract_preflight=OK' "$CONTRACT" || { echo 'grand functional contract marker missing' >&2; exit 3; }

for s in \
  'PREPARE_ONLY=1 PREPARE_ENV="$JOINT_PREPARE_ENV"' \
  'NEXTSELF_RC == 4' \
  'HYBRID_RC == 4' \
  'HYBRID_NS_RC == 4' \
  'STAGEH_RC == 4' \
  'RUN_STAGEH="${RUN_STAGEH:-1}"' \
  'STAGEH_MIN_SPEEDUP="${STAGEH_MIN_SPEEDUP:-1.002}"' \
  'b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh' \
  'INPUT_ENV="$HYBRID_NS_WINNER_ENV"' \
  'Stage-H prepared geometry does not match Stage-F geometry' \
  'MODE=stageh_nextmate_grand' \
  'MODE=stageh_nextmate_joint' \
  'MODE=hybrid8_nextself_composed_grand' \
  'MODE=hybrid8_nextself_composed_joint' \
  'MODE=nextself_hybrid8_joint' \
  'MODE=joint_fallback' \
  'FORCED_EXTRA3_BIN="$E3_BIN"' \
  'B300_GRAND_STAGEH_OK' \
  'B300_GRAND_STAGEH_WIDTH' \
  'B300_GRAND_STAGEH_DISTANCE' \
  'B300_GRAND_HYBRID8_NEXTSELF_SEARCH_WIDTHS' \
  'B300_GRAND_HYBRID8_NEXTSELF_SEARCH_DISTANCES' \
  'B300_GRAND_HYBRID8_NEXTSELF_WIDTH' \
  'B300_GRAND_HYBRID8_NEXTSELF_DISTANCE'; do
  grep -Fq "$s" "$GRAND" || { echo "grand selector Stage-H marker missing: $s" >&2; exit 3; }
done
if grep -Fq 'Stage F control does not match prepared plain hybrid8 binary' "$GRAND"; then echo 'grand selector still contains invalid byte-identical rebuild requirement' >&2; exit 3; fi

for s in \
  'SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS"' \
  'run_stageh=%s' \
  'stageh_min_speedup=%s' \
  'RUN_STAGEH="$RUN_STAGEH"' \
  'STAGEH_MIN_SPEEDUP="$STAGEH_MIN_SPEEDUP"' \
  'B300_GRAND_SELECTED_STAGEH_ENABLED=' \
  'B300_GRAND_SELECTED_STAGEH_MIN_SPEEDUP=' \
  'b300x8-grand-firstpass OK'; do
  grep -Fq "$s" "$FIRSTPASS" || { echo "grand first-pass Stage-H provenance missing: $s" >&2; exit 3; }
done
if grep -Eq 'SELECT_ONLY=0|SELECT_ONLY="?0"?' "$FIRSTPASS"; then echo 'guarded grand first-pass contains a SELECT_ONLY=0 path' >&2; exit 3; fi

for s in \
  'FORCED_EXTRA3_BIN="${FORCED_EXTRA3_BIN:-}"' \
  'HAS_FORCED_EXTRA3=0' \
  'HAS_FORCED_EXTRA3=1' \
  'smoke_forced forced_extra3' \
  'EXPECTED_OK=$((3+HAS_FORCED_BASE+HAS_FORCED_EXTRA+HAS_FORCED_EXTRA2+HAS_FORCED_EXTRA3))' \
  'FATAL single-pass residue mismatch'; do
  grep -Fq "$s" "$RACE" || { echo "external race extra3 marker missing: $s" >&2; exit 3; }
done

for s in 'B300_JOINT_PREPARED=1' 'PROFILE_FILE=%q' 'SMOKE_PRIME=%q' 'FORCED_TARGET_MIB=%q' 'MAX_WINDOW=%q'; do
  grep -Fq "$s" "$JOINT" || { echo "joint prepare marker missing: $s" >&2; exit 3; }
done

python3 - "$GRAND" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
def block(pattern,label):
    m=re.search(pattern,s,re.S)
    if not m: raise SystemExit(f'{label} candidate block missing')
    b=m.group(1)
    for name in ('P_BIN','B_BIN','E1_BIN','E2_BIN','E3_BIN'):
        if len(re.findall(rf'\b{name}=',b)) != 1: raise SystemExit(f'{label}: {name} must be assigned exactly once')
    return b

def mapping(b,label,required,forbidden=()):
    for slot,candidate in required.items():
        if not re.search(rf'\b{slot}="\${candidate}"',b): raise SystemExit(f'{label}: mapping mismatch {slot}->{candidate}')
    for bad in forbidden:
        if bad in b: raise SystemExit(f'{label}: forbidden candidate {bad} occupies budget')

h=block(r'if \(\( STAGEH_OK && NEXTSELF_OK \)\); then(.*?)elif \(\( STAGEH_OK \)\); then','stageh+nextself')
mapping(h,'stageh+nextself',{
 'P_BIN':'B300_STAGEH_PREPARED_BIN','B_BIN':'B300_STAGEH_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'},('B300_NEXTSELF_PREPARED_CONTROL_BIN','JOINT_BASE_BIN'))

c=block(r'elif \(\( HYBRID_NS_OK && NEXTSELF_OK \)\); then(.*?)elif \(\( HYBRID_NS_OK \)\); then','geometry+nextself')
mapping(c,'geometry+nextself',{
 'P_BIN':'B300_HYBRID8_NEXTSELF_PREPARED_BIN','B_BIN':'B300_HYBRID8_NEXTSELF_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'},('B300_NEXTSELF_PREPARED_CONTROL_BIN','JOINT_BASE_BIN'))

b2=block(r'elif \(\( NEXTSELF_OK && HYBRID_OK \)\); then(.*?)elif \(\( NEXTSELF_OK \)\); then','separate-nextself+hybrid8')
if 'JOINT_BASE_BIN' in b2: raise SystemExit('separate transform branch unexpectedly includes joint base')
print('grand_candidate_budget=OK forced_slots=5 profiled_slots=2 total=7 stageh_mapping=OK geometry_mapping=OK')
PY

echo 'b300_nextgen_grand_selector_preflight=OK bash_syntax=OK firstpass_guard=OK firstpass_stageh_provenance=OK provenance=OK nextself_prepare=OK hybrid8_prepare=OK hybrid8_nextself_geometry=OK stageh_prepare=OK stageh_geometry_match=OK stageh_fallback=OK candidate_budget=7 gpu_work=0 actions_triggered=0'
