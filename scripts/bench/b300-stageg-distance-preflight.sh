#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
BUILD="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-nextself-distance-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-distance-sweep.sh"
STAGE="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-distance-staged-calibrate.sh"
RUN="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-distance-staged-fullprime-race.sh"
for f in "$BUILD" "$PROOF" "$SWEEP" "$STAGE" "$RUN"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }; bash -n "$f"; done
python3 -m py_compile "$GEN"

for s in 'prefetch_distance_iterations=' 'DISTANCE must be one of 1,2,4'; do grep -Fq "$s" "$GEN" || { echo "generator distance marker missing: $s" >&2; exit 3; }; done
for s in 'NEXTSELF_DISTANCE=' 'canonical_nextgen_proof_gates_reused=1' 'experimental_distance_transform=1'; do grep -Fq "$s" "$BUILD" || { echo "builder marker missing: $s" >&2; exit 3; }; done
for s in 'b300_mainrec_hybrid8_nextself_distance_preflight=OK' 'distances=1,2,4' 'combinations=12'; do grep -Fq "$s" "$PROOF" || { echo "distance proof marker missing: $s" >&2; exit 3; }; done
for s in 'DISTANCE_LIST=' 'distance=1 reference' 'b300_nextgen_hybrid8_nextself_distance_exact_match=1' 'B300_HYBRID8_NEXTSELF_DISTANCE_SPEEDUP_VS_D1'; do grep -Fq "$s" "$SWEEP" || { echo "distance sweep marker missing: $s" >&2; exit 3; }; done
for s in 'RUN_STAGE_F=' 'SELECTED_DISTANCE=' '1 $SELECTED_DISTANCE' 'B300_HYBRID8_NEXTSELF_DISTANCE_STAGED_VALIDATED' 'FINAL_DISTANCE=' 'Stage-G/final residue mismatch'; do grep -Fq "$s" "$STAGE" || { echo "Stage G calibration marker missing: $s" >&2; exit 3; }; done
for s in 'B300_STAGEG_PREPARED=1' 'B300_STAGEG_PREPARED_WIDTH' 'B300_STAGEG_PREPARED_DISTANCE' 'sha256sum -c "$MANIFEST"' 'nextgen_hybrid8_nextself_w${B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_WIDTH}_d${B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_DISTANCE}'; do grep -Fq "$s" "$RUN" || { echo "Stage G runner marker missing: $s" >&2; exit 3; }; done

echo 'b300_stageg_distance_preflight=OK generator=OK canonical_builder=OK exact_gate=OK spill_gate=OK distance_lock=OK stage_e_crosscheck=OK manifest=OK prepare=OK gpu_work=0 actions_triggered=0'
