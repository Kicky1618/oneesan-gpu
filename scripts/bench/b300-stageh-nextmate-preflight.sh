#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-next-mate-preflight.sh"
BUILD="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-mate.sh"
AB="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-ab.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-staged-calibrate.sh"
RUN="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-staged-fullprime-race.sh"
GRAND="$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-firstpass.sh"
for f in "$PROOF" "$BUILD" "$AB" "$STAGED" "$RUN" "$GRAND"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }; bash -n "$f"; done
python3 -m py_compile "$GEN"
for s in 'b300_mainrec_hybrid8_next_mate_prefetch=1' 'prefetch_distance_iterations=' 'semantics_unchanged=1'; do grep -Fq "$s" "$GEN" || { echo "generator marker missing: $s" >&2; exit 3; }; done
for s in 'b300_mainrec_hybrid8_next_mate_preflight=OK' 'self_plus_mate=1' 'combinations=12'; do grep -Fq "$s" "$PROOF" || { echo "proof marker missing: $s" >&2; exit 3; }; done
for s in 'self_builder_proof_gates_reused=1' 'next_mate_transform=1' 'extra_shared_bytes=0'; do grep -Fq "$s" "$BUILD" || { echo "builder marker missing: $s" >&2; exit 3; }; done
for s in 'b300_stageh_exact_match=1' 'B300_STAGEH_SELF_SPILL_FREE' 'B300_STAGEH_MATE_SPILL_FREE' 'B300_STAGEH_MATE_HIGH_S' 'B300_STAGEH_SPEEDUP'; do grep -Fq "$s" "$AB" || { echo "A/B marker missing: $s" >&2; exit 3; }; done
for s in 'B300_STAGEH_STAGED_VALIDATED' 'Stage-H/Stage-E-final residue mismatch' 'Stage-H/Stage-F-final residue mismatch' 'B300_STAGEH_FINAL_HIGH_S'; do grep -Fq "$s" "$STAGED" || { echo "staged marker missing: $s" >&2; exit 3; }; done
for s in 'B300_STAGEH_PREPARED=1' 'sha256sum -c "$MANIFEST"' 'B300_STAGEH_PROMOTION_HIGH_S' 'nextgen_hybrid8_nextself_mate_w${B300_STAGEH_FINAL_WIDTH}_d${B300_STAGEH_FINAL_DISTANCE}'; do grep -Fq "$s" "$RUN" || { echo "runner marker missing: $s" >&2; exit 3; }; done
for s in 'Stage H grand final race' 'grand_previous_${BASE_BACKEND}' 'B300_GRAND_STAGEH_SELECTED_VALIDATED=1' 'B300_GRAND_STAGEH_HIGH_S'; do grep -Fq "$s" "$GRAND" || { echo "grand marker missing: $s" >&2; exit 3; }; done

echo 'b300_stageh_nextmate_preflight=OK generator=OK transform_proof=OK exact_gate=OK spill_gate=OK high_s=OK staged_crosscheck=OK manifest=OK previous_grand_retained=OK gpu_work=0 actions_triggered=0'
