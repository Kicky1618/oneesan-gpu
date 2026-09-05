#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SELF_GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
MATE_GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py"
H_PREFLIGHT="$ONEESAN_ROOT/scripts/bench/b300-stageh-nextmate-preflight.sh"
I_SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-evict-sweep.sh"
I_STAGE="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-evict-staged-calibrate.sh"
I_RUN="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-evict-staged-fullprime-race.sh"
REFINE="$ONEESAN_ROOT/scripts/run/b300x8-grand-refine-firstpass.sh"
CONT="$ONEESAN_ROOT/scripts/run/b300x8-grand-refine-continue.sh"
BASE_CONT="$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-continue.sh"
for f in "$H_PREFLIGHT" "$I_SWEEP" "$I_STAGE" "$I_RUN" "$REFINE" "$CONT" "$BASE_CONT"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }; bash -n "$f"; done
python3 -m py_compile "$SELF_GEN" "$MATE_GEN"
for f in "$SELF_GEN" "$MATE_GEN"; do
  for s in 'evict_priority=' 'eviction_hint_sm80_only=' 'EVICT must be one of default,normal,last'; do grep -Fq "$s" "$f" || { echo "eviction generator marker missing: $f $s" >&2; exit 3; }; done
done
for s in 'b300_evict_exact_match=1' 'B300_EVICT_DEFAULT_SPILL_FREE' 'B300_EVICT_HIGH_S' 'EVICT_LIST must include default'; do grep -Fq "$s" "$I_SWEEP" || { echo "Stage-I sweep marker missing: $s" >&2; exit 3; }; done
for s in 'B300_STAGEI_STAGED_VALIDATED' 'Stage-I/Stage-E-final residue mismatch' 'Stage-I/Stage-F-final residue mismatch' 'B300_STAGEI_FINAL_HINT'; do grep -Fq "$s" "$I_STAGE" || { echo "Stage-I staged marker missing: $s" >&2; exit 3; }; done
for s in 'B300_STAGEI_PREPARED=1' 'sha256sum -c "$MANIFEST"' 'B300_STAGEI_PROMOTION_HIGH_S'; do grep -Fq "$s" "$I_RUN" || { echo "Stage-I runner marker missing: $s" >&2; exit 3; }; done
for s in 'bash "$script" 27 >&2' 'B300_GRAND_REFINE_H_OK' 'B300_GRAND_REFINE_I_OK' 'prepare_stdout_isolated=1' 'refine checkpoint fingerprint mismatch'; do grep -Fq "$s" "$REFINE" || { echo "refine marker missing: $s" >&2; exit 3; }; done
if grep -Fq 'bash "$script" 27' "$REFINE" && ! grep -Fq 'bash "$script" 27 >&2' "$REFINE"; then echo 'prepare stdout isolation regressed' >&2; exit 3; fi
for s in 'selected profile SHA changed' 'solver_fingerprint' 'selected smoke residue mismatch'; do grep -Fq "$s" "$BASE_CONT" || { echo "continue marker missing: $s" >&2; exit 3; }; done
bash "$H_PREFLIGHT"
echo 'b300_grand_refine_preflight=OK stage_h=OK stage_i=OK eviction_ptx=OK exact_crosscheck=OK spill_gate=OK prepare_stdout_isolated=OK normalized_contract=OK schema3_continue=OK gpu_work=0 actions_triggered=0'
