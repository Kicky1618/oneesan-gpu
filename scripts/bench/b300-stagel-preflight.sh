#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SOURCE_PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-predicated-prefetch-source-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-prefetch-guard-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-prefetch-guard-staged-calibrate.sh"
RUNNER="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-prefetch-guard-stagel-staged-fullprime-race.sh"
GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
for f in "$SOURCE_PROOF" "$SWEEP" "$STAGED" "$RUNNER" "$GRAND"; do
  [[ -f "$f" ]] || { echo "missing Stage-L dependency=$f" >&2; exit 2; }
  bash -n "$f"
done
bash "$SOURCE_PROOF" >/dev/null

need(){ local f="$1" s="$2" label="$3"; grep -Fq "$s" "$f" || { echo "$label marker missing: $s" >&2; exit 3; }; }

for s in \
  'PROFILE_LIST="${PROFILE_LIST:-bb pb bp pp}"' \
  'PROFILE_LIST must include bb control' \
  'UPSTREAM_PREPARE_ENV=' \
  'UPSTREAM_WINNER_ENV=' \
  'kind=stagek' \
  'kind=stagej' \
  'SELF_GUARD="$sg" MATE_GUARD="$mg"' \
  'FATAL Stage-L residue mismatch' \
  'main_pull_kernel_ilp2' \
  'main_pull_kernel_ilp8_hybrid' \
  'clean=len(rv)>=2 and ss==0 and sl==0' \
  'B300_STAGEL_SELF_GUARD' \
  'B300_STAGEL_MATE_GUARD' \
  'B300_STAGEL_BEST_ENABLED' \
  'b300_stagel_exact_match=1'; do need "$SWEEP" "$s" sweep; done

for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'GUARD_LIST="${GUARD_LIST:-bb pb bp pp}"' \
  'B300_STAGEL_UPSTREAM_KIND' \
  'FATAL Stage-L/upstream residue mismatch' \
  'FATAL Stage-L/Stage-F residue mismatch' \
  'FATAL Stage-L/Stage-E residue mismatch' \
  'SELECTED_PROFILE="$B300_STAGEL_PROFILE"' \
  'B300_STAGEL_STAGED_VALIDATED=' \
  'B300_STAGEL_FINAL_ENABLED=' \
  'B300_STAGEL_FINAL_SELF_GUARD=' \
  'B300_STAGEL_FINAL_MATE_GUARD=' \
  'B300_STAGEL_FINAL_STAGE_RESIDUE='; do need "$STAGED" "$s" staged; done

for s in \
  'B300_STAGEL_STAGED_VALIDATED' \
  'B300_STAGEL_FINAL_ENABLED' \
  'B300_STAGEL_FINAL_SPILL_FREE' \
  'B300_STAGEL_PREPARED=1' \
  'B300_STAGEL_PREPARED_UPSTREAM_KIND' \
  'B300_STAGEL_PREPARED_PROFILE' \
  'B300_STAGEL_PREPARED_SELF_GUARD' \
  'B300_STAGEL_PREPARED_MATE_GUARD' \
  'B300_STAGEL_PREPARED_BIN' \
  'B300_STAGEL_PREPARED_CONTROL_BIN' \
  'sha256sum "$WINNER_ENV" "$STAGE_F_ENV" "$UPSTREAM_PREPARE_ENV" "$UPSTREAM_WINNER_ENV" "$UP_MANIFEST"' \
  'sha256sum -c "$MANIFEST"'; do need "$RUNNER" "$s" runner; done

for s in \
  'RUN_STAGEL="${RUN_STAGEL:-1}"' \
  'STAGEL_MIN_SPEEDUP="${STAGEL_MIN_SPEEDUP:-1.002}"' \
  'STAGEL_GUARD_LIST="${STAGEL_GUARD_LIST:-bb pb bp pp}"' \
  'b300x8-nextgen-hybrid8-prefetch-guard-stagel-staged-fullprime-race.sh' \
  'MODE=stagel_guard_grand' \
  'MODE=stagel_guard_joint' \
  'B300_GRAND_STAGEL_OK' \
  'B300_GRAND_STAGEL_SELF_GUARD' \
  'B300_GRAND_STAGEL_MATE_GUARD' \
  'B300_GRAND_STAGEL_INTEGRATED=1' \
  'B300_GRAND_COMPLETE_PRIME_RACES=1'; do need "$GRAND" "$s" grand; done

python3 - "$SWEEP" "$STAGED" "$GRAND" <<'PY'
from pathlib import Path
import re,sys
sweep,staged,grand=map(lambda p:Path(p).read_text(),sys.argv[1:])
# Search is joint over both guard axes; validation must lock the chosen profile.
if "PROFILE_LIST=\"${PROFILE_LIST:-bb pb bp pp}\"" not in sweep: raise SystemExit('Stage-L joint guard search missing')
if 'run_stage "$SEARCH_ROWS" "$GUARD_LIST"' not in staged: raise SystemExit('Stage-L search list missing')
if 'run_stage "$rows" "bb $SELECTED_PROFILE"' not in staged: raise SystemExit('Stage-L validation does not lock chosen guard profile')
# Main grand must prioritize Stage L over K/J but retain the same five forced slots.
m=re.search(r'if \(\(STAGEL_OK && NEXTSELF_OK\)\); then(.*?)elif \(\(STAGEL_OK\)\); then',grand,re.S)
if not m: raise SystemExit('Stage-L grand branch missing')
b=m.group(1)
required={'P_BIN':'B300_STAGEL_PREPARED_BIN','B_BIN':'B300_STAGEL_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'}
for slot,candidate in required.items():
    if not re.search(rf'\b{slot}=\"\${candidate}\"',b): raise SystemExit(f'Stage-L candidate mapping mismatch {slot}->{candidate}')
print('stagel_guard_contract_structure=OK')
PY

echo 'b300-stagel-preflight OK namespace=guard_only joint_profiles=bb,pb,bp,pp stagej_stagek_upstream=1 exact_residue=1 ptxas_spill=1 row_scoped_residue=1 search_rows=1 validate_rows=4,8 promotion_manifest=1 grand_integrated=1 complete_prime_races=1 gpu_work=0'