#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-mate-load-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-mate-load-policy.sh"
TRANSFORM_PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-mate-load-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-staged-calibrate.sh"
STAGEK="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-evict-stagek-staged-fullprime-race.sh"
for f in "$GEN"; do python3 -m py_compile "$f"; done
for f in "$BUILDER" "$TRANSFORM_PROOF" "$SWEEP" "$STAGED" "$STAGEK"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
bash "$TRANSFORM_PROOF"

for s in \
  "policy not in ('cg', 'cs')" \
  "intrinsic = '__ldcg' if policy == 'cg' else '__ldcs'" \
  'scope=ilp8_mate_reads_only lanes=8' \
  'helper_before_kernel=1' \
  'ilp2_unchanged=1' \
  'mate_writes_unchanged=1'; do
  grep -Fq "$s" "$GEN" || { echo "Stage-L generator marker missing: $s" >&2; exit 3; }
done

for s in \
  'MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-${POLICY:-cg}}"' \
  'b300-forced-nextgen-hybrid8-self-mate-geometry.sh' \
  'gen-b300-mainrec-hybrid8-mate-load-policy.py' \
  'mate_load_scope=ilp8_only' \
  'control_bin=' \
  'source_before_policy=' \
  'ilp2_unchanged=1 mate_writes_unchanged=1'; do
  grep -Fq "$s" "$BUILDER" || { echo "Stage-L builder marker missing: $s" >&2; exit 3; }
done

for s in \
  'POLICY_LIST="${POLICY_LIST:-default cg cs}"' \
  "POLICY_LIST must include default" \
  'B300_STAGEK_STAGED_VALIDATED' \
  'B300_STAGEK_FINAL_SPILL_FREE' \
  'B300_STAGEK_PREPARED_MANIFEST' \
  'sha256sum -c "$B300_STAGEK_PREPARED_MANIFEST"' \
  'FATAL Stage-L residue mismatch' \
  'main_pull_kernel_ilp2' \
  'main_pull_kernel_ilp8_hybrid' \
  'clean=len(rv)>=2 and ss==0 and sl==0' \
  'B300_STAGEL_POLICY' \
  'B300_STAGEL_SPILL_FREE' \
  'B300_STAGEL_BEST_ENABLED' \
  'b300_stagel_exact_match=1'; do
  grep -Fq "$s" "$SWEEP" || { echo "Stage-L sweep marker missing: $s" >&2; exit 3; }
done

for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'POLICY_LIST="${POLICY_LIST:-default cg cs}"' \
  'FATAL Stage-L/Stage-K residue mismatch' \
  'FATAL Stage-L/Stage-F residue mismatch' \
  'FATAL Stage-L/Stage-E residue mismatch' \
  'SELECTED_POLICY="$B300_STAGEL_POLICY"' \
  '"$B300_STAGEL_POLICY" != "$SELECTED_POLICY"' \
  'B300_STAGEL_STAGED_VALIDATED' \
  'B300_STAGEL_FINAL_ENABLED' \
  'B300_STAGEL_FINAL_SPILL_FREE=1' \
  'B300_STAGEL_FINAL_STAGE_ROWS' \
  'B300_STAGEL_FINAL_STAGE_RESIDUE' \
  'B300_STAGEL_SEARCH_POLICIES' \
  'B300_STAGEL_INPUT_STAGEK_PREPARE_ENV'; do
  grep -Fq "$s" "$STAGED" || { echo "Stage-L staged marker missing: $s" >&2; exit 3; }
done

python3 - "$SWEEP" "$STAGED" <<'PY'
from pathlib import Path
import sys
sweep=Path(sys.argv[1]).read_text(); staged=Path(sys.argv[2]).read_text()
# Control is exact Stage-K prepared binary; only cg/cs variants are newly built.
if "printf 'default\\t%s\\t-\\n' \"$CONTROL_BIN\"" not in sweep:
    raise SystemExit('Stage-L default must be exact Stage-K control')
if "[[ \"$policy\" == default ]] && continue" not in sweep:
    raise SystemExit('Stage-L must not rebuild default policy')
# Search can examine both cg/cs; validation narrows to default + one selected policy.
if 'run_stage "$SEARCH_ROWS" "$POLICY_LIST"' not in staged:
    raise SystemExit('Stage-L search policy set missing')
if 'run_stage "$rows" "default $SELECTED_POLICY"' not in staged:
    raise SystemExit('Stage-L validation must lock selected policy')
# Known residues are compared only when the row slice matches.
for marker in ('[[ "$rows" == "$K_ROWS"', '[[ "$rows" == "$F_ROWS"', '[[ "$rows" == "$E_ROWS"'):
    if marker not in staged: raise SystemExit('Stage-L row-scoped residue gate missing: '+marker)
print('stagel_contract_structure=OK')
PY

echo 'b300-stagel-preflight OK transform=1 builder=1 policies=default,cg,cs stagek_control=1 stagek_manifest=1 exact_residue=1 ptxas_spill=1 row_scoped_residue=1 search_rows=1 validate_rows=4,8 policy_lock=1 gpu_work=0'
