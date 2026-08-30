#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageo-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-pair-block-cg-l2-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageo-cg-l2-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageo-cg-l2-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED" "$PROMOTE"; do [[ -f "$f" ]] || { echo "missing Stage-O dependency=$f" >&2; exit 2; }; bash -n "$f"; done
bash "$PROOF" >/dev/null
need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-O marker missing in $f: $s" >&2; exit 3; }; }
for s in \
  'BASE_CG_L2_BYTES="${BASE_CG_L2_BYTES:-0}"' \
  'PAIR_CG_L2_BYTES="${PAIR_CG_L2_BYTES:-0}"' \
  'BLOCK_CG_L2_BYTES="${BLOCK_CG_L2_BYTES:-0}"' \
  'gen-b300-mainrec-pair-block-cg-l2-policy.py' \
  'stage_o_scope=cg_l2_fetch_size_only' \
  'pair_block_policy_preserved=1'; do need "$BUILDER" "$s"; done
for s in \
  'PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"' \
  'BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"' \
  'Stage-N manifest mismatch before Stage O' \
  'PAIR_L2_LIST omits inherited baseline' \
  'BLOCK_L2_LIST omits inherited baseline' \
  'B300_STAGEO_CONTROL_BIN' \
  'B300_STAGEO_PAIR_L2_BYTES' \
  'B300_STAGEO_BLOCK_L2_BYTES' \
  'FATAL Stage-O residue mismatch' \
  'b300_stageo_exact_match=1'; do need "$SWEEP" "$s"; done
for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'SELECTED_PAIR_L2="$B300_STAGEO_PAIR_L2_BYTES"' \
  'SELECTED_BLOCK_L2="$B300_STAGEO_BLOCK_L2_BYTES"' \
  'FATAL Stage-O/Stage-N residue mismatch' \
  'B300_STAGEO_STAGED_VALIDATED=' \
  'B300_STAGEO_FINAL_ENABLED=' \
  'B300_STAGEO_FINAL_STAGE_RESIDUE='; do need "$STAGED" "$s"; done
for s in \
  'sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST"' \
  'Stage-O control is not exact prepared Stage N' \
  'Stage-O promoted inherited L2 baseline unchanged' \
  'B300_STAGEO_PREPARED_STAGEN_MANIFEST' \
  'B300_STAGEO_PREPARED_PAIR_L2_BYTES' \
  'B300_STAGEO_PREPARED_BLOCK_L2_BYTES' \
  'B300_STAGEO_PREPARED_MANIFEST' \
  'PREPARE_ONLY' \
  'b300x8-race-external-forced-profiled-once.sh'; do need "$PROMOTE" "$s"; done
python3 - "$SWEEP" "$STAGED" "$PROMOTE" <<'PY'
from pathlib import Path
import sys
sweep=Path(sys.argv[1]).read_text(); staged=Path(sys.argv[2]).read_text(); promote=Path(sys.argv[3]).read_text()
if "printf 'control\\t%s\\t%s\\t%s\\t-\\n' \"$BASE_PAIR_L2\" \"$BASE_BLOCK_L2\" \"$CONTROL_BIN\"" not in sweep:
    raise SystemExit('Stage-O sweep control must be exact Stage-N binary')
if '[[ "$pl2" == "$BASE_PAIR_L2" && "$bl2" == "$BASE_BLOCK_L2" ]] && continue' not in sweep:
    raise SystemExit('Stage-O sweep must not rebuild inherited L2 baseline')
if 'run_stage "$rows" "$pair_validation" "$block_validation"' not in staged:
    raise SystemExit('Stage-O staged validation must lock the selected L2 pair')
if 'for rows in $VALIDATE_ROWS "$N_ROWS"' not in staged:
    raise SystemExit('Stage-O staged validation must include Stage-N residue row')
if promote.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-O promotion must expose exactly one optional full-prime race')
if '[[ "$B300_STAGEO_CONTROL_BIN" == "$N_BIN" ]]' not in promote:
    raise SystemExit('Stage-O promotion must bind control to exact Stage N')
print('stageo_contract_structure=OK')
PY
echo 'b300_stageo_preflight=OK stage_o=cg_l2_fetch_size pair_block_axes_independent=1 exact_stagen_control=1 inherited_baseline_skipped=1 residue_gate=1 ptxas_spill=1 row_scoped_validation=1 selected_l2_locked=1 promotion_manifest=1 stagen_manifest=1 single_optional_fullprime=1 gpu_work=0'
