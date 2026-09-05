#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-load-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-pair-block-load-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-pair-block-load-policy-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-pair-block-load-policy-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh"
SCREEN="$ONEESAN_ROOT/scripts/run/b300x8-grand-stagen-screen.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED" "$PROMOTE" "$SCREEN"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
bash "$PROOF" >/dev/null
need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-N marker missing in $f: $s" >&2; exit 3; }; }
for s in 'MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-default}"' 'PAIR_LOAD_POLICY="${PAIR_LOAD_POLICY:-default}"' 'BLOCK_LOAD_POLICY="${BLOCK_LOAD_POLICY:-default}"' 'b300-forced-nextgen-hybrid8-self-mate-geometry.sh' 'b300-forced-nextgen-hybrid8-mate-load-policy.sh' 'gen-b300-mainrec-pair-block-load-policy.py' 'stage_n_scope=pair_block_count_reads_only' 'geometry_eviction_guard_preserved=1'; do need "$BUILDER" "$s"; done
for s in 'UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"' 'PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"' 'BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"' 'BASE_COUNT_POLICY=default' '[[ "$CG" == 1 ]] && BASE_COUNT_POLICY=cg' 'B300_STAGEN_UPSTREAM_KIND' 'B300_STAGEN_MATE_LOAD_POLICY' 'B300_STAGEN_PAIR_POLICY' 'B300_STAGEN_BLOCK_POLICY' 'FATAL Stage-N residue mismatch' 'clean=len(rv)>=2 and ss==0 and sl==0' 'b300_stagen_exact_match=1'; do need "$SWEEP" "$s"; done
for s in 'SEARCH_ROWS="${SEARCH_ROWS:-1}"' 'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' 'RESOLVED_UPSTREAM=stagel' 'FATAL Stage-N/upstream residue mismatch' 'FATAL Stage-N/Stage-L residue mismatch' 'FATAL Stage-N/Stage-F residue mismatch' 'SELECTED_PAIR="$B300_STAGEN_PAIR_POLICY"' 'SELECTED_BLOCK="$B300_STAGEN_BLOCK_POLICY"' 'B300_STAGEN_STAGED_VALIDATED=' 'B300_STAGEN_FINAL_ENABLED=' 'B300_STAGEN_FINAL_STAGE_RESIDUE='; do need "$STAGED" "$s"; done
for s in 'B300_STAGEL_PREPARED_MANIFEST' 'sha256sum -c "$B300_STAGEL_PREPARED_MANIFEST"' 'B300_STAGEM_PREPARED_MANIFEST' 'Stage-M no longer controls against exact Stage L' 'Stage-N control is not exact prepared upstream' 'Stage-N promoted symmetric inherited baseline' 'B300_STAGEN_PREPARED_UPSTREAM_MANIFEST' 'B300_STAGEN_PREPARED_PAIR_POLICY' 'B300_STAGEN_PREPARED_BLOCK_POLICY' 'B300_STAGEN_PREPARED_MANIFEST' 'PREPARE_ONLY' 'b300x8-race-external-forced-profiled-once.sh'; do need "$PROMOTE" "$s"; done
for s in 'FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"' 'STAGE_F_ENV="${STAGE_F_ENV:-${FIRSTPASS_PREFIX}.hybrid8-nextself_winner.env}"' 'STAGEL_WINNER_ENV="${STAGEL_WINNER_ENV:-${FIRSTPASS_PREFIX}.stagel-guard_winner.env}"' 'STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-${FIRSTPASS_PREFIX}.stagem-mateload_winner.env}"' 'Stage N screen not applicable: Stage L was not accepted' 'UPSTREAM_KIND=stagem' 'RUN_STAGED=1 PREPARE_ONLY=1' 'GRAND STAGE N SCREEN PASSED'; do need "$SCREEN" "$s"; done
python3 - "$SWEEP" "$STAGED" "$PROMOTE" "$SCREEN" <<'PY'
from pathlib import Path
import sys
sweep=Path(sys.argv[1]).read_text(); staged=Path(sys.argv[2]).read_text(); promote=Path(sys.argv[3]).read_text(); screen=Path(sys.argv[4]).read_text()
if "printf 'control\\t%s\\t%s\\t%s\\t-\\n' \"$BASE_COUNT_POLICY\" \"$BASE_COUNT_POLICY\" \"$CONTROL_BIN\"" not in sweep:
    raise SystemExit('Stage-N control must be the exact upstream binary')
if '[[ "$pair" == "$BASE_COUNT_POLICY" && "$block" == "$BASE_COUNT_POLICY" ]] && continue' not in sweep:
    raise SystemExit('Stage-N must not rebuild the inherited symmetric baseline')
if 'run_stage "$rows" "$SELECTED_PAIR" "$SELECTED_BLOCK"' not in staged:
    raise SystemExit('Stage-N validation must lock pair/block winner')
if 'UPSTREAM_KIND="$RESOLVED_UPSTREAM"' not in staged:
    raise SystemExit('Stage-N staged sweep must pin upstream identity')
if promote.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-N promotion must expose exactly one optional full-prime race')
if '[[ "$B300_STAGEN_CONTROL_BIN" == "$UP_BIN" ]]' not in promote:
    raise SystemExit('Stage-N promotion must bind control to exact prepared upstream')
if 'PREPARE_ONLY=1' not in screen or 'b300x8-race-external-forced-profiled-once.sh' in screen:
    raise SystemExit('grand Stage-N screen must never start a complete-prime race itself')
print('stagen_contract_structure=OK')
PY
echo 'b300_stagen_preflight=OK stage_n=pair_block_count_loads upstream=stagel_or_stagem pair_policies=default,cg,cs block_policies=default,cg,cs exact_upstream_control=1 inherited_baseline_skipped=1 residue_gate=1 ptxas_spill=1 row_scoped_validation=1 upstream_locked=1 promotion_manifest=1 stage_l_manifest=1 stage_m_manifest_optional=1 grand_screen_prepare_only=1 gpu_work=0'
