#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-selector-stages.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='STAGER_RACE_PREFIX="${STAGER_RACE_PREFIX:-${PREFIX}.stager-ilp2-load.promote}"\n'
rep(anchor,anchor+'''STAGES_PREFIX="${STAGES_PREFIX:-${PREFIX}.stages-ilp2-cg-l2}"
STAGES_WINNER_ENV="${STAGES_WINNER_ENV:-${STAGES_PREFIX}_winner.env}"
STAGES_PREPARE_ENV="${STAGES_PREPARE_ENV:-${PREFIX}.stages-ilp2-cg-l2.prepared.env}"
STAGES_RACE_PREFIX="${STAGES_RACE_PREFIX:-${PREFIX}.stages-ilp2-cg-l2.promote}"
''','Stage-S paths')
anchor='RUN_STAGER="${RUN_STAGER:-1}"\n'; rep(anchor,anchor+'RUN_STAGES="${RUN_STAGES:-1}"\n','RUN_STAGES')
anchor='STAGER_BLOCK_POLICY_LIST="${STAGER_BLOCK_POLICY_LIST:-default cg cs}"\n'
rep(anchor,anchor+'''STAGES_MIN_SPEEDUP="${STAGES_MIN_SPEEDUP:-1.002}"
STAGES_PAIR_L2_LIST="${STAGES_PAIR_L2_LIST:-0 64 128 256}"
STAGES_BLOCK_L2_LIST="${STAGES_BLOCK_L2_LIST:-0 64 128 256}"
[[ "$RUN_STAGES" == 0 || "$RUN_STAGES" == 1 ]] || { echo 'RUN_STAGES must be 0/1' >&2; exit 2; }
python3 - "$STAGES_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('STAGES_MIN_SPEEDUP must be >=1')
PY
''','Stage-S knobs')
anchor='STAGER_BLOCK_POLICY_LIST="$(normalize_load_policies "$STAGER_BLOCK_POLICY_LIST")"\n'
rep(anchor,anchor+'''STAGES_PAIR_L2_LIST="$(normalize_l2_sizes "$STAGES_PAIR_L2_LIST")"
STAGES_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGES_BLOCK_L2_LIST")"
case " $STAGES_PAIR_L2_LIST " in *' 0 '*) ;; *) echo 'STAGES_PAIR_L2_LIST must include 0 baseline' >&2; exit 2;; esac
case " $STAGES_BLOCK_L2_LIST " in *' 0 '*) ;; *) echo 'STAGES_BLOCK_L2_LIST must include 0 baseline' >&2; exit 2;; esac
''','Stage-S L2 normalization')
rep('  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" "$(dirname "$STAGEP_PREPARE_ENV")" "$(dirname "$STAGEQ_PREPARE_ENV")" "$(dirname "$STAGER_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" "$(dirname "$STAGEP_PREPARE_ENV")" "$(dirname "$STAGEQ_PREPARE_ENV")" "$(dirname "$STAGER_PREPARE_ENV")" "$(dirname "$STAGES_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','mkdir Stage S')
stage_s=r'''
STAGES_OK=0; STAGES_RC=0
if ((STAGER_OK)); then
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" STAGEQ_PREPARE_ENV="$STAGEQ_PREPARE_ENV" STAGER_WINNER_ENV="$STAGER_WINNER_ENV" STAGER_PREPARE_ENV="$STAGER_PREPARE_ENV" \
    ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 RUN_STAGED="$RUN_STAGES" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGES_MIN_SPEEDUP" PAIR_L2_LIST="$STAGES_PAIR_L2_LIST" BLOCK_L2_LIST="$STAGES_BLOCK_L2_LIST" STAGED_PREFIX="$STAGES_PREFIX" WINNER_ENV="$STAGES_WINNER_ENV" RACE_PREFIX="$STAGES_RACE_PREFIX" PREPARE_ENV="$STAGES_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stages-ilp2-cg-l2-staged-fullprime-race.sh" 27
  STAGES_RC=$?; set -e
  if ((STAGES_RC==0)); then
    source "$STAGES_PREPARE_ENV"
    [[ "${B300_STAGES_PREPARED:-0}" == 1 && -x "$B300_STAGES_PREPARED_BIN" && -x "$B300_STAGES_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGES_PREPARED_MOD" == "$PRIME" && "$B300_STAGES_PREPARED_NGPU" == 8 ]] || { echo 'Stage-S modulus/GPU drift' >&2; exit 3; }
    [[ "$B300_STAGES_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN" ]] || { echo 'Stage-S control is not exact Stage-R winner' >&2; exit 3; }
    [[ "$B300_STAGES_PREPARED_STAGER_UPSTREAM_KIND" == "$B300_STAGER_PREPARED_UPSTREAM_KIND" ]] || { echo 'Stage-S lost Stage-R upstream provenance' >&2; exit 3; }
    [[ "$B300_STAGES_PREPARED_LOW_PAIR_POLICY" == "$B300_STAGER_PREPARED_PAIR_POLICY" && "$B300_STAGES_PREPARED_LOW_BLOCK_POLICY" == "$B300_STAGER_PREPARED_BLOCK_POLICY" ]] || { echo 'Stage-S changed Stage-R low policy' >&2; exit 3; }
    [[ "$B300_STAGES_PREPARED_HIGH_PAIR_POLICY" == "$B300_STAGER_PREPARED_HIGH_PAIR_POLICY" && "$B300_STAGES_PREPARED_HIGH_BLOCK_POLICY" == "$B300_STAGER_PREPARED_HIGH_BLOCK_POLICY" && "$B300_STAGES_PREPARED_HIGH_PAIR_L2_BYTES" == "$B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES" && "$B300_STAGES_PREPARED_HIGH_BLOCK_L2_BYTES" == "$B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES" ]] || { echo 'Stage-S changed high-state provenance' >&2; exit 3; }
    [[ "$B300_STAGES_PREPARED_PAIR_L2_BYTES" != 0 || "$B300_STAGES_PREPARED_BLOCK_L2_BYTES" != 0 ]] || { echo 'Stage-S selected exact Stage-R zero-hint tuple' >&2; exit 3; }
    STAGES_OK=1
  elif ((STAGES_RC==4)); then echo 'grand selector: Stage-S ILP2 CG L2 hints rejected/not-applicable; retaining Stage R' >&2
  else exit "$STAGES_RC"; fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','\n'+stage_s+'\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','Stage-S preparation insertion')
branches=r'''if ((STAGES_OK && NEXTSELF_OK)); then
  MODE=stages_ilp2_l2_grand
  P_BIN="$B300_STAGES_PREPARED_BIN"; P_LABEL="$B300_STAGES_PREPARED_LABEL"; P_THREADS="$B300_STAGES_PREPARED_THREADS"
  B_BIN="$B300_STAGES_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGES_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGES_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGES_OK)); then
  MODE=stages_ilp2_l2_joint
  P_BIN="$B300_STAGES_PREPARED_BIN"; P_LABEL="$B300_STAGES_PREPARED_LABEL"; P_THREADS="$B300_STAGES_PREPARED_THREADS"
  B_BIN="$B300_STAGES_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGES_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGES_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGER_OK && NEXTSELF_OK)); then'''
rep('if ((STAGER_OK && NEXTSELF_OK)); then',branches,'Stage-S branch pair')
suffix='stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand|stagep_matel2_grand|stageq_countl2_grand|stager_ilp2_grand)'
if s.count(suffix)!=2: raise SystemExit(f'Stage-S drop-mode suffix expected two anchors got {s.count(suffix)}')
s=s.replace(suffix,'stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand|stagep_matel2_grand|stageq_countl2_grand|stager_ilp2_grand|stages_ilp2_l2_grand)',2)
anchor="  printf 'B300_GRAND_STAGER_OK=%q\\n' \"$STAGER_OK\"; printf 'B300_GRAND_STAGER_MIN_SPEEDUP=%q\\n' \"$STAGER_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGER_UPSTREAM_KIND=%q\\n' \"$STAGER_UPSTREAM_KIND\"; printf 'B300_GRAND_STAGER_UPSTREAM_PAIR_POLICY=%q\\n' \"${B300_STAGER_PREPARED_UPSTREAM_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGER_UPSTREAM_BLOCK_POLICY=%q\\n' \"${B300_STAGER_PREPARED_UPSTREAM_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGER_HIGH_PAIR_POLICY=%q\\n' \"${B300_STAGER_PREPARED_HIGH_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGER_HIGH_BLOCK_POLICY=%q\\n' \"${B300_STAGER_PREPARED_HIGH_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGER_HIGH_PAIR_L2_BYTES=%q\\n' \"${B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGER_HIGH_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGER_PAIR_POLICY=%q\\n' \"${B300_STAGER_PREPARED_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGER_BLOCK_POLICY=%q\\n' \"${B300_STAGER_PREPARED_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGER_STAGED_SPEEDUP=%q\\n' \"${B300_STAGER_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGER_MANIFEST=%q\\n' \"${B300_STAGER_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGER_SEARCH_PAIR_POLICIES=%q\\n' \"$STAGER_PAIR_POLICY_LIST\"; printf 'B300_GRAND_STAGER_SEARCH_BLOCK_POLICIES=%q\\n' \"$STAGER_BLOCK_POLICY_LIST\"\n"
rep(anchor,anchor+"  printf 'B300_GRAND_STAGES_OK=%q\\n' \"$STAGES_OK\"; printf 'B300_GRAND_STAGES_MIN_SPEEDUP=%q\\n' \"$STAGES_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGES_STAGER_UPSTREAM_KIND=%q\\n' \"${B300_STAGES_PREPARED_STAGER_UPSTREAM_KIND:-}\"; printf 'B300_GRAND_STAGES_LOW_PAIR_POLICY=%q\\n' \"${B300_STAGES_PREPARED_LOW_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGES_LOW_BLOCK_POLICY=%q\\n' \"${B300_STAGES_PREPARED_LOW_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGES_HIGH_PAIR_POLICY=%q\\n' \"${B300_STAGES_PREPARED_HIGH_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGES_HIGH_BLOCK_POLICY=%q\\n' \"${B300_STAGES_PREPARED_HIGH_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGES_HIGH_PAIR_L2_BYTES=%q\\n' \"${B300_STAGES_PREPARED_HIGH_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGES_HIGH_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGES_PREPARED_HIGH_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGES_PAIR_L2_BYTES=%q\\n' \"${B300_STAGES_PREPARED_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGES_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGES_PREPARED_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGES_STAGED_SPEEDUP=%q\\n' \"${B300_STAGES_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGES_MANIFEST=%q\\n' \"${B300_STAGES_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGES_SEARCH_PAIR_L2=%q\\n' \"$STAGES_PAIR_L2_LIST\"; printf 'B300_GRAND_STAGES_SEARCH_BLOCK_L2=%q\\n' \"$STAGES_BLOCK_L2_LIST\"\n",'Stage-S summary')
rep("printf 'B300_GRAND_STAGER_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n';","printf 'B300_GRAND_STAGER_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGES_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n';",'Stage-S integrated marker')
for marker in ('RUN_STAGES=','STAGES_MIN_SPEEDUP=','STAGES_PAIR_L2_LIST=','STAGES_BLOCK_L2_LIST=','b300x8-nextgen-hybrid8-stages-ilp2-cg-l2-staged-fullprime-race.sh','MODE=stages_ilp2_l2_grand','MODE=stages_ilp2_l2_joint','B300_GRAND_STAGES_OK=','B300_GRAND_STAGES_PAIR_L2_BYTES=','B300_GRAND_STAGES_INTEGRATED=1'):
    if marker not in s: raise SystemExit('missing generated Stage-S artifact: '+marker)
if s.count('b300x8-race-external-forced-profiled-once.sh')!=1: raise SystemExit('Stage-S selector must retain exactly one complete-prime race')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_stages_integrated=1 upstream=R prepare_only=1 fallback=R single_complete_prime=1')
