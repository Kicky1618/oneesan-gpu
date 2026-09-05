#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-selector-staget.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
if 'B300_GRAND_STAGES_INTEGRATED=1' not in s or 'RUN_STAGES=' not in s:
    raise SystemExit('Stage T grand overlay requires Stage-S-integrated selector')
if 'B300_GRAND_STAGET_INTEGRATED=1' in s or 'RUN_STAGET=' in s:
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s); print(f'generated {out} from {src}: already_stage_t=1'); raise SystemExit(0)

def after_line(prefix:str,extra:str,label:str)->None:
    global s
    lines=s.splitlines(True); hits=[i for i,x in enumerate(lines) if x.startswith(prefix)]
    if len(hits)!=1: raise SystemExit(f'{label}: expected one line prefix {prefix!r}, got {len(hits)}')
    i=hits[0]; lines[i+1:i+1]=[extra if extra.endswith('\n') else extra+'\n']; s=''.join(lines)

def rep(old:str,new:str,label:str,count:int=1)->None:
    global s
    n=s.count(old)
    if n!=count: raise SystemExit(f'{label}: expected {count} anchors, got {n}')
    s=s.replace(old,new,count)

after_line('STAGES_RACE_PREFIX=', '''STAGET_PREFIX="${STAGET_PREFIX:-${PREFIX}.staget-ilp2-mate}"
STAGET_WINNER_ENV="${STAGET_WINNER_ENV:-${STAGET_PREFIX}_winner.env}"
STAGET_PREPARE_ENV="${STAGET_PREPARE_ENV:-${PREFIX}.staget-ilp2-mate.prepared.env}"
STAGET_RACE_PREFIX="${STAGET_RACE_PREFIX:-${PREFIX}.staget-ilp2-mate.promote}"''','Stage-T paths')
after_line('RUN_STAGES=', 'RUN_STAGET="${RUN_STAGET:-1}"','RUN_STAGET')
after_line('STAGES_BLOCK_L2_LIST="${STAGES_BLOCK_L2_LIST:-0 64 128 256}"', '''STAGET_MIN_SPEEDUP="${STAGET_MIN_SPEEDUP:-1.002}"
STAGET_POLICY_LIST="${STAGET_POLICY_LIST:-default cg cs}"
[[ "$RUN_STAGET" == 0 || "$RUN_STAGET" == 1 ]] || { echo 'RUN_STAGET must be 0/1' >&2; exit 2; }
python3 - "$STAGET_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('STAGET_MIN_SPEEDUP must be >=1')
PY''','Stage-T knobs')
after_line('STAGES_BLOCK_L2_LIST="$(normalize_l2_sizes ', '''STAGET_POLICY_LIST="$(normalize_load_policies "$STAGET_POLICY_LIST")"
case " $STAGET_POLICY_LIST " in *' default '*) ;; *) echo 'STAGET_POLICY_LIST must include default baseline' >&2; exit 2;; esac''','Stage-T normalization')

stage_t=r'''
STAGET_OK=0; STAGET_RC=0; STAGET_UPSTREAM_KIND=""
if ((STAGER_OK)); then
  if ((STAGES_OK)); then STAGET_UPSTREAM_KIND=stages; else STAGET_UPSTREAM_KIND=stager; fi
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" STAGEQ_PREPARE_ENV="$STAGEQ_PREPARE_ENV" STAGER_WINNER_ENV="$STAGER_WINNER_ENV" STAGER_PREPARE_ENV="$STAGER_PREPARE_ENV" STAGES_WINNER_ENV="$STAGES_WINNER_ENV" STAGES_PREPARE_ENV="$STAGES_PREPARE_ENV" \
    ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 UPSTREAM_KIND="$STAGET_UPSTREAM_KIND" RUN_STAGED="$RUN_STAGET" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGET_MIN_SPEEDUP" POLICY_LIST="$STAGET_POLICY_LIST" STAGED_PREFIX="$STAGET_PREFIX" WINNER_ENV="$STAGET_WINNER_ENV" RACE_PREFIX="$STAGET_RACE_PREFIX" PREPARE_ENV="$STAGET_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-fullprime-race.sh" 27
  STAGET_RC=$?; set -e
  if ((STAGET_RC==0)); then
    source "$STAGET_PREPARE_ENV"
    [[ "${B300_STAGET_PREPARED:-0}" == 1 && -x "$B300_STAGET_PREPARED_BIN" && -x "$B300_STAGET_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGET_PREPARED_MOD" == "$PRIME" && "$B300_STAGET_PREPARED_NGPU" == 8 && "$B300_STAGET_PREPARED_UPSTREAM_KIND" == "$STAGET_UPSTREAM_KIND" ]] || { echo 'Stage-T modulus/GPU/upstream drift' >&2; exit 3; }
    case "$B300_STAGET_PREPARED_POLICY" in cg|cs) ;; *) echo 'Stage-T prepared default/non-policy candidate' >&2; exit 3;; esac
    if [[ "$STAGET_UPSTREAM_KIND" == stages ]]; then
      [[ "$B300_STAGET_PREPARED_CONTROL_BIN" == "$B300_STAGES_PREPARED_BIN" ]] || { echo 'Stage-T control is not exact Stage-S winner' >&2; exit 3; }
      [[ "$B300_STAGET_PREPARED_LOW_PAIR_POLICY" == "$B300_STAGES_PREPARED_LOW_PAIR_POLICY" && "$B300_STAGET_PREPARED_LOW_BLOCK_POLICY" == "$B300_STAGES_PREPARED_LOW_BLOCK_POLICY" && "$B300_STAGET_PREPARED_LOW_PAIR_L2_BYTES" == "$B300_STAGES_PREPARED_PAIR_L2_BYTES" && "$B300_STAGET_PREPARED_LOW_BLOCK_L2_BYTES" == "$B300_STAGES_PREPARED_BLOCK_L2_BYTES" ]] || { echo 'Stage-T changed Stage-S low Count tuple' >&2; exit 3; }
    else
      [[ "$B300_STAGET_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN" ]] || { echo 'Stage-T control is not exact Stage-R winner' >&2; exit 3; }
      [[ "$B300_STAGET_PREPARED_LOW_PAIR_POLICY" == "$B300_STAGER_PREPARED_PAIR_POLICY" && "$B300_STAGET_PREPARED_LOW_BLOCK_POLICY" == "$B300_STAGER_PREPARED_BLOCK_POLICY" && "$B300_STAGET_PREPARED_LOW_PAIR_L2_BYTES" == 0 && "$B300_STAGET_PREPARED_LOW_BLOCK_L2_BYTES" == 0 ]] || { echo 'Stage-T changed Stage-R low Count tuple' >&2; exit 3; }
    fi
    [[ "$B300_STAGET_PREPARED_HIGH_PAIR_POLICY" == "$B300_STAGER_PREPARED_HIGH_PAIR_POLICY" && "$B300_STAGET_PREPARED_HIGH_BLOCK_POLICY" == "$B300_STAGER_PREPARED_HIGH_BLOCK_POLICY" && "$B300_STAGET_PREPARED_HIGH_PAIR_L2_BYTES" == "$B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES" && "$B300_STAGET_PREPARED_HIGH_BLOCK_L2_BYTES" == "$B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES" ]] || { echo 'Stage-T changed high Count tuple' >&2; exit 3; }
    STAGET_OK=1
  elif ((STAGET_RC==4)); then echo 'grand selector: Stage-T ILP2 mate policy rejected/not-applicable; retaining Stage S/R' >&2
  else exit "$STAGET_RC"; fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','\n'+stage_t+'\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','Stage-T preparation insertion')

old='if ((STAGES_OK && NEXTSELF_OK)); then'
new=r'''if ((STAGET_OK && NEXTSELF_OK)); then
  MODE=staget_ilp2_mate_grand
  P_BIN="$B300_STAGET_PREPARED_BIN"; P_LABEL="$B300_STAGET_PREPARED_LABEL"; P_THREADS="$B300_STAGET_PREPARED_THREADS"
  B_BIN="$B300_STAGET_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGET_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGET_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGET_OK)); then
  MODE=staget_ilp2_mate_joint
  P_BIN="$B300_STAGET_PREPARED_BIN"; P_LABEL="$B300_STAGET_PREPARED_LABEL"; P_THREADS="$B300_STAGET_PREPARED_THREADS"
  B_BIN="$B300_STAGET_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGET_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGET_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGES_OK && NEXTSELF_OK)); then'''
rep(old,new,'Stage-T priority branches')
# Any grand-mode filter that already knows Stage S must also know Stage T.
suffix='|stages_ilp2_l2_grand)'
n=s.count(suffix)
if n<1: raise SystemExit('Stage-T grand-mode suffix anchor missing')
s=s.replace(suffix,'|stages_ilp2_l2_grand|staget_ilp2_mate_grand)')
# Append summary after the Stage-S summary line.
lines=s.splitlines(True); hits=[i for i,x in enumerate(lines) if 'B300_GRAND_STAGES_OK=%q' in x]
if len(hits)!=1: raise SystemExit(f'Stage-T summary anchor expected one got {len(hits)}')
summary='''  printf 'B300_GRAND_STAGET_OK=%q\\n' "$STAGET_OK"; printf 'B300_GRAND_STAGET_MIN_SPEEDUP=%q\\n' "$STAGET_MIN_SPEEDUP"; printf 'B300_GRAND_STAGET_UPSTREAM_KIND=%q\\n' "${B300_STAGET_PREPARED_UPSTREAM_KIND:-}"; printf 'B300_GRAND_STAGET_STAGER_UPSTREAM_KIND=%q\\n' "${B300_STAGET_PREPARED_STAGER_UPSTREAM_KIND:-}"; printf 'B300_GRAND_STAGET_LOW_PAIR_POLICY=%q\\n' "${B300_STAGET_PREPARED_LOW_PAIR_POLICY:-default}"; printf 'B300_GRAND_STAGET_LOW_BLOCK_POLICY=%q\\n' "${B300_STAGET_PREPARED_LOW_BLOCK_POLICY:-default}"; printf 'B300_GRAND_STAGET_LOW_PAIR_L2_BYTES=%q\\n' "${B300_STAGET_PREPARED_LOW_PAIR_L2_BYTES:-0}"; printf 'B300_GRAND_STAGET_LOW_BLOCK_L2_BYTES=%q\\n' "${B300_STAGET_PREPARED_LOW_BLOCK_L2_BYTES:-0}"; printf 'B300_GRAND_STAGET_HIGH_PAIR_POLICY=%q\\n' "${B300_STAGET_PREPARED_HIGH_PAIR_POLICY:-default}"; printf 'B300_GRAND_STAGET_HIGH_BLOCK_POLICY=%q\\n' "${B300_STAGET_PREPARED_HIGH_BLOCK_POLICY:-default}"; printf 'B300_GRAND_STAGET_HIGH_PAIR_L2_BYTES=%q\\n' "${B300_STAGET_PREPARED_HIGH_PAIR_L2_BYTES:-0}"; printf 'B300_GRAND_STAGET_HIGH_BLOCK_L2_BYTES=%q\\n' "${B300_STAGET_PREPARED_HIGH_BLOCK_L2_BYTES:-0}"; printf 'B300_GRAND_STAGET_HIGH_MATE_POLICY=%q\\n' "${B300_STAGET_PREPARED_HIGH_MATE_POLICY:-default}"; printf 'B300_GRAND_STAGET_HIGH_MATE_L2_BYTES=%q\\n' "${B300_STAGET_PREPARED_HIGH_MATE_L2_BYTES:-0}"; printf 'B300_GRAND_STAGET_POLICY=%q\\n' "${B300_STAGET_PREPARED_POLICY:-default}"; printf 'B300_GRAND_STAGET_STAGED_SPEEDUP=%q\\n' "${B300_STAGET_PREPARED_STAGED_SPEEDUP:-1.0}"; printf 'B300_GRAND_STAGET_MANIFEST=%q\\n' "${B300_STAGET_PREPARED_MANIFEST:-}"; printf 'B300_GRAND_STAGET_SEARCH_POLICIES=%q\\n' "$STAGET_POLICY_LIST"\n'''
lines[hits[0]+1:hits[0]+1]=[summary]; s=''.join(lines)
rep("printf 'B300_GRAND_STAGES_INTEGRATED=1\\n';","printf 'B300_GRAND_STAGES_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGET_INTEGRATED=1\\n';",'Stage-T integrated marker')
for marker in ('RUN_STAGET=','STAGET_MIN_SPEEDUP=','STAGET_POLICY_LIST=','b300x8-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-fullprime-race.sh','MODE=staget_ilp2_mate_grand','MODE=staget_ilp2_mate_joint','B300_GRAND_STAGET_OK=','B300_GRAND_STAGET_POLICY=','B300_GRAND_STAGET_INTEGRATED=1'):
    if marker not in s: raise SystemExit('missing generated Stage-T artifact: '+marker)
if s.count('b300x8-race-external-forced-profiled-once.sh')!=1: raise SystemExit('Stage-T selector must retain exactly one complete-prime race')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_staget_integrated=1 upstream_priority=T>S>R prepare_only=1 fallback=S_or_R single_complete_prime=1')
