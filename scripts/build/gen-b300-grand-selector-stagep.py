#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-selector-stagep.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='STAGEO_RACE_PREFIX="${STAGEO_RACE_PREFIX:-${PREFIX}.stageo-cgl2.promote}"\n'; rep(anchor,anchor+'''STAGEP_PREFIX="${STAGEP_PREFIX:-${PREFIX}.stagep-matel2}"
STAGEP_WINNER_ENV="${STAGEP_WINNER_ENV:-${STAGEP_PREFIX}_winner.env}"
STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-${PREFIX}.stagep-matel2.prepared.env}"
STAGEP_RACE_PREFIX="${STAGEP_RACE_PREFIX:-${PREFIX}.stagep-matel2.promote}"
''','Stage-P paths')
anchor='RUN_STAGEO="${RUN_STAGEO:-1}"\n'; rep(anchor,anchor+'RUN_STAGEP="${RUN_STAGEP:-1}"\n','RUN_STAGEP')
anchor='STAGEO_BLOCK_L2_LIST="${STAGEO_BLOCK_L2_LIST:-0 64 128 256}"\n'; rep(anchor,anchor+'STAGEP_MIN_SPEEDUP="${STAGEP_MIN_SPEEDUP:-1.002}"\nSTAGEP_MATE_L2_LIST="${STAGEP_MATE_L2_LIST:-0 64 128 256}"\n','Stage-P knobs')
rep('for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO; do','for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO RUN_STAGEP; do'.replace('RUN_HYBRID_STAGE RUN_STAGEI','RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI'),'boolean validation')
rep('python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" "$STAGEP_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\',\'STAGEP_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speedup validation')
anchor='STAGEO_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGEO_BLOCK_L2_LIST")"\n'; rep(anchor,anchor+'STAGEP_MATE_L2_LIST="$(normalize_l2_sizes "$STAGEP_MATE_L2_LIST")"\ncase " $STAGEP_MATE_L2_LIST " in *\' 0 \'*) ;; *) echo \'STAGEP_MATE_L2_LIST must include 0 baseline\' >&2; exit 2;; esac\n','Stage-P L2 normalization')
rep('  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" "$(dirname "$STAGEP_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','mkdir Stage P')
stage_p=r'''
STAGEP_OK=0; STAGEP_RC=0; STAGEP_COUNT_UPSTREAM=""
if ((STAGEO_OK)); then STAGEP_COUNT_UPSTREAM=stageo
elif ((STAGEN_OK)); then STAGEP_COUNT_UPSTREAM=stagen
fi
if [[ -n "$STAGEP_COUNT_UPSTREAM" ]]; then
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" \
    STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" \
    STAGEO_WINNER_ENV="$STAGEO_WINNER_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" \
    UPSTREAM_KIND="$STAGEP_COUNT_UPSTREAM" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 \
    RUN_STAGED="$RUN_STAGEP" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEP_MIN_SPEEDUP" MATE_L2_LIST="$STAGEP_MATE_L2_LIST" \
    STAGED_PREFIX="$STAGEP_PREFIX" WINNER_ENV="$STAGEP_WINNER_ENV" RACE_PREFIX="$STAGEP_RACE_PREFIX" PREPARE_ENV="$STAGEP_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stagep-mate-cg-l2-staged-fullprime-race.sh" 27
  STAGEP_RC=$?; set -e
  if ((STAGEP_RC==0)); then
    source "$STAGEP_PREPARE_ENV"
    [[ "${B300_STAGEP_PREPARED:-0}" == 1 && -x "$B300_STAGEP_PREPARED_BIN" && -x "$B300_STAGEP_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGEP_PREPARED_MOD" == "$PRIME" && "$B300_STAGEP_PREPARED_NGPU" == 8 && "$B300_STAGEP_PREPARED_COUNT_UPSTREAM" == "$STAGEP_COUNT_UPSTREAM" ]] || { echo 'Stage-P modulus/GPU/upstream drift' >&2; exit 3; }
    if [[ "$STAGEP_COUNT_UPSTREAM" == stageo ]]; then [[ "$B300_STAGEP_PREPARED_CONTROL_BIN" == "$B300_STAGEO_PREPARED_BIN" ]] || { echo 'Stage-P control is not exact Stage-O winner' >&2; exit 3; }; else [[ "$B300_STAGEP_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" ]] || { echo 'Stage-P control is not exact Stage-N winner' >&2; exit 3; }; fi
    [[ "$B300_STAGEP_PREPARED_MATE_LOAD_POLICY" == cg && "$B300_STAGEP_PREPARED_BASE_MATE_L2_BYTES" == 0 && "$B300_STAGEP_PREPARED_MATE_L2_BYTES" != 0 ]] || { echo 'Stage-P mate L2 contract drift' >&2; exit 3; }
    [[ "$B300_STAGEP_PREPARED_PAIR_POLICY" == "$B300_STAGEN_PREPARED_PAIR_POLICY" && "$B300_STAGEP_PREPARED_BLOCK_POLICY" == "$B300_STAGEN_PREPARED_BLOCK_POLICY" ]] || { echo 'Stage-P changed Stage-N Count policy' >&2; exit 3; }
    STAGEP_OK=1
  elif ((STAGEP_RC==4)); then echo 'grand selector: Stage-P mate CG L2 refinement rejected/not-applicable; retaining Stage O/N' >&2
  else exit "$STAGEP_RC"; fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','\n'+stage_p+'\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','Stage-P preparation insertion')
branches=r'''if ((STAGEP_OK && NEXTSELF_OK)); then
  MODE=stagep_matel2_grand
  P_BIN="$B300_STAGEP_PREPARED_BIN"; P_LABEL="$B300_STAGEP_PREPARED_LABEL"; P_THREADS="$B300_STAGEP_PREPARED_THREADS"
  B_BIN="$B300_STAGEP_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEP_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEP_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGEP_OK)); then
  MODE=stagep_matel2_joint
  P_BIN="$B300_STAGEP_PREPARED_BIN"; P_LABEL="$B300_STAGEP_PREPARED_LABEL"; P_THREADS="$B300_STAGEP_PREPARED_THREADS"
  B_BIN="$B300_STAGEP_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEP_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEP_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGEO_OK && NEXTSELF_OK)); then'''
rep('if ((STAGEO_OK && NEXTSELF_OK)); then',branches,'Stage-P branch pair')
suffix='stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand)'
if s.count(suffix)!=2: raise SystemExit(f'drop-mode suffix expected two anchors got {s.count(suffix)}')
s=s.replace(suffix,'stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand|stagep_matel2_grand)',2)
anchor="  printf 'B300_GRAND_STAGEO_OK=%q\\n' \"$STAGEO_OK\"; printf 'B300_GRAND_STAGEO_MIN_SPEEDUP=%q\\n' \"$STAGEO_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEO_PAIR_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_BASE_PAIR_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_BASE_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_BASE_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_BASE_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEO_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEO_MANIFEST=%q\\n' \"${B300_STAGEO_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEO_SEARCH_PAIR_L2=%q\\n' \"$STAGEO_PAIR_L2_LIST\"; printf 'B300_GRAND_STAGEO_SEARCH_BLOCK_L2=%q\\n' \"$STAGEO_BLOCK_L2_LIST\"\n"
rep(anchor,anchor+"  printf 'B300_GRAND_STAGEP_OK=%q\\n' \"$STAGEP_OK\"; printf 'B300_GRAND_STAGEP_MIN_SPEEDUP=%q\\n' \"$STAGEP_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEP_COUNT_UPSTREAM=%q\\n' \"$STAGEP_COUNT_UPSTREAM\"; printf 'B300_GRAND_STAGEP_BASE_MATE_L2_BYTES=%q\\n' \"${B300_STAGEP_PREPARED_BASE_MATE_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEP_MATE_L2_BYTES=%q\\n' \"${B300_STAGEP_PREPARED_MATE_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEP_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEP_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEP_MANIFEST=%q\\n' \"${B300_STAGEP_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEP_SEARCH_MATE_L2=%q\\n' \"$STAGEP_MATE_L2_LIST\"\n",'Stage-P summary')
rep("  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEO_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'","  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEO_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEP_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",'Stage-P integrated marker')
for marker in ('RUN_STAGEP=','STAGEP_MIN_SPEEDUP=','STAGEP_MATE_L2_LIST=','b300x8-nextgen-hybrid8-stagep-mate-cg-l2-staged-fullprime-race.sh','MODE=stagep_matel2_grand','MODE=stagep_matel2_joint','B300_GRAND_STAGEP_OK=','B300_GRAND_STAGEP_MATE_L2_BYTES=','B300_GRAND_STAGEP_INTEGRATED=1'):
    if marker not in s: raise SystemExit('missing generated Stage-P artifact: '+marker)
if s.count('b300x8-race-external-forced-profiled-once.sh')!=1: raise SystemExit('generated selector must retain exactly one complete-prime race')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_stagep_integrated=1 stage_p_after_o_n=1 prepare_only=1 fallback_on=1 single_complete_prime=1')
