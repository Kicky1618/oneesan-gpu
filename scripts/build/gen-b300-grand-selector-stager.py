#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-selector-stager.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='STAGEQ_RACE_PREFIX="${STAGEQ_RACE_PREFIX:-${PREFIX}.stageq-countl2.promote}"\n'
rep(anchor,anchor+'''STAGER_PREFIX="${STAGER_PREFIX:-${PREFIX}.stager-ilp2-load}"
STAGER_WINNER_ENV="${STAGER_WINNER_ENV:-${STAGER_PREFIX}_winner.env}"
STAGER_PREPARE_ENV="${STAGER_PREPARE_ENV:-${PREFIX}.stager-ilp2-load.prepared.env}"
STAGER_RACE_PREFIX="${STAGER_RACE_PREFIX:-${PREFIX}.stager-ilp2-load.promote}"
''','Stage-R paths')
anchor='RUN_STAGEQ="${RUN_STAGEQ:-1}"\n'; rep(anchor,anchor+'RUN_STAGER="${RUN_STAGER:-1}"\n','RUN_STAGER')
anchor='STAGEQ_BLOCK_L2_LIST="${STAGEQ_BLOCK_L2_LIST:-0 64 128 256}"\n'
rep(anchor,anchor+'''STAGER_MIN_SPEEDUP="${STAGER_MIN_SPEEDUP:-1.002}"
STAGER_PAIR_POLICY_LIST="${STAGER_PAIR_POLICY_LIST:-default cg cs}"
STAGER_BLOCK_POLICY_LIST="${STAGER_BLOCK_POLICY_LIST:-default cg cs}"
''','Stage-R knobs')
rep('for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO RUN_STAGEP RUN_STAGEQ; do','for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO RUN_STAGEP RUN_STAGEQ RUN_STAGER; do','boolean validation')
rep('python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" "$STAGEP_MIN_SPEEDUP" "$STAGEQ_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\',\'STAGEP_MIN_SPEEDUP\',\'STAGEQ_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" "$STAGEP_MIN_SPEEDUP" "$STAGEQ_MIN_SPEEDUP" "$STAGER_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\',\'STAGEP_MIN_SPEEDUP\',\'STAGEQ_MIN_SPEEDUP\',\'STAGER_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speed validation')
anchor='STAGEQ_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGEQ_BLOCK_L2_LIST")"\n'
rep(anchor,anchor+'''normalize_load_policies(){
  local raw="$1" out=() p old seen
  for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-R load policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done
  ((${#out[@]})) || { echo 'Stage-R policy list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
STAGER_PAIR_POLICY_LIST="$(normalize_load_policies "$STAGER_PAIR_POLICY_LIST")"
STAGER_BLOCK_POLICY_LIST="$(normalize_load_policies "$STAGER_BLOCK_POLICY_LIST")"
''','Stage-R policy normalization')
rep('  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" "$(dirname "$STAGEP_PREPARE_ENV")" "$(dirname "$STAGEQ_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" "$(dirname "$STAGEP_PREPARE_ENV")" "$(dirname "$STAGEQ_PREPARE_ENV")" "$(dirname "$STAGER_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','mkdir Stage R')
stage_r=r'''
STAGER_OK=0; STAGER_RC=0; STAGER_UPSTREAM_KIND=""
if ((STAGEQ_OK)); then STAGER_UPSTREAM_KIND=stageq
elif ((STAGEP_OK)); then STAGER_UPSTREAM_KIND=stagep
elif ((STAGEO_OK)); then STAGER_UPSTREAM_KIND=stageo
elif ((STAGEN_OK)); then STAGER_UPSTREAM_KIND=stagen
fi
if [[ -n "$STAGER_UPSTREAM_KIND" ]]; then
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" STAGEQ_WINNER_ENV="$STAGEQ_WINNER_ENV" STAGEQ_PREPARE_ENV="$STAGEQ_PREPARE_ENV" \
    UPSTREAM_KIND="$STAGER_UPSTREAM_KIND" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 RUN_STAGED="$RUN_STAGER" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGER_MIN_SPEEDUP" PAIR_POLICY_LIST="$STAGER_PAIR_POLICY_LIST" BLOCK_POLICY_LIST="$STAGER_BLOCK_POLICY_LIST" STAGED_PREFIX="$STAGER_PREFIX" WINNER_ENV="$STAGER_WINNER_ENV" RACE_PREFIX="$STAGER_RACE_PREFIX" PREPARE_ENV="$STAGER_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stager-ilp2-load-policy-staged-fullprime-race.sh" 27
  STAGER_RC=$?; set -e
  if ((STAGER_RC==0)); then
    source "$STAGER_PREPARE_ENV"
    [[ "${B300_STAGER_PREPARED:-0}" == 1 && -x "$B300_STAGER_PREPARED_BIN" && -x "$B300_STAGER_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGER_PREPARED_MOD" == "$PRIME" && "$B300_STAGER_PREPARED_NGPU" == 8 && "$B300_STAGER_PREPARED_UPSTREAM_KIND" == "$STAGER_UPSTREAM_KIND" ]] || { echo 'Stage-R modulus/GPU/upstream drift' >&2; exit 3; }
    case "$STAGER_UPSTREAM_KIND" in
      stageq) [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEQ_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-Q winner' >&2; exit 3; } ;;
      stagep) [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEP_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-P winner' >&2; exit 3; } ;;
      stageo) [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEO_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-O winner' >&2; exit 3; } ;;
      stagen) [[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" ]] || { echo 'Stage-R control is not exact Stage-N winner' >&2; exit 3; } ;;
      *) exit 3;;
    esac
    [[ "$B300_STAGER_PREPARED_PAIR_POLICY" != "$B300_STAGER_PREPARED_UPSTREAM_PAIR_POLICY" || "$B300_STAGER_PREPARED_BLOCK_POLICY" != "$B300_STAGER_PREPARED_UPSTREAM_BLOCK_POLICY" ]] || { echo 'Stage-R selected exact upstream ILP2 tuple' >&2; exit 3; }
    STAGER_OK=1
  elif ((STAGER_RC==4)); then echo 'grand selector: Stage-R ILP2 load policy rejected; retaining Q/P/O/N' >&2
  else exit "$STAGER_RC"; fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','\n'+stage_r+'\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','Stage-R preparation insertion')
branches=r'''if ((STAGER_OK && NEXTSELF_OK)); then
  MODE=stager_ilp2_grand
  P_BIN="$B300_STAGER_PREPARED_BIN"; P_LABEL="$B300_STAGER_PREPARED_LABEL"; P_THREADS="$B300_STAGER_PREPARED_THREADS"
  B_BIN="$B300_STAGER_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGER_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGER_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGER_OK)); then
  MODE=stager_ilp2_joint
  P_BIN="$B300_STAGER_PREPARED_BIN"; P_LABEL="$B300_STAGER_PREPARED_LABEL"; P_THREADS="$B300_STAGER_PREPARED_THREADS"
  B_BIN="$B300_STAGER_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGER_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGER_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGEQ_OK && NEXTSELF_OK)); then'''
rep('if ((STAGEQ_OK && NEXTSELF_OK)); then',branches,'Stage-R branch pair')
suffix='stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand|stagep_matel2_grand|stageq_countl2_grand)'
if s.count(suffix)!=2: raise SystemExit(f'drop-mode suffix expected two anchors got {s.count(suffix)}')
s=s.replace(suffix,'stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand|stagep_matel2_grand|stageq_countl2_grand|stager_ilp2_grand)',2)
anchor="  printf 'B300_GRAND_STAGEQ_OK=%q\\n' \"$STAGEQ_OK\"; printf 'B300_GRAND_STAGEQ_MIN_SPEEDUP=%q\\n' \"$STAGEQ_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEQ_UPSTREAM_KIND=%q\\n' \"$STAGEQ_UPSTREAM_KIND\"; printf 'B300_GRAND_STAGEQ_UPSTREAM_PAIR_L2_BYTES=%q\\n' \"${B300_STAGEQ_PREPARED_UPSTREAM_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEQ_UPSTREAM_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGEQ_PREPARED_UPSTREAM_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEQ_PAIR_L2_BYTES=%q\\n' \"${B300_STAGEQ_PREPARED_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEQ_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGEQ_PREPARED_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEQ_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEQ_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEQ_MANIFEST=%q\\n' \"${B300_STAGEQ_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEQ_SEARCH_PAIR_L2=%q\\n' \"$STAGEQ_PAIR_L2_LIST\"; printf 'B300_GRAND_STAGEQ_SEARCH_BLOCK_L2=%q\\n' \"$STAGEQ_BLOCK_L2_LIST\"\n"
rep(anchor,anchor+"  printf 'B300_GRAND_STAGER_OK=%q\\n' \"$STAGER_OK\"; printf 'B300_GRAND_STAGER_MIN_SPEEDUP=%q\\n' \"$STAGER_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGER_UPSTREAM_KIND=%q\\n' \"$STAGER_UPSTREAM_KIND\"; printf 'B300_GRAND_STAGER_UPSTREAM_PAIR_POLICY=%q\\n' \"${B300_STAGER_PREPARED_UPSTREAM_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGER_UPSTREAM_BLOCK_POLICY=%q\\n' \"${B300_STAGER_PREPARED_UPSTREAM_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGER_HIGH_PAIR_POLICY=%q\\n' \"${B300_STAGER_PREPARED_HIGH_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGER_HIGH_BLOCK_POLICY=%q\\n' \"${B300_STAGER_PREPARED_HIGH_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGER_HIGH_PAIR_L2_BYTES=%q\\n' \"${B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGER_HIGH_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGER_PAIR_POLICY=%q\\n' \"${B300_STAGER_PREPARED_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGER_BLOCK_POLICY=%q\\n' \"${B300_STAGER_PREPARED_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGER_STAGED_SPEEDUP=%q\\n' \"${B300_STAGER_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGER_MANIFEST=%q\\n' \"${B300_STAGER_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGER_SEARCH_PAIR_POLICIES=%q\\n' \"$STAGER_PAIR_POLICY_LIST\"; printf 'B300_GRAND_STAGER_SEARCH_BLOCK_POLICIES=%q\\n' \"$STAGER_BLOCK_POLICY_LIST\"\n",'Stage-R summary')
rep("  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEO_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEP_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEQ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'","  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEO_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEP_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEQ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGER_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",'Stage-R integrated marker')
for marker in ('RUN_STAGER=','STAGER_MIN_SPEEDUP=','STAGER_PAIR_POLICY_LIST=','STAGER_BLOCK_POLICY_LIST=','b300x8-nextgen-hybrid8-stager-ilp2-load-policy-staged-fullprime-race.sh','MODE=stager_ilp2_grand','MODE=stager_ilp2_joint','B300_GRAND_STAGER_OK=','B300_GRAND_STAGER_PAIR_POLICY=','B300_GRAND_STAGER_INTEGRATED=1'):
    if marker not in s: raise SystemExit('missing generated Stage-R artifact: '+marker)
if s.count('b300x8-race-external-forced-profiled-once.sh')!=1: raise SystemExit('Stage-R selector must retain exactly one complete-prime race')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_stager_integrated=1 upstream=Q>P>O>N prepare_only=1 fallback=1 single_complete_prime=1')
