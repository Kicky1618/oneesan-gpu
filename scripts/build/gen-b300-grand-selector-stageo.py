#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-grand-selector-stageo.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()

def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)

rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')

anchor='STAGEN_RACE_PREFIX="${STAGEN_RACE_PREFIX:-${PREFIX}.stagen-pairblock.promote}"\n'
rep(anchor,anchor+'''STAGEO_PREFIX="${STAGEO_PREFIX:-${PREFIX}.stageo-cgl2}"
STAGEO_WINNER_ENV="${STAGEO_WINNER_ENV:-${STAGEO_PREFIX}_winner.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-${PREFIX}.stageo-cgl2.prepared.env}"
STAGEO_RACE_PREFIX="${STAGEO_RACE_PREFIX:-${PREFIX}.stageo-cgl2.promote}"
''','Stage-O paths')
anchor='RUN_STAGEN="${RUN_STAGEN:-1}"\n'
rep(anchor,anchor+'RUN_STAGEO="${RUN_STAGEO:-1}"\n','RUN_STAGEO')
anchor='STAGEN_BLOCK_POLICY_LIST="${STAGEN_BLOCK_POLICY_LIST:-default cg cs}"\n'
rep(anchor,anchor+'''STAGEO_MIN_SPEEDUP="${STAGEO_MIN_SPEEDUP:-1.002}"
STAGEO_PAIR_L2_LIST="${STAGEO_PAIR_L2_LIST:-0 64 128 256}"
STAGEO_BLOCK_L2_LIST="${STAGEO_BLOCK_L2_LIST:-0 64 128 256}"
''','Stage-O knobs')
rep('for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN; do','for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO; do','boolean validation')
rep(
 'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
 'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speedup validation')
anchor='STAGEN_BLOCK_POLICY_LIST="$(normalize_policies "$STAGEN_BLOCK_POLICY_LIST")"\n'
rep(anchor,anchor+'''normalize_l2_sizes(){
  local raw="$1" out=() b old seen
  for b in $raw; do case "$b" in 0|64|128|256) ;; *) echo "bad CG L2 bytes=$b" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$b" ]] && seen=1; done; ((seen)) || out+=("$b"); done
  ((${#out[@]})) || { echo 'CG L2 list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
STAGEO_PAIR_L2_LIST="$(normalize_l2_sizes "$STAGEO_PAIR_L2_LIST")"
STAGEO_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGEO_BLOCK_L2_LIST")"
''','Stage-O L2 normalization')
rep('  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','  "$(dirname "$STAGEK_PREPARE_ENV")" "$(dirname "$STAGEL_PREPARE_ENV")" "$(dirname "$STAGEM_PREPARE_ENV")" "$(dirname "$STAGEN_PREPARE_ENV")" "$(dirname "$STAGEO_PREPARE_ENV")" \\\n  "$(dirname "$RACE_PREFIX")" "$WORK_ROOT"','mkdir Stage O')

stage_o=r'''
STAGEO_OK=0; STAGEO_RC=0
if ((STAGEN_OK)); then
  set +e
  PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$HYBRID_NS_WINNER_ENV" \
    STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" \
    ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" NGPU=8 \
    RUN_STAGED="$RUN_STAGEO" PREPARE_ONLY=1 MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP" \
    PAIR_L2_LIST="$STAGEO_PAIR_L2_LIST" BLOCK_L2_LIST="$STAGEO_BLOCK_L2_LIST" \
    STAGED_PREFIX="$STAGEO_PREFIX" WINNER_ENV="$STAGEO_WINNER_ENV" RACE_PREFIX="$STAGEO_RACE_PREFIX" PREPARE_ENV="$STAGEO_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh" 27
  STAGEO_RC=$?; set -e
  if ((STAGEO_RC==0)); then
    source "$STAGEO_PREPARE_ENV"
    [[ "${B300_STAGEO_PREPARED:-0}" == 1 && -x "$B300_STAGEO_PREPARED_BIN" && -x "$B300_STAGEO_PREPARED_CONTROL_BIN" ]] || exit 3
    [[ "$B300_STAGEO_PREPARED_MOD" == "$PRIME" && "$B300_STAGEO_PREPARED_NGPU" == 8 ]] || { echo 'Stage-O modulus/GPU drift' >&2; exit 3; }
    [[ "$B300_STAGEO_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" ]] || { echo 'Stage-O control is not exact Stage-N winner' >&2; exit 3; }
    [[ "$B300_STAGEO_PREPARED_PAIR_POLICY" == "$B300_STAGEN_PREPARED_PAIR_POLICY" && "$B300_STAGEO_PREPARED_BLOCK_POLICY" == "$B300_STAGEN_PREPARED_BLOCK_POLICY" && "$B300_STAGEO_PREPARED_MATE_LOAD_POLICY" == "$B300_STAGEN_PREPARED_MATE_LOAD_POLICY" ]] || { echo 'Stage-O changed Stage-N load policy' >&2; exit 3; }
    [[ "$B300_STAGEO_PREPARED_PAIR_L2_BYTES" != "$B300_STAGEO_PREPARED_BASE_PAIR_L2_BYTES" || "$B300_STAGEO_PREPARED_BLOCK_L2_BYTES" != "$B300_STAGEO_PREPARED_BASE_BLOCK_L2_BYTES" ]] || { echo 'Stage-O returned inherited L2 baseline' >&2; exit 3; }
    STAGEO_OK=1
  elif ((STAGEO_RC==4)); then
    echo 'grand selector: Stage-O CG L2 refinement rejected; retaining Stage N' >&2
  else
    exit "$STAGEO_RC"
  fi
fi
'''
rep('\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','\n'+stage_o+'\nP_BIN=""; P_LABEL=""; P_THREADS=256\n','Stage-O preparation insertion')

branches=r'''if ((STAGEO_OK && NEXTSELF_OK)); then
  MODE=stageo_cgl2_grand
  P_BIN="$B300_STAGEO_PREPARED_BIN"; P_LABEL="$B300_STAGEO_PREPARED_LABEL"; P_THREADS="$B300_STAGEO_PREPARED_THREADS"
  B_BIN="$B300_STAGEO_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEO_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEO_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$B300_NEXTSELF_PREPARED_BIN"; E2_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; E2_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif ((STAGEO_OK)); then
  MODE=stageo_cgl2_joint
  P_BIN="$B300_STAGEO_PREPARED_BIN"; P_LABEL="$B300_STAGEO_PREPARED_LABEL"; P_THREADS="$B300_STAGEO_PREPARED_THREADS"
  B_BIN="$B300_STAGEO_PREPARED_CONTROL_BIN"; B_LABEL="$B300_STAGEO_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_STAGEO_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E2_BIN="$JOINT_PRIMARY_BIN"; E2_LABEL="$JOINT_PRIMARY_LABEL"; E2_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then E3_BIN="$JOINT_BASE_BIN"; E3_LABEL="$JOINT_BASE_LABEL"; E3_THREADS="$JOINT_BASE_THREADS"; fi
elif ((STAGEN_OK && NEXTSELF_OK)); then'''
rep('if ((STAGEN_OK && NEXTSELF_OK)); then',branches,'Stage-O branch pair')

suffix='stagel_guard_grand|stagem_mateload_grand|stagen_pairblock_grand)'
if s.count(suffix)!=2: raise SystemExit(f'drop-mode suffix: expected two anchors, got {s.count(suffix)}')
s=s.replace(suffix,'stagel_guard_grand|stagem_mateload_grand|stagen_pairblock_grand|stageo_cgl2_grand)',2)

anchor="  printf 'B300_GRAND_STAGEN_OK=%q\\n' \"$STAGEN_OK\"; printf 'B300_GRAND_STAGEN_MIN_SPEEDUP=%q\\n' \"$STAGEN_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEN_UPSTREAM_KIND=%q\\n' \"$STAGEN_UPSTREAM_KIND\"; printf 'B300_GRAND_STAGEN_PAIR_POLICY=%q\\n' \"${B300_STAGEN_PREPARED_PAIR_POLICY:-default}\"; printf 'B300_GRAND_STAGEN_BLOCK_POLICY=%q\\n' \"${B300_STAGEN_PREPARED_BLOCK_POLICY:-default}\"; printf 'B300_GRAND_STAGEN_BASE_COUNT_POLICY=%q\\n' \"${B300_STAGEN_PREPARED_BASE_COUNT_POLICY:-default}\"; printf 'B300_GRAND_STAGEN_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEN_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEN_MANIFEST=%q\\n' \"${B300_STAGEN_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEN_SEARCH_PAIR_POLICIES=%q\\n' \"$STAGEN_PAIR_POLICY_LIST\"; printf 'B300_GRAND_STAGEN_SEARCH_BLOCK_POLICIES=%q\\n' \"$STAGEN_BLOCK_POLICY_LIST\"\n"
rep(anchor,anchor+"  printf 'B300_GRAND_STAGEO_OK=%q\\n' \"$STAGEO_OK\"; printf 'B300_GRAND_STAGEO_MIN_SPEEDUP=%q\\n' \"$STAGEO_MIN_SPEEDUP\"; printf 'B300_GRAND_STAGEO_PAIR_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_BASE_PAIR_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_BASE_PAIR_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_BASE_BLOCK_L2_BYTES=%q\\n' \"${B300_STAGEO_PREPARED_BASE_BLOCK_L2_BYTES:-0}\"; printf 'B300_GRAND_STAGEO_STAGED_SPEEDUP=%q\\n' \"${B300_STAGEO_PREPARED_STAGED_SPEEDUP:-1.0}\"; printf 'B300_GRAND_STAGEO_MANIFEST=%q\\n' \"${B300_STAGEO_PREPARED_MANIFEST:-}\"; printf 'B300_GRAND_STAGEO_SEARCH_PAIR_L2=%q\\n' \"$STAGEO_PAIR_L2_LIST\"; printf 'B300_GRAND_STAGEO_SEARCH_BLOCK_L2=%q\\n' \"$STAGEO_BLOCK_L2_LIST\"\n",'Stage-O summary')
rep("  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'","  printf 'B300_GRAND_STAGEJ_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEK_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEL_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEM_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEN_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEO_INTEGRATED=1\\n'; printf 'B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1\\n'; printf 'B300_GRAND_COMPLETE_PRIME_RACES=1\\n'",'Stage-O integrated marker')

for marker in ('RUN_STAGEO=','STAGEO_MIN_SPEEDUP=','STAGEO_PAIR_L2_LIST=','STAGEO_BLOCK_L2_LIST=','b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh','MODE=stageo_cgl2_grand','MODE=stageo_cgl2_joint','B300_GRAND_STAGEO_OK=','B300_GRAND_STAGEO_PAIR_L2_BYTES=','B300_GRAND_STAGEO_BLOCK_L2_BYTES=','B300_GRAND_STAGEO_INTEGRATED=1','B300_GRAND_COMPLETE_PRIME_RACES=1'):
    if marker not in s: raise SystemExit('missing generated Stage-O artifact: '+marker)
if s.count('b300x8-race-external-forced-profiled-once.sh')!=1: raise SystemExit('generated selector must retain exactly one complete-prime race')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_stageo_integrated=1 stage_o_after_n=1 prepare_only=1 fallback_n=1 single_complete_prime=1')
