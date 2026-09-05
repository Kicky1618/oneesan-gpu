#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-grand-firstpass-stagem.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()

def rep(old:str,new:str,label:str)->None:
    global s
    if new in s:
        return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)

rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')

anchor='RUN_STAGEK="${RUN_STAGEK:-1}"\n'
rep(anchor,anchor+'RUN_STAGEL="${RUN_STAGEL:-1}"\nRUN_STAGEM="${RUN_STAGEM:-1}"\n','run L/M')
anchor='STAGEK_MIN_SPEEDUP="${STAGEK_MIN_SPEEDUP:-1.002}"\n'
rep(anchor,anchor+'STAGEL_MIN_SPEEDUP="${STAGEL_MIN_SPEEDUP:-1.002}"\nSTAGEL_GUARD_LIST="${STAGEL_GUARD_LIST:-bb pb bp pp}"\nSTAGEM_MIN_SPEEDUP="${STAGEM_MIN_SPEEDUP:-1.002}"\nSTAGEM_POLICY_LIST="${STAGEM_POLICY_LIST:-default cg cs}"\n','L/M knobs')
rep(
 'for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK; do',
 'for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM; do',
 'boolean validation')

anchor='MATE_EVICT_LIST="$(normalize_evicts "$MATE_EVICT_LIST")"\n'
rep(anchor,anchor+'''normalize_guards(){
  local raw="$1" out=() g old seen
  for g in $raw; do case "$g" in bb|pb|bp|pp) ;; *) echo "bad guard profile=$g" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$g" ]] && seen=1; done; ((seen)) || out+=("$g"); done
  ((${#out[@]})) || { echo 'guard list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
normalize_mate_load_policies(){
  local raw="$1" out=() p old seen
  for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad mate-load policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done
  ((${#out[@]})) || { echo 'mate-load policy list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
STAGEL_GUARD_LIST="$(normalize_guards "$STAGEL_GUARD_LIST")"
STAGEM_POLICY_LIST="$(normalize_mate_load_policies "$STAGEM_POLICY_LIST")"
case " $STAGEL_GUARD_LIST " in *' bb '*) ;; *) echo 'STAGEL_GUARD_LIST must include bb' >&2; exit 2;; esac
case " $STAGEM_POLICY_LIST " in *' default '*) ;; *) echo 'STAGEM_POLICY_LIST must include default' >&2; exit 2;; esac
''','L/M list normalization')

rep(
 'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
 'python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY',
 'speed validation')

anchor="  printf 'run_stagek=%s\\n' \"$RUN_STAGEK\"\n"
rep(anchor,anchor+"  printf 'run_stagel=%s\\n' \"$RUN_STAGEL\"\n  printf 'run_stagem=%s\\n' \"$RUN_STAGEM\"\n",'meta run L/M')
anchor="  printf 'stagek_mate_evict_list=%s\\n' \"$MATE_EVICT_LIST\"\n"
rep(anchor,anchor+"  printf 'stagel_min_speedup=%s\\n' \"$STAGEL_MIN_SPEEDUP\"\n  printf 'stagel_guard_list=%s\\n' \"$STAGEL_GUARD_LIST\"\n  printf 'stagem_min_speedup=%s\\n' \"$STAGEM_MIN_SPEEDUP\"\n  printf 'stagem_policy_list=%s\\n' \"$STAGEM_POLICY_LIST\"\n  printf 'stagem_firstpass_base_sha256=%s\\n' \"${B300_STAGEM_FIRSTPASS_BASE_SHA256:-}\"\n  printf 'stagem_firstpass_generator_sha256=%s\\n' \"${B300_STAGEM_FIRSTPASS_GENERATOR_SHA256:-}\"\n",'meta L/M knobs')

anchor='bash "$ONEESAN_ROOT/scripts/bench/b300-grand-stagek-contract-preflight.sh"\n'
rep(anchor,anchor+'bash "$ONEESAN_ROOT/scripts/bench/b300-stagem-preflight.sh"\nPATCH_ONLY=1 bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagem.sh" 27 >/dev/null\n','GPU-free Stage M preflight')

rep(
 '  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" \\\n',
 '  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" \\\n',
 'selector run flags')
rep(
 '  STAGEI_MIN_SPEEDUP="$STAGEI_MIN_SPEEDUP" STAGEJ_MIN_SPEEDUP="$STAGEJ_MIN_SPEEDUP" STAGEK_MIN_SPEEDUP="$STAGEK_MIN_SPEEDUP" \\\n',
 '  STAGEI_MIN_SPEEDUP="$STAGEI_MIN_SPEEDUP" STAGEJ_MIN_SPEEDUP="$STAGEJ_MIN_SPEEDUP" STAGEK_MIN_SPEEDUP="$STAGEK_MIN_SPEEDUP" STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" \\\n',
 'selector speed flags')
rep(
 '  MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" MATE_EVICT="$MATE_EVICT" MATE_EVICT_LIST="$MATE_EVICT_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh" 27 "$@" \\\n',
 '  MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" MATE_EVICT="$MATE_EVICT" MATE_EVICT_LIST="$MATE_EVICT_LIST" STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagem.sh" 27 "$@" \\\n',
 'Stage-M selector entrypoint')

rep(
 '[[ "${B300_GRAND_PREPARED:-0}" == 1 && "${B300_GRAND_STAGEJ_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEK_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && \\\n   "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K single-race provenance markers missing\' >&2; exit 4;\n}',
 '[[ "${B300_GRAND_PREPARED:-0}" == 1 && "${B300_GRAND_STAGEJ_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEK_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEL_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && \\\n   "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M single-race provenance markers missing\' >&2; exit 4;\n}',
 'summary provenance')

anchor="  printf 'B300_GRAND_SELECTED_STAGEK_SEARCH_EVICTS=%q\\n' \"$MATE_EVICT_LIST\"\n"
rep(anchor,anchor+'''  printf 'B300_GRAND_SELECTED_STAGEL_ENABLED=%q\n' "$RUN_STAGEL"
  printf 'B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP=%q\n' "$STAGEL_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEL_ACCEPTED=%q\n' "${B300_GRAND_STAGEL_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEL_PROFILE=%q\n' "${B300_GRAND_STAGEL_PROFILE:-bb}"
  printf 'B300_GRAND_SELECTED_STAGEL_SELF_GUARD=%q\n' "${B300_GRAND_STAGEL_SELF_GUARD:-branch}"
  printf 'B300_GRAND_SELECTED_STAGEL_MATE_GUARD=%q\n' "${B300_GRAND_STAGEL_MATE_GUARD:-branch}"
  printf 'B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEL_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES=%q\n' "$STAGEL_GUARD_LIST"
  printf 'B300_GRAND_SELECTED_STAGEM_ENABLED=%q\n' "$RUN_STAGEM"
  printf 'B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP=%q\n' "$STAGEM_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEM_ACCEPTED=%q\n' "${B300_GRAND_STAGEM_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEM_POLICY=%q\n' "${B300_GRAND_STAGEM_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEM_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES=%q\n' "$STAGEM_POLICY_LIST"
''','selected L/M contract')

anchor="  printf 'stagek_staged_speedup=%s\\n' \"${B300_GRAND_STAGEK_STAGED_SPEEDUP:-1.0}\"\n"
rep(anchor,anchor+'''  printf 'stagel_enabled=%s\n' "$RUN_STAGEL"
  printf 'stagel_min_speedup=%s\n' "$STAGEL_MIN_SPEEDUP"
  printf 'stagel_accepted=%s\n' "${B300_GRAND_STAGEL_OK:-0}"
  printf 'stagel_profile=%s\n' "${B300_GRAND_STAGEL_PROFILE:-bb}"
  printf 'stagel_guards=%s/%s\n' "${B300_GRAND_STAGEL_SELF_GUARD:-branch}" "${B300_GRAND_STAGEL_MATE_GUARD:-branch}"
  printf 'stagel_staged_speedup=%s\n' "${B300_GRAND_STAGEL_STAGED_SPEEDUP:-1.0}"
  printf 'stagem_enabled=%s\n' "$RUN_STAGEM"
  printf 'stagem_min_speedup=%s\n' "$STAGEM_MIN_SPEEDUP"
  printf 'stagem_accepted=%s\n' "${B300_GRAND_STAGEM_OK:-0}"
  printf 'stagem_policy=%s\n' "${B300_GRAND_STAGEM_POLICY:-default}"
  printf 'stagem_staged_speedup=%s\n' "${B300_GRAND_STAGEM_STAGED_SPEEDUP:-1.0}"
''','final meta L/M')

for marker in ('RUN_STAGEL=','RUN_STAGEM=','STAGEM_POLICY_LIST=','b300x8-joint-nextself-hybrid8-select-stagem.sh','B300_GRAND_SELECTED_STAGEM_ACCEPTED','B300_GRAND_SELECTED_STAGEL_ACCEPTED','B300_GRAND_STAGEM_INTEGRATED'):
    if marker not in s: raise SystemExit('missing generated firstpass artifact: '+marker)
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_firstpass_stagem=1 selected_schema_compatible=3 stage_lm_provenance=1')