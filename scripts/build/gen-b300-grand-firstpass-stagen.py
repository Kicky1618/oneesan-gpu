#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-firstpass-stagen.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='RUN_STAGEM="${RUN_STAGEM:-1}"\n'; rep(anchor,anchor+'RUN_STAGEN="${RUN_STAGEN:-1}"\n','RUN_STAGEN')
anchor='STAGEM_POLICY_LIST="${STAGEM_POLICY_LIST:-default cg cs}"\n'; rep(anchor,anchor+'STAGEN_MIN_SPEEDUP="${STAGEN_MIN_SPEEDUP:-1.002}"\nSTAGEN_PAIR_POLICY_LIST="${STAGEN_PAIR_POLICY_LIST:-default cg cs}"\nSTAGEN_BLOCK_POLICY_LIST="${STAGEN_BLOCK_POLICY_LIST:-default cg cs}"\n','Stage-N knobs')
rep('for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM; do','for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN; do','boolean validation')
anchor='STAGEM_POLICY_LIST="$(normalize_mate_load_policies "$STAGEM_POLICY_LIST")"\n'; rep(anchor,anchor+'STAGEN_PAIR_POLICY_LIST="$(normalize_mate_load_policies "$STAGEN_PAIR_POLICY_LIST")"\nSTAGEN_BLOCK_POLICY_LIST="$(normalize_mate_load_policies "$STAGEN_BLOCK_POLICY_LIST")"\n','Stage-N list normalization')
rep('python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speed validation')
anchor="  printf 'stagem_policy_list=%s\\n' \"$STAGEM_POLICY_LIST\"\n"; rep(anchor,anchor+"  printf 'run_stagen=%s\\n' \"$RUN_STAGEN\"\n  printf 'stagen_min_speedup=%s\\n' \"$STAGEN_MIN_SPEEDUP\"\n  printf 'stagen_pair_policy_list=%s\\n' \"$STAGEN_PAIR_POLICY_LIST\"\n  printf 'stagen_block_policy_list=%s\\n' \"$STAGEN_BLOCK_POLICY_LIST\"\n",'meta Stage N')
anchor='bash "$ONEESAN_ROOT/scripts/bench/b300-stagem-preflight.sh"\n'; rep(anchor,anchor+'bash "$ONEESAN_ROOT/scripts/bench/b300-stagen-preflight.sh"\nbash "$ONEESAN_ROOT/scripts/bench/b300-grand-stagen-contract-preflight.sh"\n','Stage-N GPU-free preflight')
rep(' mate_load=[$STAGEM_POLICY_LIST] ===',' mate_load=[$STAGEM_POLICY_LIST] pair_load=[$STAGEN_PAIR_POLICY_LIST] block_load=[$STAGEN_BLOCK_POLICY_LIST] ===','Stage-N banner')
rep('  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" \\\n','  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" \\\n','selector run flags')
rep('  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" \\\n','  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" \\\n','selector speed flags')
rep('  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh" 27 "$@" \\\n','  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagen.sh" 27 "$@" \\\n','Stage-N selector entrypoint')
rep('   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && \\\n   "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M single-race provenance markers missing\' >&2; exit 4;','   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N single-race provenance markers missing\' >&2; exit 4;','summary provenance')
anchor="  printf 'B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES=%q\\n' \"$STAGEM_POLICY_LIST\"\n"; rep(anchor,anchor+'''  printf 'B300_GRAND_SELECTED_STAGEN_ENABLED=%q\n' "$RUN_STAGEN"
  printf 'B300_GRAND_SELECTED_STAGEN_MIN_SPEEDUP=%q\n' "$STAGEN_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEN_ACCEPTED=%q\n' "${B300_GRAND_STAGEN_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND=%q\n' "${B300_GRAND_STAGEN_UPSTREAM_KIND:-}"
  printf 'B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=%q\n' "${B300_GRAND_STAGEN_PAIR_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY=%q\n' "${B300_GRAND_STAGEN_BLOCK_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY=%q\n' "${B300_GRAND_STAGEN_BASE_COUNT_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEN_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEN_SEARCH_PAIR_POLICIES=%q\n' "$STAGEN_PAIR_POLICY_LIST"
  printf 'B300_GRAND_SELECTED_STAGEN_SEARCH_BLOCK_POLICIES=%q\n' "$STAGEN_BLOCK_POLICY_LIST"
''','selected Stage N')
for marker in ('RUN_STAGEN=','STAGEN_PAIR_POLICY_LIST=','STAGEN_BLOCK_POLICY_LIST=','b300x8-joint-nextself-hybrid8-select-stagen.sh','B300_GRAND_SELECTED_STAGEN_ACCEPTED','B300_GRAND_STAGEN_INTEGRATED'):
    if marker not in s: raise SystemExit('missing generated Stage-N firstpass artifact: '+marker)
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_firstpass_stagen=1 selected_schema_compatible=3 single_complete_prime=1')
