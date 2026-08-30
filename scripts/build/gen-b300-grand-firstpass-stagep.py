#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-firstpass-stagep.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='RUN_STAGEO="${RUN_STAGEO:-1}"\n'; rep(anchor,anchor+'RUN_STAGEP="${RUN_STAGEP:-1}"\n','RUN_STAGEP')
anchor='STAGEO_BLOCK_L2_LIST="${STAGEO_BLOCK_L2_LIST:-0 64 128 256}"\n'; rep(anchor,anchor+'STAGEP_MIN_SPEEDUP="${STAGEP_MIN_SPEEDUP:-1.002}"\nSTAGEP_MATE_L2_LIST="${STAGEP_MATE_L2_LIST:-0 64 128 256}"\n','Stage-P knobs')
rep('for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO; do','for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO RUN_STAGEP; do','boolean validation')
anchor='STAGEO_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGEO_BLOCK_L2_LIST")"\n'; rep(anchor,anchor+'STAGEP_MATE_L2_LIST="$(normalize_l2_sizes "$STAGEP_MATE_L2_LIST")"\ncase " $STAGEP_MATE_L2_LIST " in *\' 0 \'*) ;; *) echo \'STAGEP_MATE_L2_LIST must include 0 baseline\' >&2; exit 2;; esac\n','Stage-P L2 normalization')
rep('python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" "$STAGEP_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\',\'STAGEP_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speed validation')
anchor="  printf 'stageo_block_l2_list=%s\\n' \"$STAGEO_BLOCK_L2_LIST\"\n"; rep(anchor,anchor+"  printf 'run_stagep=%s\\n' \"$RUN_STAGEP\"\n  printf 'stagep_min_speedup=%s\\n' \"$STAGEP_MIN_SPEEDUP\"\n  printf 'stagep_mate_l2_list=%s\\n' \"$STAGEP_MATE_L2_LIST\"\n",'meta Stage P')
anchor='bash "$ONEESAN_ROOT/scripts/bench/b300-grand-stageo-contract-preflight.sh"\n'; rep(anchor,anchor+'bash "$ONEESAN_ROOT/scripts/bench/b300-stagep-preflight.sh"\nbash "$ONEESAN_ROOT/scripts/bench/b300-grand-stagep-contract-preflight.sh"\n','Stage-P GPU-free preflight')
rep(' block_l2=[$STAGEO_BLOCK_L2_LIST] ===',' block_l2=[$STAGEO_BLOCK_L2_LIST] mate_l2=[$STAGEP_MATE_L2_LIST] ===','Stage-P banner')
rep('  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" RUN_STAGEO="$RUN_STAGEO" \\\n','  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" RUN_STAGEO="$RUN_STAGEO" RUN_STAGEP="$RUN_STAGEP" \\\n','selector run flags')
rep('  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" STAGEO_MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP" \\\n','  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" STAGEO_MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP" STAGEP_MIN_SPEEDUP="$STAGEP_MIN_SPEEDUP" \\\n','selector speed flags')
rep('  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" STAGEO_PAIR_L2_LIST="$STAGEO_PAIR_L2_LIST" STAGEO_BLOCK_L2_LIST="$STAGEO_BLOCK_L2_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageo.sh" 27 "$@" \\\n','  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" STAGEO_PAIR_L2_LIST="$STAGEO_PAIR_L2_LIST" STAGEO_BLOCK_L2_LIST="$STAGEO_BLOCK_L2_LIST" STAGEP_MATE_L2_LIST="$STAGEP_MATE_L2_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagep.sh" 27 "$@" \\\n','Stage-P selector entrypoint')
rep('   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEO_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N/O single-race provenance markers missing\' >&2; exit 4;','   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEO_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEP_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N/O/P single-race provenance markers missing\' >&2; exit 4;','summary provenance')
# Insert selected Stage-P provenance immediately before the stable single-prime marker.
anchor="  printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1\\n'\n"
rep(anchor,'''  printf 'B300_GRAND_SELECTED_STAGEP_ENABLED=%q\n' "$RUN_STAGEP"
  printf 'B300_GRAND_SELECTED_STAGEP_MIN_SPEEDUP=%q\n' "$STAGEP_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEP_ACCEPTED=%q\n' "${B300_GRAND_STAGEP_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM=%q\n' "${B300_GRAND_STAGEP_COUNT_UPSTREAM:-}"
  printf 'B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES=%q\n' "${B300_GRAND_STAGEP_BASE_MATE_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=%q\n' "${B300_GRAND_STAGEP_MATE_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGEP_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEP_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEP_SEARCH_MATE_L2=%q\n' "$STAGEP_MATE_L2_LIST"
'''+anchor,'selected Stage P')
for marker in ('RUN_STAGEP=','STAGEP_MATE_L2_LIST=','b300x8-joint-nextself-hybrid8-select-stagep.sh','B300_GRAND_SELECTED_STAGEP_ACCEPTED','B300_GRAND_STAGEP_INTEGRATED'):
    if marker not in s: raise SystemExit('missing generated Stage-P firstpass artifact: '+marker)
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_firstpass_stagep=1 selected_schema_compatible=3 single_complete_prime=1')
