#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-firstpass-stageo.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='RUN_STAGEN="${RUN_STAGEN:-1}"\n'; rep(anchor,anchor+'RUN_STAGEO="${RUN_STAGEO:-1}"\n','RUN_STAGEO')
anchor='STAGEN_BLOCK_POLICY_LIST="${STAGEN_BLOCK_POLICY_LIST:-default cg cs}"\n'; rep(anchor,anchor+'STAGEO_MIN_SPEEDUP="${STAGEO_MIN_SPEEDUP:-1.002}"\nSTAGEO_PAIR_L2_LIST="${STAGEO_PAIR_L2_LIST:-0 64 128 256}"\nSTAGEO_BLOCK_L2_LIST="${STAGEO_BLOCK_L2_LIST:-0 64 128 256}"\n','Stage-O knobs')
rep('for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN; do','for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO; do','boolean validation')
anchor='STAGEN_BLOCK_POLICY_LIST="$(normalize_mate_load_policies "$STAGEN_BLOCK_POLICY_LIST")"\n'
rep(anchor,anchor+'''normalize_l2_sizes(){
  local raw="$1" out=() b old seen
  for b in $raw; do case "$b" in 0|64|128|256) ;; *) echo "bad CG L2 bytes=$b" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$b" ]] && seen=1; done; ((seen)) || out+=("$b"); done
  ((${#out[@]})) || { echo 'CG L2 list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
STAGEO_PAIR_L2_LIST="$(normalize_l2_sizes "$STAGEO_PAIR_L2_LIST")"
STAGEO_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGEO_BLOCK_L2_LIST")"
''','Stage-O list normalization')
rep('python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speed validation')
anchor="  printf 'stagen_block_policy_list=%s\\n' \"$STAGEN_BLOCK_POLICY_LIST\"\n"
rep(anchor,anchor+"  printf 'run_stageo=%s\\n' \"$RUN_STAGEO\"\n  printf 'stageo_min_speedup=%s\\n' \"$STAGEO_MIN_SPEEDUP\"\n  printf 'stageo_pair_l2_list=%s\\n' \"$STAGEO_PAIR_L2_LIST\"\n  printf 'stageo_block_l2_list=%s\\n' \"$STAGEO_BLOCK_L2_LIST\"\n",'meta Stage O')
anchor='bash "$ONEESAN_ROOT/scripts/bench/b300-grand-stagen-contract-preflight.sh"\n'
rep(anchor,anchor+'bash "$ONEESAN_ROOT/scripts/bench/b300-stageo-preflight.sh"\nbash "$ONEESAN_ROOT/scripts/bench/b300-grand-stageo-contract-preflight.sh"\n','Stage-O GPU-free preflight')
rep(' block_load=[$STAGEN_BLOCK_POLICY_LIST] ===',' block_load=[$STAGEN_BLOCK_POLICY_LIST] pair_l2=[$STAGEO_PAIR_L2_LIST] block_l2=[$STAGEO_BLOCK_L2_LIST] ===','Stage-O banner')
rep('  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" \\\n','  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" RUN_STAGEO="$RUN_STAGEO" \\\n','selector run flags')
rep('  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" \\\n','  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" STAGEO_MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP" \\\n','selector speed flags')
rep('  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagen.sh" 27 "$@" \\\n','  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" STAGEO_PAIR_L2_LIST="$STAGEO_PAIR_L2_LIST" STAGEO_BLOCK_L2_LIST="$STAGEO_BLOCK_L2_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageo.sh" 27 "$@" \\\n','Stage-O selector entrypoint')
rep('   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N single-race provenance markers missing\' >&2; exit 4;','   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEO_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N/O single-race provenance markers missing\' >&2; exit 4;','summary provenance')
anchor="  printf 'B300_GRAND_SELECTED_STAGEN_SEARCH_BLOCK_POLICIES=%q\\n' \"$STAGEN_BLOCK_POLICY_LIST\"\n"
rep(anchor,anchor+'''  printf 'B300_GRAND_SELECTED_STAGEO_ENABLED=%q\n' "$RUN_STAGEO"
  printf 'B300_GRAND_SELECTED_STAGEO_MIN_SPEEDUP=%q\n' "$STAGEO_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGEO_ACCEPTED=%q\n' "${B300_GRAND_STAGEO_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES=%q\n' "${B300_GRAND_STAGEO_PAIR_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES=%q\n' "${B300_GRAND_STAGEO_BLOCK_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES=%q\n' "${B300_GRAND_STAGEO_BASE_PAIR_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES=%q\n' "${B300_GRAND_STAGEO_BASE_BLOCK_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGEO_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGEO_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGEO_SEARCH_PAIR_L2=%q\n' "$STAGEO_PAIR_L2_LIST"
  printf 'B300_GRAND_SELECTED_STAGEO_SEARCH_BLOCK_L2=%q\n' "$STAGEO_BLOCK_L2_LIST"
''','selected Stage O')
for marker in ('RUN_STAGEO=','STAGEO_PAIR_L2_LIST=','STAGEO_BLOCK_L2_LIST=','b300x8-joint-nextself-hybrid8-select-stageo.sh','B300_GRAND_SELECTED_STAGEO_ACCEPTED','B300_GRAND_STAGEO_INTEGRATED'):
    if marker not in s: raise SystemExit('missing generated Stage-O firstpass artifact: '+marker)
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_firstpass_stageo=1 selected_schema_compatible=3 single_complete_prime=1')
