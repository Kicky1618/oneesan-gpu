#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-firstpass-stager.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
def rep(old:str,new:str,label:str)->None:
    global s
    if new in s: return
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one anchor, got {n}')
    s=s.replace(old,new,1)
rep('source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"','source "${ONEESAN_ROOT:?ONEESAN_ROOT must be exported}/scripts/lib/common.sh"','common source')
anchor='RUN_STAGEQ="${RUN_STAGEQ:-1}"\n'; rep(anchor,anchor+'RUN_STAGER="${RUN_STAGER:-1}"\n','RUN_STAGER')
anchor='STAGEQ_BLOCK_L2_LIST="${STAGEQ_BLOCK_L2_LIST:-0 64 128 256}"\n'
rep(anchor,anchor+'''STAGER_MIN_SPEEDUP="${STAGER_MIN_SPEEDUP:-1.002}"
STAGER_PAIR_POLICY_LIST="${STAGER_PAIR_POLICY_LIST:-default cg cs}"
STAGER_BLOCK_POLICY_LIST="${STAGER_BLOCK_POLICY_LIST:-default cg cs}"
''','Stage-R knobs')
rep('for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO RUN_STAGEP RUN_STAGEQ; do','for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE RUN_HYBRID_NS_STAGE RUN_STAGEI RUN_STAGEJ RUN_STAGEK RUN_STAGEL RUN_STAGEM RUN_STAGEN RUN_STAGEO RUN_STAGEP RUN_STAGEQ RUN_STAGER; do','boolean validation')
anchor='STAGEQ_BLOCK_L2_LIST="$(normalize_l2_sizes "$STAGEQ_BLOCK_L2_LIST")"\n'
rep(anchor,anchor+'''normalize_load_policies(){
  local raw="$1" out=() p old seen
  for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-R load policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done
  ((${#out[@]})) || { echo 'Stage-R policy list must not be empty' >&2; exit 2; }; printf '%s' "${out[*]}"
}
STAGER_PAIR_POLICY_LIST="$(normalize_load_policies "$STAGER_PAIR_POLICY_LIST")"
STAGER_BLOCK_POLICY_LIST="$(normalize_load_policies "$STAGER_BLOCK_POLICY_LIST")"
''','Stage-R policy normalization')
rep('python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" "$STAGEP_MIN_SPEEDUP" "$STAGEQ_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\',\'STAGEP_MIN_SPEEDUP\',\'STAGEQ_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','python3 - "$NEXTSELF_MIN_SPEEDUP" "$HYBRID_MIN_SPEEDUP" "$HYBRID_NS_MIN_SPEEDUP" "$STAGEI_MIN_SPEEDUP" "$STAGEJ_MIN_SPEEDUP" "$STAGEK_MIN_SPEEDUP" "$STAGEL_MIN_SPEEDUP" "$STAGEM_MIN_SPEEDUP" "$STAGEN_MIN_SPEEDUP" "$STAGEO_MIN_SPEEDUP" "$STAGEP_MIN_SPEEDUP" "$STAGEQ_MIN_SPEEDUP" "$STAGER_MIN_SPEEDUP" <<\'PY\'\nimport sys\nnames=(\'NEXTSELF_MIN_SPEEDUP\',\'HYBRID_MIN_SPEEDUP\',\'HYBRID_NS_MIN_SPEEDUP\',\'STAGEI_MIN_SPEEDUP\',\'STAGEJ_MIN_SPEEDUP\',\'STAGEK_MIN_SPEEDUP\',\'STAGEL_MIN_SPEEDUP\',\'STAGEM_MIN_SPEEDUP\',\'STAGEN_MIN_SPEEDUP\',\'STAGEO_MIN_SPEEDUP\',\'STAGEP_MIN_SPEEDUP\',\'STAGEQ_MIN_SPEEDUP\',\'STAGER_MIN_SPEEDUP\')\nfor name,v in zip(names,map(float,sys.argv[1:])):\n    if v < 1.0: raise SystemExit(f\'{name} must be >=1\')\nPY','speed validation')
anchor="  printf 'stageq_block_l2_list=%s\\n' \"$STAGEQ_BLOCK_L2_LIST\"\n"
rep(anchor,anchor+"  printf 'run_stager=%s\\n' \"$RUN_STAGER\"\n  printf 'stager_min_speedup=%s\\n' \"$STAGER_MIN_SPEEDUP\"\n  printf 'stager_pair_policy_list=%s\\n' \"$STAGER_PAIR_POLICY_LIST\"\n  printf 'stager_block_policy_list=%s\\n' \"$STAGER_BLOCK_POLICY_LIST\"\n",'meta Stage R')
anchor='bash "$ONEESAN_ROOT/scripts/bench/b300-grand-stageq-contract-preflight.sh"\n'
rep(anchor,anchor+'bash "$ONEESAN_ROOT/scripts/bench/b300-stager-preflight.sh"\nbash "$ONEESAN_ROOT/scripts/bench/b300-grand-stager-contract-preflight.sh"\n','Stage-R GPU-free preflight')
rep(' q_pair_l2=[$STAGEQ_PAIR_L2_LIST] q_block_l2=[$STAGEQ_BLOCK_L2_LIST] ===',' q_pair_l2=[$STAGEQ_PAIR_L2_LIST] q_block_l2=[$STAGEQ_BLOCK_L2_LIST] r_pair=[$STAGER_PAIR_POLICY_LIST] r_block=[$STAGER_BLOCK_POLICY_LIST] ===','Stage-R banner')
rep('  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" RUN_STAGEO="$RUN_STAGEO" RUN_STAGEP="$RUN_STAGEP" RUN_STAGEQ="$RUN_STAGEQ" \\\n','  RUN_STAGEI="$RUN_STAGEI" RUN_STAGEJ="$RUN_STAGEJ" RUN_STAGEK="$RUN_STAGEK" RUN_STAGEL="$RUN_STAGEL" RUN_STAGEM="$RUN_STAGEM" RUN_STAGEN="$RUN_STAGEN" RUN_STAGEO="$RUN_STAGEO" RUN_STAGEP="$RUN_STAGEP" RUN_STAGEQ="$RUN_STAGEQ" RUN_STAGER="$RUN_STAGER" \\\n','selector run flags')
rep('  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" STAGEO_MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP" STAGEP_MIN_SPEEDUP="$STAGEP_MIN_SPEEDUP" STAGEQ_MIN_SPEEDUP="$STAGEQ_MIN_SPEEDUP" \\\n','  STAGEL_MIN_SPEEDUP="$STAGEL_MIN_SPEEDUP" STAGEM_MIN_SPEEDUP="$STAGEM_MIN_SPEEDUP" STAGEN_MIN_SPEEDUP="$STAGEN_MIN_SPEEDUP" STAGEO_MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP" STAGEP_MIN_SPEEDUP="$STAGEP_MIN_SPEEDUP" STAGEQ_MIN_SPEEDUP="$STAGEQ_MIN_SPEEDUP" STAGER_MIN_SPEEDUP="$STAGER_MIN_SPEEDUP" \\\n','selector speed flags')
rep('  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" STAGEO_PAIR_L2_LIST="$STAGEO_PAIR_L2_LIST" STAGEO_BLOCK_L2_LIST="$STAGEO_BLOCK_L2_LIST" STAGEP_MATE_L2_LIST="$STAGEP_MATE_L2_LIST" STAGEQ_PAIR_L2_LIST="$STAGEQ_PAIR_L2_LIST" STAGEQ_BLOCK_L2_LIST="$STAGEQ_BLOCK_L2_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageq.sh" 27 "$@" \\\n','  STAGEL_GUARD_LIST="$STAGEL_GUARD_LIST" STAGEM_POLICY_LIST="$STAGEM_POLICY_LIST" STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST" STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST" STAGEO_PAIR_L2_LIST="$STAGEO_PAIR_L2_LIST" STAGEO_BLOCK_L2_LIST="$STAGEO_BLOCK_L2_LIST" STAGEP_MATE_L2_LIST="$STAGEP_MATE_L2_LIST" STAGEQ_PAIR_L2_LIST="$STAGEQ_PAIR_L2_LIST" STAGEQ_BLOCK_L2_LIST="$STAGEQ_BLOCK_L2_LIST" STAGER_PAIR_POLICY_LIST="$STAGER_PAIR_POLICY_LIST" STAGER_BLOCK_POLICY_LIST="$STAGER_BLOCK_POLICY_LIST" \\\n  PREFIX="$PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stager.sh" 27 "$@" \\\n','Stage-R selector entrypoint')
rep('   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEO_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEP_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEQ_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N/O/P/Q single-race provenance markers missing\' >&2; exit 4;','   "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEO_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEP_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGEQ_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGER_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {\n  echo \'grand Stage-I/J/K/L/M/N/O/P/Q/R single-race provenance markers missing\' >&2; exit 4;','summary provenance')
anchor="  printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1\\n'\n"
rep(anchor,'''  printf 'B300_GRAND_SELECTED_STAGER_ENABLED=%q\n' "$RUN_STAGER"
  printf 'B300_GRAND_SELECTED_STAGER_MIN_SPEEDUP=%q\n' "$STAGER_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGER_ACCEPTED=%q\n' "${B300_GRAND_STAGER_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=%q\n' "${B300_GRAND_STAGER_UPSTREAM_KIND:-}"
  printf 'B300_GRAND_SELECTED_STAGER_UPSTREAM_PAIR_POLICY=%q\n' "${B300_GRAND_STAGER_UPSTREAM_PAIR_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGER_UPSTREAM_BLOCK_POLICY=%q\n' "${B300_GRAND_STAGER_UPSTREAM_BLOCK_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY=%q\n' "${B300_GRAND_STAGER_HIGH_PAIR_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY=%q\n' "${B300_GRAND_STAGER_HIGH_BLOCK_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES=%q\n' "${B300_GRAND_STAGER_HIGH_PAIR_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES=%q\n' "${B300_GRAND_STAGER_HIGH_BLOCK_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGER_PAIR_POLICY=%q\n' "${B300_GRAND_STAGER_PAIR_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=%q\n' "${B300_GRAND_STAGER_BLOCK_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGER_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGER_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGER_SEARCH_PAIR_POLICIES=%q\n' "$STAGER_PAIR_POLICY_LIST"
  printf 'B300_GRAND_SELECTED_STAGER_SEARCH_BLOCK_POLICIES=%q\n' "$STAGER_BLOCK_POLICY_LIST"
'''+anchor,'selected Stage R')
for marker in ('RUN_STAGER=','STAGER_PAIR_POLICY_LIST=','STAGER_BLOCK_POLICY_LIST=','b300x8-joint-nextself-hybrid8-select-stager.sh','B300_GRAND_SELECTED_STAGER_ACCEPTED','B300_GRAND_STAGER_INTEGRATED'):
    if marker not in s: raise SystemExit('missing generated Stage-R firstpass artifact: '+marker)
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_firstpass_stager=1 selected_schema_compatible=3 single_complete_prime=1')
