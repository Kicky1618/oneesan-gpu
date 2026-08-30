#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-firstpass-staget.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
if 'B300_GRAND_SELECTED_STAGES_ACCEPTED' not in s or 'RUN_STAGES=' not in s:
    raise SystemExit('Stage T firstpass overlay requires Stage-S-aware firstpass')
if 'B300_GRAND_SELECTED_STAGET_ACCEPTED' in s or 'RUN_STAGET=' in s:
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

after_line('RUN_STAGES=', 'RUN_STAGET="${RUN_STAGET:-1}"','RUN_STAGET')
after_line('STAGES_BLOCK_L2_LIST="${STAGES_BLOCK_L2_LIST:-0 64 128 256}"', '''STAGET_MIN_SPEEDUP="${STAGET_MIN_SPEEDUP:-1.002}"
STAGET_POLICY_LIST="${STAGET_POLICY_LIST:-default cg cs}"
python3 - "$STAGET_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('STAGET_MIN_SPEEDUP must be >=1')
PY''','Stage-T knobs')
# Boolean validation is a single generated loop in the Stage-S firstpass.
old='RUN_STAGER RUN_STAGES; do'; rep(old,'RUN_STAGER RUN_STAGES RUN_STAGET; do','boolean validation')
after_line('STAGES_BLOCK_L2_LIST="$(normalize_l2_sizes ', '''STAGET_POLICY_LIST="$(normalize_load_policies "$STAGET_POLICY_LIST")"
case " $STAGET_POLICY_LIST " in *' default '*) ;; *) echo 'STAGET_POLICY_LIST must include default baseline' >&2; exit 2;; esac''','Stage-T normalization')
# Meta/provenance input log.
lines=s.splitlines(True); hits=[i for i,x in enumerate(lines) if 'stages_block_l2_list=%s' in x]
if len(hits)!=1: raise SystemExit(f'Stage-T meta anchor expected one got {len(hits)}')
lines[hits[0]+1:hits[0]+1]=["  printf 'run_staget=%s\\n' \"$RUN_STAGET\"\n  printf 'staget_min_speedup=%s\\n' \"$STAGET_MIN_SPEEDUP\"\n  printf 'staget_policy_list=%s\\n' \"$STAGET_POLICY_LIST\"\n"]; s=''.join(lines)
# GPU-free contracts before any selector GPU work.
anchor='bash "$ONEESAN_ROOT/scripts/bench/b300-grand-stages-contract-preflight.sh"\n'
rep(anchor,anchor+'bash "$ONEESAN_ROOT/scripts/bench/b300-staget-preflight.sh"\nbash "$ONEESAN_ROOT/scripts/bench/b300-staget-promotion-preflight.sh"\nbash "$ONEESAN_ROOT/scripts/bench/b300-grand-staget-contract-preflight.sh"\n','Stage-T preflights')
# Banner is informational; add T policy axis after the existing Stage-S lists.
rep(' s_pair_l2=[$STAGES_PAIR_L2_LIST] s_block_l2=[$STAGES_BLOCK_L2_LIST] ===',' s_pair_l2=[$STAGES_PAIR_L2_LIST] s_block_l2=[$STAGES_BLOCK_L2_LIST] t_mate=[$STAGET_POLICY_LIST] ===','Stage-T banner')
# Pass T controls to selector.
rep('RUN_STAGER="$RUN_STAGER" RUN_STAGES="$RUN_STAGES"','RUN_STAGER="$RUN_STAGER" RUN_STAGES="$RUN_STAGES" RUN_STAGET="$RUN_STAGET"','selector run flag')
rep('STAGER_MIN_SPEEDUP="$STAGER_MIN_SPEEDUP" STAGES_MIN_SPEEDUP="$STAGES_MIN_SPEEDUP"','STAGER_MIN_SPEEDUP="$STAGER_MIN_SPEEDUP" STAGES_MIN_SPEEDUP="$STAGES_MIN_SPEEDUP" STAGET_MIN_SPEEDUP="$STAGET_MIN_SPEEDUP"','selector speed flag')
rep('STAGES_PAIR_L2_LIST="$STAGES_PAIR_L2_LIST" STAGES_BLOCK_L2_LIST="$STAGES_BLOCK_L2_LIST"','STAGES_PAIR_L2_LIST="$STAGES_PAIR_L2_LIST" STAGES_BLOCK_L2_LIST="$STAGES_BLOCK_L2_LIST" STAGET_POLICY_LIST="$STAGET_POLICY_LIST"','selector policy flag')
rep('b300x8-joint-nextself-hybrid8-select-stages.sh','b300x8-joint-nextself-hybrid8-select-staget.sh','selector entrypoint')
# Require Stage-T summary provenance in addition to all prior stages.
rep('"${B300_GRAND_STAGES_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1','"${B300_GRAND_STAGES_INTEGRATED:-0}" == 1 && "${B300_GRAND_STAGET_INTEGRATED:-0}" == 1 && \\\n   "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1','summary provenance')
s=s.replace('grand Stage-I/J/K/L/M/N/O/P/Q/R/S single-race provenance markers missing','grand Stage-I/J/K/L/M/N/O/P/Q/R/S/T single-race provenance markers missing')
# Selected-env schema remains v3. Insert T immediately before the stable complete-prime marker.
anchor="  printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1\\n'\n"
selected='''  printf 'B300_GRAND_SELECTED_STAGET_ENABLED=%q\n' "$RUN_STAGET"
  printf 'B300_GRAND_SELECTED_STAGET_MIN_SPEEDUP=%q\n' "$STAGET_MIN_SPEEDUP"
  printf 'B300_GRAND_SELECTED_STAGET_ACCEPTED=%q\n' "${B300_GRAND_STAGET_OK:-0}"
  printf 'B300_GRAND_SELECTED_STAGET_UPSTREAM_KIND=%q\n' "${B300_GRAND_STAGET_UPSTREAM_KIND:-}"
  printf 'B300_GRAND_SELECTED_STAGET_STAGER_UPSTREAM_KIND=%q\n' "${B300_GRAND_STAGET_STAGER_UPSTREAM_KIND:-}"
  printf 'B300_GRAND_SELECTED_STAGET_LOW_PAIR_POLICY=%q\n' "${B300_GRAND_STAGET_LOW_PAIR_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGET_LOW_BLOCK_POLICY=%q\n' "${B300_GRAND_STAGET_LOW_BLOCK_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGET_LOW_PAIR_L2_BYTES=%q\n' "${B300_GRAND_STAGET_LOW_PAIR_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGET_LOW_BLOCK_L2_BYTES=%q\n' "${B300_GRAND_STAGET_LOW_BLOCK_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGET_HIGH_PAIR_POLICY=%q\n' "${B300_GRAND_STAGET_HIGH_PAIR_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_POLICY=%q\n' "${B300_GRAND_STAGET_HIGH_BLOCK_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGET_HIGH_PAIR_L2_BYTES=%q\n' "${B300_GRAND_STAGET_HIGH_PAIR_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_L2_BYTES=%q\n' "${B300_GRAND_STAGET_HIGH_BLOCK_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGET_HIGH_MATE_POLICY=%q\n' "${B300_GRAND_STAGET_HIGH_MATE_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGET_HIGH_MATE_L2_BYTES=%q\n' "${B300_GRAND_STAGET_HIGH_MATE_L2_BYTES:-0}"
  printf 'B300_GRAND_SELECTED_STAGET_POLICY=%q\n' "${B300_GRAND_STAGET_POLICY:-default}"
  printf 'B300_GRAND_SELECTED_STAGET_STAGED_SPEEDUP=%q\n' "${B300_GRAND_STAGET_STAGED_SPEEDUP:-1.0}"
  printf 'B300_GRAND_SELECTED_STAGET_SEARCH_POLICIES=%q\n' "$STAGET_POLICY_LIST"
'''
rep(anchor,selected+anchor,'selected Stage T')
for marker in ('RUN_STAGET=','STAGET_POLICY_LIST=','b300x8-joint-nextself-hybrid8-select-staget.sh','B300_GRAND_SELECTED_STAGET_ACCEPTED','B300_GRAND_SELECTED_STAGET_POLICY','B300_GRAND_STAGET_INTEGRATED'):
    if marker not in s: raise SystemExit('missing generated Stage-T firstpass artifact: '+marker)
if s.count('B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1')!=1: raise SystemExit('selected complete-prime marker drift')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_grand_firstpass_staget=1 selected_schema_compatible=3 priority=T>S>R single_complete_prime=1')
