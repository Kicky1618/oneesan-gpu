#!/usr/bin/env python3
from __future__ import annotations
import pathlib
base=pathlib.Path(__file__).with_name('gen-b300-grand-firstpass-staget.py')
s=base.read_text()
old="after_line('STAGES_BLOCK_L2_LIST=', '''STAGET_MIN_SPEEDUP="
fixed="after_line('STAGES_BLOCK_L2_LIST=\"${STAGES_BLOCK_L2_LIST:-0 64 128 256}\"', '''STAGET_MIN_SPEEDUP="
if old in s:
    if s.count(old)!=1: raise SystemExit(f'Stage-T firstpass compat ambiguous old anchor count={s.count(old)}')
    s=s.replace(old,fixed,1)
elif fixed not in s:
    raise SystemExit('Stage-T firstpass compat found neither old nor fixed default-L2 anchor')
exec(compile(s,str(base),'exec'),{'__name__':'__main__','__file__':str(base)})
