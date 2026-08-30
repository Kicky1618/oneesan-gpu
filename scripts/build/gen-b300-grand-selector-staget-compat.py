#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
base=pathlib.Path(__file__).with_name('gen-b300-grand-selector-staget.py')
s=base.read_text()
old="after_line('STAGES_BLOCK_L2_LIST=', '''STAGET_MIN_SPEEDUP="
new="after_line('STAGES_BLOCK_L2_LIST=\"${STAGES_BLOCK_L2_LIST:-0 64 128 256}\"', '''STAGET_MIN_SPEEDUP="
if s.count(old)!=1:
    raise SystemExit(f'Stage-T compat default-L2 anchor expected one got {s.count(old)}')
s=s.replace(old,new,1)
code=compile(s,str(base), 'exec')
exec(code,{'__name__':'__main__','__file__':str(base)})
