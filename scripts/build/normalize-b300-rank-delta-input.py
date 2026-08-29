#!/usr/bin/env python3
from __future__ import annotations
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: normalize-b300-rank-delta-input.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()

old='''size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);if(!useMate){std::cerr<<"planner regression: main MateID cache does not fit selected group bytes="<<(countBytes+mateBytes)<<" target="<<target<<"\\n";std::exit(19);}
    size_t blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useBlockMate=(countBytes+mateBytes+blockMateBytes<=target);if(!useBlockMate){std::cerr<<"planner regression: blocked MateID cache does not fit selected group bytes="<<(countBytes+mateBytes+blockMateBytes)<<" target="<<target<<"\\n";std::exit(20);}
    c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size());'''
new='''size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID),blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);bool useBlockMate=(countBytes+(useMate?mateBytes:0)+blockMateBytes<=target);
    c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size());'''

n=s.count(old)
if n!=1:
    raise SystemExit(f'rank-delta sizing compatibility anchor expected one match got {n}')
s=s.replace(old,new,1)
for required in ('blockMateBytes=size_t(ds.size)*sizeof(MateID)','c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size())'):
    if required not in s:raise SystemExit(f'missing normalized artifact: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'normalized {out} from {src}: rank_delta_input_compat=1 planner_asserts_consumed_by_rank_delta_transform=1')
