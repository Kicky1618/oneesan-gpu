#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-high-recurrence-fixed-gate.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
old='return (D_MAIN_FIXED&HIGH)==0;'
new='return (D_MAIN_FIXED&HIGH)==0 && __popc(D_MAIN_FIXED)>=7;'
if s.count(old)!=1:raise SystemExit(f'high recurrence gate anchor expected once, got {s.count(old)}')
s=s.replace(old,new,1)
if '__popc(D_MAIN_FIXED)>=7' not in s:raise SystemExit('fixed-bit recurrence gate missing after transform')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: high_recurrence_min_fixed=7 fixed_lt7_fallback=raw_mate_rank signed35_gate=1')
