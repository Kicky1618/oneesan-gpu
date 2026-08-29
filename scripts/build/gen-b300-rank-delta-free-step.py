#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-rank-delta-free-step.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
old='''__device__ __forceinline__ RankDelta b300_rank_delta_step(MateValue v,int p,int h){
    Code a=0,b=0;
    if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,p,N))a+=D_MAIN_DP[p][h];
    if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,p,R))a+=D_MAIN_DP[p][h-1];
    int q=p-1;
    if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];
    if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];
    return RankDelta(b)-RankDelta(a);
}'''
new='''__device__ __forceinline__ RankDelta b300_rank_delta_step(MateValue v,int p,int h){
    // p is inside the active main window and p-1 is inside the corresponding
    // blocked window, so neither coordinate can be fixed by window_masks().
    // Remove four constant-memory fixed/occupancy checks from every state-step.
    Code a=0,b=0;
    if(v>N)a+=D_MAIN_DP[p][h];
    if(v>R&&h>0)a+=D_MAIN_DP[p][h-1];
    const int q=p-1;
    if(v>N)b+=D_BLOCK_DP[q][h];
    if(v>R&&h>0)b+=D_BLOCK_DP[q][h-1];
    return RankDelta(b)-RankDelta(a);
}'''
n=s.count(old)
if n!=1:raise SystemExit(f'rank-delta free-step anchor expected one match got {n}')
s=s.replace(old,new,1)
if 'allowed(D_MAIN_FIXED,D_MAIN_OCC,p,N)' in s or 'allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N)' in s:
    raise SystemExit('stale moving-position allowed check remains')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: rank_delta_moving_positions_free=1 allowed_checks_per_state_step=0 constant_fixed_loads_removed=1')
