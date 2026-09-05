#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys
if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-rank-delta-report.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'dMainRankDelta' not in s or 'useRankDelta=' not in s: raise SystemExit('rank-delta report requires rank-delta cache transform')

def once(a,b,label):
    global s
    n=s.count(a)
    if n!=1: raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(a,b,1)

once(
'RankDelta*dMainRankDelta=nullptr,*dBlockRankDelta=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
'RankDelta*dMainRankDelta=nullptr,*dBlockRankDelta=nullptr;uint64_t rankDeltaGroups=0,rankDeltaFallbackGroups=0;PeerInterval*dIM=nullptr,*dID=nullptr;',
'DeviceCtx rank-delta counters')
once(
'bool useRankDelta=useMate&&useBlockMate&&(countBytes+mateBytes+blockMateBytes+rankDeltaBytes<=target);\n    c.ensure(',
'bool useRankDelta=useMate&&useBlockMate&&(countBytes+mateBytes+blockMateBytes+rankDeltaBytes<=target);if(useRankDelta)++c.rankDeltaGroups;else ++c.rankDeltaFallbackGroups;\n    c.ensure(',
'process_group rank-delta counter')
once(
'size_t maxIntervals=0;for(auto&c:ctx){',
'size_t maxIntervals=0;uint64_t rankDeltaGroups=0,rankDeltaFallbackGroups=0;for(auto&c:ctx){rankDeltaGroups+=c.rankDeltaGroups;rankDeltaFallbackGroups+=c.rankDeltaFallbackGroups;',
'rank-delta aggregate counters')
once(
'<<" max_intervals="<<maxIntervals<<" active_max_s="',
'<<" max_intervals="<<maxIntervals<<" rank_delta_groups="<<rankDeltaGroups<<" rank_delta_fallback_groups="<<rankDeltaFallbackGroups<<" active_max_s="',
'rank-delta backend output')
for x in ('rankDeltaGroups+=c.rankDeltaGroups','rank_delta_groups=','rank_delta_fallback_groups='):
    if x not in s: raise SystemExit(f'missing rank-delta report artifact: {x}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: rank_delta_coverage_report=1 per_gpu_counters=1 backend_summary=1')
