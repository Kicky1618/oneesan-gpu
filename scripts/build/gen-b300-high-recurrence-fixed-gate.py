#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-high-recurrence-fixed-gate.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
old='return (D_MAIN_FIXED&HIGH)==0;'
new='return (D_MAIN_FIXED&HIGH)==0 && __popc(D_MAIN_FIXED)>=7;'
if s.count(old)!=1:raise SystemExit(f'high recurrence gate anchor expected once, got {s.count(old)}')
s=s.replace(old,new,1)

process='static void process_group(DeviceCtx&c,int W,const WindowPlan&wp,const PreparedGroup&pg,int threads,size_t target){'
if s.count(process)!=1:raise SystemExit(f'process_group marker expected once, got {s.count(process)}')
s=s.replace(process,'''static std::atomic<unsigned long long> B300_HIGH_REC_GROUPS{0};
static std::atomic<unsigned long long> B300_HIGH_REC_FALLBACK_GROUPS{0};

'''+process,1)

prologue='auto const&ms=pg.ms;auto const&ds=pg.ds;if(!ms.size&&!ds.size)return;'
if s.count(prologue)!=1:raise SystemExit(f'process_group prologue expected once, got {s.count(prologue)}')
s=s.replace(prologue,prologue+'''
    if constexpr(TARGET_W==28){
        constexpr uint32_t REC_HIGH=((uint32_t(1)<<28)-1u)^((uint32_t(1)<<13)-1u);
        if((pg.mf&REC_HIGH)==0){
            if(__builtin_popcount(pg.mf)>=7)B300_HIGH_REC_GROUPS.fetch_add(1,std::memory_order_relaxed);
            else B300_HIGH_REC_FALLBACK_GROUPS.fetch_add(1,std::memory_order_relaxed);
        }
    }''',1)

resloop='for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];'
if s.count(resloop)!=1:raise SystemExit(f'residue loop expected once, got {s.count(resloop)}')
s=s.replace(resloop,resloop+'B300_HIGH_REC_GROUPS.store(0,std::memory_order_relaxed);B300_HIGH_REC_FALLBACK_GROUPS.store(0,std::memory_order_relaxed);',1)

result='<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<std::endl;'
if s.count(result)!=1:raise SystemExit(f'backend result anchor expected once, got {s.count(result)}')
s=s.replace(result,'<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<" high_rec_groups="<<B300_HIGH_REC_GROUPS.load(std::memory_order_relaxed)<<" high_rec_fallback_groups="<<B300_HIGH_REC_FALLBACK_GROUPS.load(std::memory_order_relaxed)<<std::endl;',1)

for required in ('__popc(D_MAIN_FIXED)>=7','__builtin_popcount(pg.mf)>=7','high_rec_groups=','high_rec_fallback_groups='):
    if required not in s:raise SystemExit(f'fixed-bit recurrence artifact missing: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: high_recurrence_min_fixed=7 fixed_lt7_fallback=raw_mate_rank signed35_gate=1 runtime_coverage_report=1 coverage_unit=process_group')
