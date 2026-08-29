#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-main-recurrence-report.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_high_main_state_active','static void process_group','backend=gridfp-b300-hbm32-forced2window-opt-batch'):
    if req not in s:raise SystemExit(f'main recurrence report requires artifact: {req}')

# Host-side coverage follows exactly the device safety predicate. It is counted
# per process_group invocation (therefore per row/group), which is sufficient to
# distinguish a real recurrent run from a fixed<7 fallback run.
marker='static void process_group(DeviceCtx&c,int W,const WindowPlan&wp,const PreparedGroup&pg,int threads,size_t target){'
if s.count(marker)!=1:raise SystemExit(f'process_group marker expected once got {s.count(marker)}')
insert='''static std::atomic<unsigned long long> B300_HIGH_REC_GROUPS{0};
static std::atomic<unsigned long long> B300_HIGH_REC_FALLBACK_GROUPS{0};

'''
s=s.replace(marker,insert+marker,1)

anchor='''    auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");
    auto const&ms=pg.ms;auto const&ds=pg.ds;if(!ms.size&&!ds.size)return;'''
if s.count(anchor)!=1:raise SystemExit(f'process_group prologue anchor expected once got {s.count(anchor)}')
repl='''    auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");
    auto const&ms=pg.ms;auto const&ds=pg.ds;if(!ms.size&&!ds.size)return;
    if constexpr(TARGET_W==28){
        constexpr uint32_t HIGH=((uint32_t(1)<<28)-1u)^((uint32_t(1)<<13)-1u);
        if((pg.mf&HIGH)==0){
            if(__builtin_popcount(pg.mf)>=7)B300_HIGH_REC_GROUPS.fetch_add(1,std::memory_order_relaxed);
            else B300_HIGH_REC_FALLBACK_GROUPS.fetch_add(1,std::memory_order_relaxed);
        }
    }'''
s=s.replace(anchor,repl,1)

res_anchor='''for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];'''
if s.count(res_anchor)!=1:raise SystemExit(f'residue loop anchor expected once got {s.count(res_anchor)}')
s=s.replace(res_anchor,res_anchor+'B300_HIGH_REC_GROUPS.store(0,std::memory_order_relaxed);B300_HIGH_REC_FALLBACK_GROUPS.store(0,std::memory_order_relaxed);',1)

out_anchor='<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<std::endl;'
if s.count(out_anchor)!=1:raise SystemExit(f'backend output anchor expected once got {s.count(out_anchor)}')
s=s.replace(out_anchor,'<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<" high_rec_groups="<<B300_HIGH_REC_GROUPS.load(std::memory_order_relaxed)<<" high_rec_fallback_groups="<<B300_HIGH_REC_FALLBACK_GROUPS.load(std::memory_order_relaxed)<<std::endl;',1)

for required in ('B300_HIGH_REC_GROUPS','__builtin_popcount(pg.mf)>=7','high_rec_groups=','high_rec_fallback_groups='):
    if required not in s:raise SystemExit(f'missing main recurrence report artifact: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: main_recurrence_report=1 high_symbol_range=13..27 high_min_fixed=7 coverage_unit=process_group high_rec_groups=1 high_rec_fallback_groups=1')
