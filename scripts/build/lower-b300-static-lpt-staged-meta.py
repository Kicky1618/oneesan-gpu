#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PW_OLD='''struct PreparedWindow{
    WindowPlan wp;
    std::vector<PreparedGroup> groups;
};'''
PW_NEW='''struct PreparedWindow{
    WindowPlan wp;
    std::vector<PreparedGroup> groups;
    std::vector<std::vector<int>> by_gpu;
    std::vector<Code> assigned_work;
};'''

SCHEDULE_OLD='''        PreparedWindow pw;pw.wp=wp;pw.groups.reserve(nj);
        for(int g=0;g<nj;++g)pw.groups.push_back(prepare_group(W,pw.wp,g));
        std::sort(pw.groups.begin(),pw.groups.end(),[](auto const&a,auto const&b){return a.work>b.work;});
        schedule.push_back(std::move(pw));hi=wp.p_lo-1;'''
SCHEDULE_NEW='''        PreparedWindow pw;pw.wp=wp;pw.groups.reserve(nj);
        for(int g=0;g<nj;++g)pw.groups.push_back(prepare_group(W,pw.wp,g));
        std::sort(pw.groups.begin(),pw.groups.end(),[](auto const&a,auto const&b){return a.work>b.work;});
        pw.by_gpu.resize(ng);pw.assigned_work.assign(ng,0);
        for(int q=0;q<nj;++q){int d=0;for(int x=1;x<ng;++x)if(pw.assigned_work[x]<pw.assigned_work[d])d=x;pw.by_gpu[d].push_back(q);pw.assigned_work[d]+=pw.groups[q].work;}
        schedule.push_back(std::move(pw));hi=wp.p_lo-1;'''

STAGE_OLD='''    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();
    std::vector<DeviceGroupMeta> staged_group_meta;staged_group_meta.reserve(staged_group_count);
    for(auto&pw:schedule)for(auto&pg:pw.groups){pg.meta_id=staged_group_meta.size();DeviceGroupMeta m{};std::memcpy(m.main_dp,pg.ms.dp,sizeof(pg.ms.dp));std::memcpy(m.block_dp,pg.ds.dp,sizeof(pg.ds.dp));m.main_fixed=pg.mf;m.main_occ=pg.mo;m.block_fixed=pg.bf;m.block_occ=pg.bo;staged_group_meta.push_back(m);}
    size_t staged_group_meta_bytes=staged_group_meta.size()*sizeof(DeviceGroupMeta);
    unsigned long long staged_group_meta_max_mib=512;if(const char*e=std::getenv("B300_STAGED_META_MAX_MIB")){char*end=nullptr;unsigned long long v=std::strtoull(e,&end,10);if(!end||*end||v<1){std::cerr<<"invalid B300_STAGED_META_MAX_MIB="<<e<<"\\n";return 13;}staged_group_meta_max_mib=v;}
    if(staged_group_meta_max_mib>std::numeric_limits<size_t>::max()/(1ull<<20)||staged_group_meta_bytes>size_t(staged_group_meta_max_mib)*(1ull<<20)){std::cerr<<"staged group meta exceeds cap: bytes_per_gpu="<<staged_group_meta_bytes<<" cap_mib="<<staged_group_meta_max_mib<<" groups="<<staged_group_count<<"\\n";return 13;}
    if(staged_group_meta_bytes>reserve){std::cerr<<"staged group meta exceeds HBM reserve: bytes_per_gpu="<<staged_group_meta_bytes<<" reserve_mib="<<reserve_mib<<"; increase GRIDFP_VRAM_RESERVE_MIB or reduce staged metadata\\n";return 13;}
    for(auto&c:ctx)c.stage_group_meta(staged_group_meta);
    std::cerr<<"staged group meta: groups="<<staged_group_count<<" bytes_per_gpu="<<staged_group_meta_bytes<<" mib_per_gpu="<<double(staged_group_meta_bytes)/(1<<20)<<" cap_mib="<<staged_group_meta_max_mib<<" reserve_mib="<<reserve_mib<<" total_h2d_gib="<<double(staged_group_meta_bytes)*ng/(1ull<<30)<<" copy_mode=H2D_once_then_D2D_per_group\\n";
    staged_group_meta.clear();staged_group_meta.shrink_to_fit();'''
STAGE_NEW='''    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();
    std::vector<std::vector<DeviceGroupMeta>> staged_group_meta(ng);
    std::vector<Code> static_lpt_work(ng,0);std::vector<size_t> static_lpt_groups(ng,0);
    for(auto&pw:schedule)for(int d=0;d<ng;++d)for(int q:pw.by_gpu[d]){auto&pg=pw.groups[q];pg.meta_id=staged_group_meta[d].size();DeviceGroupMeta m{};std::memcpy(m.main_dp,pg.ms.dp,sizeof(pg.ms.dp));std::memcpy(m.block_dp,pg.ds.dp,sizeof(pg.ds.dp));m.main_fixed=pg.mf;m.main_occ=pg.mo;m.block_fixed=pg.bf;m.block_occ=pg.bo;staged_group_meta[d].push_back(m);static_lpt_work[d]+=pg.work;++static_lpt_groups[d];}
    size_t staged_group_meta_total_bytes=0,staged_group_meta_max_bytes=0;size_t static_lpt_group_min=static_lpt_groups[0],static_lpt_group_max=static_lpt_groups[0];Code static_lpt_work_min=static_lpt_work[0],static_lpt_work_max=static_lpt_work[0],static_lpt_work_sum=0;
    for(int d=0;d<ng;++d){size_t b=staged_group_meta[d].size()*sizeof(DeviceGroupMeta);staged_group_meta_total_bytes+=b;staged_group_meta_max_bytes=std::max(staged_group_meta_max_bytes,b);static_lpt_group_min=std::min(static_lpt_group_min,static_lpt_groups[d]);static_lpt_group_max=std::max(static_lpt_group_max,static_lpt_groups[d]);static_lpt_work_min=std::min(static_lpt_work_min,static_lpt_work[d]);static_lpt_work_max=std::max(static_lpt_work_max,static_lpt_work[d]);static_lpt_work_sum+=static_lpt_work[d];}
    unsigned long long staged_group_meta_max_mib=512;if(const char*e=std::getenv("B300_STAGED_META_MAX_MIB")){char*end=nullptr;unsigned long long v=std::strtoull(e,&end,10);if(!end||*end||v<1){std::cerr<<"invalid B300_STAGED_META_MAX_MIB="<<e<<"\\n";return 13;}staged_group_meta_max_mib=v;}
    if(staged_group_meta_max_mib>std::numeric_limits<size_t>::max()/(1ull<<20)||staged_group_meta_max_bytes>size_t(staged_group_meta_max_mib)*(1ull<<20)){std::cerr<<"static LPT staged group meta exceeds cap: max_bytes_per_gpu="<<staged_group_meta_max_bytes<<" cap_mib="<<staged_group_meta_max_mib<<" groups="<<staged_group_count<<"\\n";return 13;}
    if(staged_group_meta_max_bytes>reserve){std::cerr<<"static LPT staged group meta exceeds HBM reserve: max_bytes_per_gpu="<<staged_group_meta_max_bytes<<" reserve_mib="<<reserve_mib<<"; increase GRIDFP_VRAM_RESERVE_MIB or reduce staged metadata\\n";return 13;}
    if(W==28&&ng==8&&effective_target_mib==16384&&max_window==14&&(staged_group_count!=16384||static_lpt_group_min!=2044||static_lpt_group_max!=2052||staged_group_meta_total_bytes!=228327424ull||staged_group_meta_max_bytes!=28596672ull)){std::cerr<<"static LPT default-plan proof mismatch: groups="<<staged_group_count<<" group_min="<<static_lpt_group_min<<" group_max="<<static_lpt_group_max<<" total_bytes="<<staged_group_meta_total_bytes<<" max_bytes="<<staged_group_meta_max_bytes<<"\\n";return 14;}
    for(int d=0;d<ng;++d)ctx[d].stage_group_meta(staged_group_meta[d]);
    double static_lpt_work_avg=double(static_lpt_work_sum)/ng;double static_lpt_spread_pct=static_lpt_work_avg?100.0*double(static_lpt_work_max-static_lpt_work_min)/static_lpt_work_avg:0.0;
    std::cerr<<"static LPT staged group meta: groups="<<staged_group_count<<" group_min="<<static_lpt_group_min<<" group_max="<<static_lpt_group_max<<" max_bytes_per_gpu="<<staged_group_meta_max_bytes<<" max_mib_per_gpu="<<double(staged_group_meta_max_bytes)/(1<<20)<<" cap_mib="<<staged_group_meta_max_mib<<" reserve_mib="<<reserve_mib<<" total_h2d_gib="<<double(staged_group_meta_total_bytes)/(1ull<<30)<<" work_min="<<static_lpt_work_min<<" work_max="<<static_lpt_work_max<<" work_spread_pct="<<static_lpt_spread_pct<<" copy_mode=H2D_once_local_then_D2D_per_group scheduler=static_lpt\\n";
    for(auto&v:staged_group_meta){v.clear();v.shrink_to_fit();}'''

WORKER_OLD='''        for(auto const&pw:schedule){
            int nj=(int)pw.groups.size();std::atomic<int>next{0};std::vector<std::thread>ths;ths.reserve(ng);
            for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(;;){int q=next.fetch_add(1,std::memory_order_relaxed);if(q>=nj)break;process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);}});
            for(auto&t:ths)t.join();++done_windows;
        }'''
WORKER_NEW='''        for(auto const&pw:schedule){
            std::vector<std::thread>ths;ths.reserve(ng);
            for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);});
            for(auto&t:ths)t.join();++done_windows;
        }'''


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1:
        raise SystemExit(f'{label}: expected exactly one staged-basearg match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,PW_OLD,PW_NEW,'PreparedWindow static LPT lists')
    text=once(text,SCHEDULE_OLD,SCHEDULE_NEW,'window LPT assignment')
    text=once(text,STAGE_OLD,STAGE_NEW,'per-GPU staged metadata')
    text=once(text,WORKER_OLD,WORKER_NEW,'static LPT worker loop')
    for stale in ('std::atomic<int>next{0}','next.fetch_add(1,std::memory_order_relaxed)','for(auto&c:ctx)c.stage_group_meta(staged_group_meta);','copy_mode=H2D_once_then_D2D_per_group'):
        if stale in text: raise SystemExit(f'dynamic/replicated staged artifact remains: {stale}')
    for required in ('std::vector<std::vector<int>> by_gpu','pw.by_gpu.resize(ng)','pw.assigned_work.assign(ng,0)','for(int q:pw.by_gpu[d])','ctx[d].stage_group_meta(staged_group_meta[d])','scheduler=static_lpt','copy_mode=H2D_once_local_then_D2D_per_group','static LPT default-plan proof mismatch:'):
        if required not in text: raise SystemExit(f'missing static LPT artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} scheduler=static_lpt work_stealing=0 metadata_replication=0 per_gpu_local_meta_ids=1 staged_h2d_once_local=1 per_group_constant_copy=d2d_sync default_expected_group_min=2044 default_expected_group_max=2052 default_expected_max_mib_per_gpu=27.271911621 default_expected_total_h2d_gib=0.212646484375')

if __name__=='__main__':main()
