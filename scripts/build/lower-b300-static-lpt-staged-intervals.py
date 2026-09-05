#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

CTX_OLD='int dev=-1;Count*authMain=nullptr,*authBlock=nullptr;DeviceGroupMeta*dGroupMeta=nullptr;size_t groupMetaCount=0;uint8_t*arena=nullptr;'
CTX_NEW='int dev=-1;Count*authMain=nullptr,*authBlock=nullptr;DeviceGroupMeta*dGroupMeta=nullptr;size_t groupMetaCount=0;PeerInterval*dStageIntervals=nullptr;size_t stageIntervalCount=0;uint8_t*arena=nullptr;'

ENSURE_OLD='''    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){'''
ENSURE_NEW='''    void stage_intervals(const std::vector<PeerInterval>& h){
        ck(cudaSetDevice(dev),"set stage intervals");
        if(dStageIntervals){cudaFree(dStageIntervals);dStageIntervals=nullptr;stageIntervalCount=0;}
        stageIntervalCount=h.size();
        if(!h.empty()){ck(cudaMalloc(&dStageIntervals,h.size()*sizeof(PeerInterval)),"stage intervals alloc");ck(cudaMemcpy(dStageIntervals,h.data(),h.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"stage intervals H2D");}
    }
    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){'''

DESTROY_OLD='if(dGroupMeta)cudaFree(dGroupMeta);if(arena)cudaFree(arena);if(dIM)cudaFree(dIM);if(dID)cudaFree(dID);'
DESTROY_NEW='if(dGroupMeta)cudaFree(dGroupMeta);if(dStageIntervals)cudaFree(dStageIntervals);if(arena)cudaFree(arena);if(dIM)cudaFree(dIM);if(dID)cudaFree(dID);'

PG_OLD='int g=0;size_t meta_id=0;'
PG_NEW='int g=0;size_t meta_id=0;size_t mi_stage_off=0,mi_stage_count=0,di_stage_off=0,di_stage_count=0;'

COPY_OLD='''    c.ensure(ms.size,ds.size,useMate,pg.mi.size(),pg.di.size());
    if(!pg.mi.empty())ck(cudaMemcpy(c.dIM,pg.mi.data(),pg.mi.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"copy main intervals");
    if(!pg.di.empty())ck(cudaMemcpy(c.dID,pg.di.data(),pg.di.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"copy block intervals");'''
COPY_NEW='''    c.ensure(ms.size,ds.size,useMate,0,0);
    c.maxIntervals=std::max({c.maxIntervals,pg.mi_stage_count,pg.di_stage_count});
    if(pg.mi_stage_off+pg.mi_stage_count>c.stageIntervalCount||pg.di_stage_off+pg.di_stage_count>c.stageIntervalCount){std::cerr<<"staged interval range out of bounds gpu="<<c.dev<<" mi="<<pg.mi_stage_off<<"+"<<pg.mi_stage_count<<" di="<<pg.di_stage_off<<"+"<<pg.di_stage_count<<" total="<<c.stageIntervalCount<<"\\n";std::exit(16);}
    const PeerInterval*dmi=pg.mi_stage_count?c.dStageIntervals+pg.mi_stage_off:nullptr;
    const PeerInterval*ddi=pg.di_stage_count?c.dStageIntervals+pg.di_stage_off:nullptr;'''

STAGE_ANCHOR='''    for(int d=0;d<ng;++d)ctx[d].stage_group_meta(staged_group_meta[d]);
    double static_lpt_work_avg=double(static_lpt_work_sum)/ng;'''
STAGE_REPL='''    for(int d=0;d<ng;++d)ctx[d].stage_group_meta(staged_group_meta[d]);
    std::vector<std::vector<PeerInterval>> staged_intervals(ng);size_t staged_interval_total_count=0,staged_interval_max_count=0,combined_stage_max_bytes=0;
    for(auto&pw:schedule)for(int d=0;d<ng;++d)for(int q:pw.by_gpu[d]){auto&pg=pw.groups[q];pg.mi_stage_off=staged_intervals[d].size();pg.mi_stage_count=pg.mi.size();staged_intervals[d].insert(staged_intervals[d].end(),pg.mi.begin(),pg.mi.end());pg.di_stage_off=staged_intervals[d].size();pg.di_stage_count=pg.di.size();staged_intervals[d].insert(staged_intervals[d].end(),pg.di.begin(),pg.di.end());std::vector<PeerInterval>().swap(pg.mi);std::vector<PeerInterval>().swap(pg.di);}
    for(int d=0;d<ng;++d){staged_interval_total_count+=staged_intervals[d].size();staged_interval_max_count=std::max(staged_interval_max_count,staged_intervals[d].size());combined_stage_max_bytes=std::max(combined_stage_max_bytes,staged_intervals[d].size()*sizeof(PeerInterval)+staged_group_meta[d].size()*sizeof(DeviceGroupMeta));}
    unsigned long long staged_interval_max_mib=256;if(const char*e=std::getenv("B300_STAGED_INTERVAL_MAX_MIB")){char*end=nullptr;unsigned long long v=std::strtoull(e,&end,10);if(!end||*end||v<1){std::cerr<<"invalid B300_STAGED_INTERVAL_MAX_MIB="<<e<<"\\n";return 16;}staged_interval_max_mib=v;}
    size_t staged_interval_max_bytes=staged_interval_max_count*sizeof(PeerInterval);size_t staged_interval_total_bytes=staged_interval_total_count*sizeof(PeerInterval);
    if(staged_interval_max_mib>std::numeric_limits<size_t>::max()/(1ull<<20)||staged_interval_max_bytes>size_t(staged_interval_max_mib)*(1ull<<20)){std::cerr<<"static LPT staged intervals exceed cap: max_bytes_per_gpu="<<staged_interval_max_bytes<<" cap_mib="<<staged_interval_max_mib<<" descriptors="<<staged_interval_total_count<<"\\n";return 16;}
    if(combined_stage_max_bytes>reserve){std::cerr<<"static LPT metadata+interval stage exceeds HBM reserve: max_bytes_per_gpu="<<combined_stage_max_bytes<<" reserve_mib="<<reserve_mib<<"\\n";return 16;}
    if(W==28&&ng==8&&effective_target_mib==16384&&max_window==14&&(staged_interval_total_count!=8453518ull||staged_interval_max_count!=1057352ull||staged_interval_total_bytes!=202884432ull||staged_interval_max_bytes!=25376448ull||combined_stage_max_bytes!=53961480ull)){std::cerr<<"static LPT default interval-stage proof mismatch: descriptors="<<staged_interval_total_count<<" max_descriptors="<<staged_interval_max_count<<" total_bytes="<<staged_interval_total_bytes<<" max_bytes="<<staged_interval_max_bytes<<" combined_max_bytes="<<combined_stage_max_bytes<<"\\n";return 17;}
    for(int d=0;d<ng;++d)ctx[d].stage_intervals(staged_intervals[d]);
    std::cerr<<"static LPT staged intervals: descriptors="<<staged_interval_total_count<<" max_descriptors_per_gpu="<<staged_interval_max_count<<" total_h2d_gib="<<double(staged_interval_total_bytes)/(1ull<<30)<<" max_mib_per_gpu="<<double(staged_interval_max_bytes)/(1<<20)<<" cap_mib="<<staged_interval_max_mib<<" combined_stage_max_mib_per_gpu="<<double(combined_stage_max_bytes)/(1<<20)<<" old_repeated_h2d_gib_full_rows="<<double(staged_interval_total_bytes)*W/(1ull<<30)<<" copy_mode=H2D_once_local_then_zero_interval_copy_per_group scheduler=static_lpt\\n";
    for(auto&v:staged_intervals){v.clear();v.shrink_to_fit();}
    double static_lpt_work_avg=double(static_lpt_work_sum)/ng;'''

LAUNCHES=(
('interval_io_kernel<false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size(),c.authMain)', 'interval_io_kernel<false><<<interval_blocks(pg.mi_stage_count,threads),threads>>>(c.dA,dmi,pg.mi_stage_count,c.authMain)'),
('interval_io_kernel<false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size(),c.authBlock)', 'interval_io_kernel<false><<<interval_blocks(pg.di_stage_count,threads),threads>>>(c.dD,ddi,pg.di_stage_count,c.authBlock)'),
('interval_io_kernel<true><<<interval_blocks(pg.mi.size(),threads),threads>>>(cur,c.dIM,pg.mi.size(),c.authMain)', 'interval_io_kernel<true><<<interval_blocks(pg.mi_stage_count,threads),threads>>>(cur,dmi,pg.mi_stage_count,c.authMain)'),
('interval_io_kernel<true><<<interval_blocks(pg.di.size(),threads),threads>>>(dcur,c.dID,pg.di.size(),c.authBlock)', 'interval_io_kernel<true><<<interval_blocks(pg.di_stage_count,threads),threads>>>(dcur,ddi,pg.di_stage_count,c.authBlock)'),
)

def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1:raise SystemExit(f'{label}: expected exactly one static-LPT match, got {n}')
    return text.replace(old,new,1)

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();text=a.src.read_text()
    text=once(text,CTX_OLD,CTX_NEW,'DeviceCtx staged interval fields')
    text=once(text,ENSURE_OLD,ENSURE_NEW,'DeviceCtx interval stage method')
    text=once(text,DESTROY_OLD,DESTROY_NEW,'DeviceCtx staged interval cleanup')
    text=once(text,PG_OLD,PG_NEW,'PreparedGroup staged interval ranges')
    text=once(text,COPY_OLD,COPY_NEW,'remove per-group interval H2D')
    text=once(text,STAGE_ANCHOR,STAGE_REPL,'flatten and stage intervals')
    for old,new in LAUNCHES:text=once(text,old,new,'staged interval launch')
    for stale in ('cudaMemcpy(c.dIM,pg.mi.data()','cudaMemcpy(c.dID,pg.di.data()','interval_blocks(pg.mi.size(),threads)','interval_blocks(pg.di.size(),threads)'):
        if stale in text:raise SystemExit(f'per-group interval-copy artifact remains: {stale}')
    for required in ('PeerInterval*dStageIntervals=nullptr','void stage_intervals(const std::vector<PeerInterval>& h)','mi_stage_off=0,mi_stage_count=0','const PeerInterval*dmi=','B300_STAGED_INTERVAL_MAX_MIB','staged_interval_total_count!=8453518ull','staged_interval_max_count!=1057352ull','combined_stage_max_bytes!=53961480ull','copy_mode=H2D_once_local_then_zero_interval_copy_per_group','c.maxIntervals=std::max({c.maxIntervals,pg.mi_stage_count,pg.di_stage_count})'):
        if required not in text:raise SystemExit(f'missing staged interval artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} staged_intervals=1 per_group_interval_h2d=0 descriptor_bytes=24 expected_default_descriptors=8453518 expected_default_total_h2d_gib=0.188950851560 expected_old_repeated_h2d_gib=5.290623843670 expected_h2d_reduction=28x expected_combined_stage_max_mib_per_gpu=51.461677551')

if __name__=='__main__':main()
