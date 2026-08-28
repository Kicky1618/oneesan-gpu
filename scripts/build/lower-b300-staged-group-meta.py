#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

CTX_FIELD_OLD='int dev=-1;Count*authMain=nullptr,*authBlock=nullptr;uint8_t*arena=nullptr;'
CTX_FIELD_NEW='int dev=-1;Count*authMain=nullptr,*authBlock=nullptr;DeviceGroupMeta*dGroupMeta=nullptr;size_t groupMetaCount=0;uint8_t*arena=nullptr;'

ENSURE_OLD='''    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){'''
ENSURE_NEW='''    void stage_group_meta(const std::vector<DeviceGroupMeta>& h){
        ck(cudaSetDevice(dev),"set stage group meta");
        if(dGroupMeta){cudaFree(dGroupMeta);dGroupMeta=nullptr;groupMetaCount=0;}
        groupMetaCount=h.size();
        if(!h.empty()){ck(cudaMalloc(&dGroupMeta,h.size()*sizeof(DeviceGroupMeta)),"stage group meta alloc");ck(cudaMemcpy(dGroupMeta,h.data(),h.size()*sizeof(DeviceGroupMeta),cudaMemcpyHostToDevice),"stage group meta H2D");}
    }
    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){'''

DESTROY_OLD='''if(arena)cudaFree(arena);if(dIM)cudaFree(dIM);if(dID)cudaFree(dID);'''
DESTROY_NEW='''if(dGroupMeta)cudaFree(dGroupMeta);if(arena)cudaFree(arena);if(dIM)cudaFree(dIM);if(dID)cudaFree(dID);'''

PG_OLD='''struct PreparedGroup{
    int g=0;'''
PG_NEW='''struct PreparedGroup{
    int g=0;size_t meta_id=0;'''

COPY_OLD='''DeviceGroupMeta hmeta{};
    std::memcpy(hmeta.main_dp,ms.dp,sizeof(ms.dp));std::memcpy(hmeta.block_dp,ds.dp,sizeof(ds.dp));
    hmeta.main_fixed=pg.mf;hmeta.main_occ=pg.mo;hmeta.block_fixed=pg.bf;hmeta.block_occ=pg.bo;
    ck(cudaMemcpyToSymbol(D_GROUP_META,&hmeta,sizeof(hmeta)),"packed group meta");'''
COPY_NEW='''if(pg.meta_id>=c.groupMetaCount){std::cerr<<"staged group meta id out of range "<<pg.meta_id<<"/"<<c.groupMetaCount<<"\n";std::exit(12);}
    ck(cudaMemcpyToSymbol(D_GROUP_META,c.dGroupMeta+pg.meta_id,sizeof(DeviceGroupMeta),0,cudaMemcpyDeviceToDevice),"staged group meta D2D");'''

PREP_END_OLD='''    double prepare_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-prep0).count();
    std::cerr<<"prepared windows="<<schedule.size()<<" max_groups="<<maxgroups<<" prepare_s="<<prepare_s<<"\n";'''
PREP_END_NEW='''    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();
    std::vector<DeviceGroupMeta> staged_group_meta;staged_group_meta.reserve(staged_group_count);
    for(auto&pw:schedule)for(auto&pg:pw.groups){pg.meta_id=staged_group_meta.size();DeviceGroupMeta m{};std::memcpy(m.main_dp,pg.ms.dp,sizeof(pg.ms.dp));std::memcpy(m.block_dp,pg.ds.dp,sizeof(pg.ds.dp));m.main_fixed=pg.mf;m.main_occ=pg.mo;m.block_fixed=pg.bf;m.block_occ=pg.bo;staged_group_meta.push_back(m);}
    size_t staged_group_meta_bytes=staged_group_meta.size()*sizeof(DeviceGroupMeta);
    for(auto&c:ctx)c.stage_group_meta(staged_group_meta);
    std::cerr<<"staged group meta: groups="<<staged_group_count<<" bytes_per_gpu="<<staged_group_meta_bytes<<" mib_per_gpu="<<double(staged_group_meta_bytes)/(1<<20)<<" total_h2d_gib="<<double(staged_group_meta_bytes)*ng/(1ull<<30)<<" copy_mode=H2D_once_then_D2D_per_group\n";
    staged_group_meta.clear();staged_group_meta.shrink_to_fit();
    double prepare_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-prep0).count();
    std::cerr<<"prepared windows="<<schedule.size()<<" max_groups="<<maxgroups<<" prepare_s="<<prepare_s<<"\n";'''


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1: raise SystemExit(f'{label}: expected exactly one packed-basearg match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,CTX_FIELD_OLD,CTX_FIELD_NEW,'DeviceCtx staged field')
    text=once(text,ENSURE_OLD,ENSURE_NEW,'DeviceCtx stage method')
    text=once(text,DESTROY_OLD,DESTROY_NEW,'DeviceCtx stage cleanup')
    text=once(text,PG_OLD,PG_NEW,'PreparedGroup meta id')
    text=once(text,COPY_OLD,COPY_NEW,'group metadata D2D switch')
    text=once(text,PREP_END_OLD,PREP_END_NEW,'schedule metadata staging')
    for stale in ('cudaMemcpyToSymbol(D_GROUP_META,&hmeta','DeviceGroupMeta hmeta{}'):
        if stale in text: raise SystemExit(f'per-group host metadata artifact remains: {stale}')
    for required in ('DeviceGroupMeta*dGroupMeta=nullptr','void stage_group_meta(const std::vector<DeviceGroupMeta>& h)','size_t meta_id=0','cudaMemcpyDeviceToDevice','staged group meta: groups=','staged_group_meta.clear()'):
        if required not in text: raise SystemExit(f'missing staged metadata artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} staged_group_meta=1 per_group_meta_source=device staged_h2d_once=1 per_group_constant_copy=d2d_sync host_meta_rebuild_per_group=0')

if __name__=='__main__':main()
