#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

DECL_OLD='''__constant__ DeviceGroupMeta D_GROUP_META;
#define D_MAIN_DP (D_GROUP_META.main_dp)
#define D_BLOCK_DP (D_GROUP_META.block_dp)
#define D_MAIN_FIXED (D_GROUP_META.main_fixed)
#define D_MAIN_OCC (D_GROUP_META.main_occ)
#define D_BLOCK_FIXED (D_GROUP_META.block_fixed)
#define D_BLOCK_OCC (D_GROUP_META.block_occ)'''
DECL_NEW='''__constant__ const DeviceGroupMeta* D_GROUP_META_PTR;
#define D_MAIN_DP (D_GROUP_META_PTR->main_dp)
#define D_BLOCK_DP (D_GROUP_META_PTR->block_dp)
#define D_MAIN_FIXED (D_GROUP_META_PTR->main_fixed)
#define D_MAIN_OCC (D_GROUP_META_PTR->main_occ)
#define D_BLOCK_FIXED (D_GROUP_META_PTR->block_fixed)
#define D_BLOCK_OCC (D_GROUP_META_PTR->block_occ)'''

COPY_OLD='''ck(cudaMemcpyToSymbol(D_GROUP_META,c.dGroupMeta+pg.meta_id,sizeof(DeviceGroupMeta),0,cudaMemcpyDeviceToDevice),"staged group meta D2D");'''
COPY_NEW='''const DeviceGroupMeta*group_meta_ptr=c.dGroupMeta+pg.meta_id;ck(cudaMemcpyToSymbol(D_GROUP_META_PTR,&group_meta_ptr,sizeof(group_meta_ptr)),"staged group meta pointer H2D");'''
LOG_OLD='copy_mode=H2D_once_local_then_D2D_per_group scheduler=static_lpt'
LOG_NEW='copy_mode=H2D_once_local_then_pointer_H2D_per_group scheduler=static_lpt meta_access=staged_global'


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1: raise SystemExit(f'{label}: expected exactly one staged-metadata match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,DECL_OLD,DECL_NEW,'metadata pointer declaration')
    text=once(text,COPY_OLD,COPY_NEW,'metadata pointer update')
    text=once(text,LOG_OLD,LOG_NEW,'metadata pointer runtime log')
    for stale in ('__constant__ DeviceGroupMeta D_GROUP_META;','cudaMemcpyToSymbol(D_GROUP_META,c.dGroupMeta+pg.meta_id','cudaMemcpyDeviceToDevice),"staged group meta D2D"','copy_mode=H2D_once_local_then_D2D_per_group'):
        if stale in text: raise SystemExit(f'full metadata constant-copy artifact remains: {stale}')
    for required in ('__constant__ const DeviceGroupMeta* D_GROUP_META_PTR;','D_GROUP_META_PTR->main_dp','D_GROUP_META_PTR->block_dp','const DeviceGroupMeta*group_meta_ptr=c.dGroupMeta+pg.meta_id','cudaMemcpyToSymbol(D_GROUP_META_PTR,&group_meta_ptr,sizeof(group_meta_ptr))','copy_mode=H2D_once_local_then_pointer_H2D_per_group','meta_access=staged_global'):
        if required not in text: raise SystemExit(f'missing staged metadata pointer artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} group_meta_source=staged_global_via_constant_pointer constant_payload_bytes_per_group=8 old_constant_payload_bytes_per_group=13936 payload_reduction=1742x per_group_symbol_copy_calls=1 runtime_copy_mode=pointer_h2d')

if __name__=='__main__':main()
