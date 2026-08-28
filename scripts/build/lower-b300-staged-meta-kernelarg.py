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
DECL_NEW=''

COPY_OLD='''ck(cudaMemcpyToSymbol(D_GROUP_META,c.dGroupMeta+pg.meta_id,sizeof(DeviceGroupMeta),0,cudaMemcpyDeviceToDevice),"staged group meta D2D");'''
COPY_NEW='''const DeviceGroupMeta*group_meta=c.dGroupMeta+pg.meta_id;'''

SIG_REPLACEMENTS=(
('gather_main_kernel(Count*out,MateID*mates,Code n,const Count* auth)', 'gather_main_kernel(Count*out,MateID*mates,Code n,const Count* auth,const DeviceGroupMeta*meta)'),
('gather_block_kernel(Count*out,Code n,const Count* auth)', 'gather_block_kernel(Count*out,Code n,const Count* auth,const DeviceGroupMeta*meta)'),
('scatter_main_kernel(const Count*in,Code n,Count* auth)', 'scatter_main_kernel(const Count*in,Code n,Count* auth,const DeviceGroupMeta*meta)'),
('scatter_block_kernel(const Count*in,Code n,Count* auth)', 'scatter_block_kernel(const Count*in,Code n,Count* auth,const DeviceGroupMeta*meta)'),
('rank_drop_n_t(Code src_rank,MateID m,int p)', 'rank_drop_n_t(Code src_rank,MateID m,int p,const DeviceGroupMeta*meta)'),
('blocked_group_kernel(const Count*in,Code n,Count*out_main,int p)', 'blocked_group_kernel(const Count*in,Code n,Count*out_main,int p,const DeviceGroupMeta*meta)'),
('main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p)', 'main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p,const DeviceGroupMeta*meta)'),
)

LAUNCH_REPLACEMENTS=(
('gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain)', 'gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain,group_meta)'),
('gather_block_kernel<<<bd,threads>>>(c.dD,ds.size,c.authBlock)', 'gather_block_kernel<<<bd,threads>>>(c.dD,ds.size,c.authBlock,group_meta)'),
('main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p)', 'main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p,group_meta)'),
('blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,nxt,p)', 'blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,nxt,p,group_meta)'),
('scatter_main_kernel<<<bm,threads>>>(cur,ms.size,c.authMain)', 'scatter_main_kernel<<<bm,threads>>>(cur,ms.size,c.authMain,group_meta)'),
('scatter_block_kernel<<<bd,threads>>>(dcur,ds.size,c.authBlock)', 'scatter_block_kernel<<<bd,threads>>>(dcur,ds.size,c.authBlock,group_meta)'),
)

TOKEN_REPLACEMENTS=(
('D_MAIN_FIXED','meta->main_fixed'),('D_MAIN_OCC','meta->main_occ'),('D_BLOCK_FIXED','meta->block_fixed'),('D_BLOCK_OCC','meta->block_occ'),('D_MAIN_DP','meta->main_dp'),('D_BLOCK_DP','meta->block_dp'),
)

RANK_DROP_CALL_OLD='rank_drop_n_t<TARGET_W>(i,m,p)'
RANK_DROP_CALL_NEW='rank_drop_n_t<TARGET_W>(i,m,p,meta)'
LOG_OLD='copy_mode=H2D_once_local_then_D2D_per_group scheduler=static_lpt'
LOG_NEW='copy_mode=H2D_once_local_then_no_meta_copy_per_group scheduler=static_lpt meta_access=staged_global_kernelarg'


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1: raise SystemExit(f'{label}: expected exactly one staged/static match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,DECL_OLD,DECL_NEW,'remove full constant metadata object/macros')
    text=once(text,COPY_OLD,COPY_NEW,'remove per-group metadata copy')
    for old,new in SIG_REPLACEMENTS:text=once(text,old,new,f'kernel/helper signature {old.split("(")[0]}')
    text=once(text,RANK_DROP_CALL_OLD,RANK_DROP_CALL_NEW,'rank-drop metadata argument')
    for old,new in TOKEN_REPLACEMENTS:
        n=text.count(old)
        if n<1: raise SystemExit(f'metadata token {old}: expected at least one device use, got {n}')
        text=text.replace(old,new)
    for old,new in LAUNCH_REPLACEMENTS:text=once(text,old,new,f'kernel launch {old.split("<")[0]}')
    text=once(text,LOG_OLD,LOG_NEW,'zero-copy metadata runtime log')
    for stale in ('D_GROUP_META','D_MAIN_DP','D_BLOCK_DP','D_MAIN_FIXED','D_MAIN_OCC','D_BLOCK_FIXED','D_BLOCK_OCC','cudaMemcpyToSymbol(D_GROUP_META','staged group meta D2D','copy_mode=H2D_once_local_then_D2D_per_group'):
        if stale in text: raise SystemExit(f'constant metadata artifact remains after kernel-arg lowering: {stale}')
    for required in ('const DeviceGroupMeta*group_meta=c.dGroupMeta+pg.meta_id','gather_main_kernel(Count*out,MateID*mates,Code n,const Count* auth,const DeviceGroupMeta*meta)','rank_drop_n_t(Code src_rank,MateID m,int p,const DeviceGroupMeta*meta)','rank_drop_n_t<TARGET_W>(i,m,p,meta)','main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p,group_meta)','copy_mode=H2D_once_local_then_no_meta_copy_per_group','meta_access=staged_global_kernelarg'):
        if required not in text: raise SystemExit(f'missing zero-copy metadata kernel-arg artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} group_meta_source=staged_global_kernelarg constant_metadata_object=0 per_group_meta_copy_bytes=0 per_group_meta_symbol_copy_calls=0 metadata_pointer_kernel_args=1 scheduler=static_lpt')

if __name__=='__main__':main()
