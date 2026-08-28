#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

WIDTH_NGPU_OLD='__constant__ int D_MAIN_W,D_BLOCK_W,D_NGPU;'
WIDTH_NGPU_NEW=''

SHARD_SYMBOLS_OLD='''__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Count* D_MAIN_VBASE;
__constant__ Count* D_BLOCK_VBASE;
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;'''
SHARD_SYMBOLS_NEW='''__constant__ Count* D_MAIN_VBASE;
__constant__ Count* D_BLOCK_VBASE;'''

INTERVAL_SIG_OLD='''static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,Code chunk,int ng,bool& use_interval){
    (void)chunk;(void)ng;'''
INTERVAL_SIG_NEW='''static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,bool& use_interval){'''

PREPARE_OLD='''static PreparedGroup prepare_group(int W,const WindowPlan&wp,int g,Code mc,Code bc,int ng){
    PreparedGroup pg;pg.g=g;
    window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,pg.mf,pg.mo,pg.bf,pg.bo);
    pg.ms=make_spec(W,pg.mf,pg.mo);pg.ds=make_spec(W-1,pg.bf,pg.bo);
    pg.work=2*pg.ms.size+pg.ds.size;
    pg.mi=make_peer_intervals(pg.ms,mc,ng,pg.use_mi);
    pg.di=make_peer_intervals(pg.ds,bc,ng,pg.use_di);
    return pg;
}'''
PREPARE_NEW='''static PreparedGroup prepare_group(int W,const WindowPlan&wp,int g){
    PreparedGroup pg;pg.g=g;
    window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,pg.mf,pg.mo,pg.bf,pg.bo);
    pg.ms=make_spec(W,pg.mf,pg.mo);pg.ds=make_spec(W-1,pg.bf,pg.bo);
    pg.work=2*pg.ms.size+pg.ds.size;
    pg.mi=make_peer_intervals(pg.ms,pg.use_mi);
    pg.di=make_peer_intervals(pg.ds,pg.use_di);
    return pg;
}'''

COUNTS_OLD='Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;'
COUNTS_NEW='Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];'

LOGICAL_VIEWS_OLD='''Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));mp[d]=main_base+Code(d)*mc;bp[d]=block_base+Code(d)*bc;}'''
LOGICAL_VIEWS_NEW=''

INIT_OLD='''void init(int d,Count mod,Count**mp,Count**bp,Code mc,Code bc,int ng){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"main ptrs");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"block ptrs");ck(cudaMemcpyToSymbol(D_MAIN_CHUNK,&mc,sizeof(mc)),"main chunk");ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK,&bc,sizeof(bc)),"block chunk");ck(cudaMemcpyToSymbol(D_NGPU,&ng,sizeof(ng)),"ngpu");ck(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sBlock,cudaStreamNonBlocking),"stream block");ck(cudaEventCreateWithFlags(&copyDone,cudaEventDisableTiming),"event copy");ck(cudaEventCreateWithFlags(&clearDone,cudaEventDisableTiming),"event clear");ck(cudaEventCreateWithFlags(&mainDone,cudaEventDisableTiming),"event main");ck(cudaEventCreateWithFlags(&blockDone,cudaEventDisableTiming),"event block");}'''
INIT_NEW='''void init(int d,Count mod){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sBlock,cudaStreamNonBlocking),"stream block");ck(cudaEventCreateWithFlags(&copyDone,cudaEventDisableTiming),"event copy");ck(cudaEventCreateWithFlags(&clearDone,cudaEventDisableTiming),"event clear");ck(cudaEventCreateWithFlags(&mainDone,cudaEventDisableTiming),"event main");ck(cudaEventCreateWithFlags(&blockDone,cudaEventDisableTiming),"event block");}'''

INIT_CALL_OLD='ctx[d].init(d,mod,mp,bp,mc,bc,ng);'
INIT_CALL_NEW='ctx[d].init(d,mod);'

WIDTH_COPY_OLD='''int mw=W,bw=W-1;ck(cudaMemcpyToSymbol(D_MAIN_W,&mw,sizeof(mw)),"mw");ck(cudaMemcpyToSymbol(D_BLOCK_W,&bw,sizeof(bw)),"bw");'''
WIDTH_COPY_NEW=''

SCHEDULE_CALL_OLD='prepare_group(W,pw.wp,g,mc,bc,ng)'
SCHEDULE_CALL_NEW='prepare_group(W,pw.wp,g)'

INIT_STATE_OLD='''MateID init=MateID(R)<<(2*(W-1));Code ig=rank_full(init,W);int io=int(ig/mc);Count one=1;cudaSetDevice(io);ck(cudaMemcpy(mp[io]+(ig-Code(io)*mc),&one,sizeof(one),cudaMemcpyHostToDevice),"init one");'''
INIT_STATE_NEW='''MateID init=MateID(R)<<(2*(W-1));Code ig=rank_full(init,W);Count one=1;cudaSetDevice(0);ck(cudaMemcpy(main_base+ig,&one,sizeof(one),cudaMemcpyHostToDevice),"init one VMM direct");'''

FINAL_STATE_OLD='''double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();Code fg=rank_full(MateID(R),W);int fo=int(fg/mc);Count ans=0;cudaSetDevice(fo);ck(cudaMemcpy(&ans,mp[fo]+(fg-Code(fo)*mc),sizeof(ans),cudaMemcpyDeviceToHost),"answer");'''
FINAL_STATE_NEW='''double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();Code fg=rank_full(MateID(R),W);Count ans=0;cudaSetDevice(0);ck(cudaMemcpy(&ans,main_base+fg,sizeof(ans),cudaMemcpyDeviceToHost),"answer VMM direct");'''


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1:
        raise SystemExit(f'{label}: expected exactly one generated-source match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('src',type=Path)
    ap.add_argument('out',type=Path)
    a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,WIDTH_NGPU_OLD,WIDTH_NGPU_NEW,'stale width/ngpu declarations')
    text=once(text,SHARD_SYMBOLS_OLD,SHARD_SYMBOLS_NEW,'stale shard symbol declarations')
    text=once(text,INTERVAL_SIG_OLD,INTERVAL_SIG_NEW,'shard-free interval signature')
    text=once(text,PREPARE_OLD,PREPARE_NEW,'shard-free prepare_group')
    text=once(text,COUNTS_OLD,COUNTS_NEW,'remove logical shard chunks')
    text=once(text,LOGICAL_VIEWS_OLD,LOGICAL_VIEWS_NEW,'remove logical shard views')
    text=once(text,INIT_OLD,INIT_NEW,'DeviceCtx stale shard symbol copies')
    text=once(text,INIT_CALL_OLD,INIT_CALL_NEW,'DeviceCtx stale shard init arguments')
    text=once(text,WIDTH_COPY_OLD,WIDTH_COPY_NEW,'per-group stale width symbol copies')
    text=once(text,SCHEDULE_CALL_OLD,SCHEDULE_CALL_NEW,'shard-free prepare_group call')
    text=once(text,INIT_STATE_OLD,INIT_STATE_NEW,'direct VMM initial state')
    text=once(text,FINAL_STATE_OLD,FINAL_STATE_NEW,'direct VMM final state')
    for token in ('D_MAIN_PTR','D_BLOCK_PTR','D_MAIN_CHUNK','D_BLOCK_CHUNK','D_NGPU','D_MAIN_W','D_BLOCK_W'):
        if token in text:
            raise SystemExit(f'stale VMM symbol remains after prune: {token}')
    for token in ('Code mc=','bc=(blockN+ng-1)/ng','Count*mp[MAXGPU]','*bp[MAXGPU]','std::vector<Code>ml','ml[d]=','bl[d]=','int io=int(ig/mc)','int fo=int(fg/mc)','prepare_group(W,pw.wp,g,mc,bc,ng)','make_peer_intervals(pg.ms,mc,ng'):
        if token in text:
            raise SystemExit(f'logical shard artifact remains after prune: {token}')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(text)
    print(f'pruned {a.out} stale_shard_symbols=0 stale_shard_symbol_copies=0 stale_shard_init_args=0 stale_width_symbols=0 per_group_width_symbol_copies=0 logical_shard_chunks=0 logical_shard_views=0 host_owner_div=0 direct_vmm_symbols=2')

if __name__=='__main__':
    main()
