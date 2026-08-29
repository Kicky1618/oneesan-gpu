#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

ENSURE_OLD='''    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,need=2*ab+2*db+mb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}'''
ENSURE_NEW='''    void reserve_arena(size_t need){ck(cudaSetDevice(dev),"set reserve arena");if(arena)cudaFree(arena);arena=nullptr;capArena=need;if(need)ck(cudaMalloc(&arena,need),"preallocate scratch arena");}
    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,need=2*ab+2*db+mb;if(need>capArena){std::cerr<<"preallocated scratch arena too small gpu="<<dev<<" need="<<need<<" cap="<<capArena<<"\\n";std::exit(18);}'''
PREP_OLD='''    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();'''
PREP_NEW='''    std::vector<size_t> prealloc_arena_bytes(ng,0);auto arena_align=[](size_t x){return(x+255)&~size_t(255);};
    for(auto const&pw:schedule)for(int d=0;d<ng;++d)for(int q:pw.by_gpu[d]){auto const&pg=pw.groups[q];auto const&ms=pg.ms;auto const&ds=pg.ds;size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=!pg.use_mi&&(countBytes+mateBytes<=target);size_t ab=arena_align(size_t(ms.size)*sizeof(Count)),db=arena_align(size_t(ds.size)*sizeof(Count)),mb=useMate?arena_align(size_t(ms.size)*sizeof(MateID)):0;prealloc_arena_bytes[d]=std::max(prealloc_arena_bytes[d],2*ab+2*db+mb);}
    for(int d=0;d<ng;++d){if(prealloc_arena_bytes[d]>target){std::cerr<<"preallocated scratch exceeds target gpu="<<d<<" bytes="<<prealloc_arena_bytes[d]<<" target="<<target<<"\\n";return 18;}ctx[d].reserve_arena(prealloc_arena_bytes[d]);}
    std::cerr<<"preallocated scratch arenas: max_mib="<<double(*std::max_element(prealloc_arena_bytes.begin(),prealloc_arena_bytes.end()))/(1<<20)<<" realloc_per_group=0\\n";
    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();'''
START_OLD='''    auto t0=std::chrono::steady_clock::now();'''
START_NEW=''
END_OLD='''    ck(cudaGetLastError(),"doubleD scatter");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block scatter done");ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block scatter");ck(cudaStreamSynchronize(c.sMain),"scatter join sync");c.groups++;c.active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();'''
END_NEW='''    ck(cudaGetLastError(),"doubleD scatter");ck(cudaEventRecord(c.mainDone,c.sMain),"record main scatter done");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block scatter done");ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block scatter");ck(cudaStreamWaitEvent(c.sBlock,c.mainDone,0),"block wait main scatter");c.groups++;'''
WORK_OLD='''for(int row=0;row<b300_row_limit;++row){for(size_t wi=0;wi<schedule.size();++wi){auto const&pw=schedule[wi];for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);host_barrier.wait();if(d==0){int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}}}'''
WORK_NEW='''for(int row=0;row<b300_row_limit;++row){for(size_t wi=0;wi<schedule.size();++wi){auto const&pw=schedule[wi];auto window_t0=std::chrono::steady_clock::now();for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);ck(cudaStreamSynchronize(ctx[d].sMain),"window batch sync");ctx[d].active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_t0).count();host_barrier.wait();if(d==0){int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}}}'''

def once(t,o,n,l):
    c=t.count(o)
    if c!=1:raise SystemExit(f'{l}: expected one persistent-async match, got {c}')
    return t.replace(o,n,1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();t=a.src.read_text();t=once(t,ENSURE_OLD,ENSURE_NEW,'preallocated ensure');t=once(t,PREP_OLD,PREP_NEW,'arena preallocation');t=once(t,START_OLD,START_NEW,'remove per-group timer');t=once(t,END_OLD,END_NEW,'remove per-group host sync');t=once(t,WORK_OLD,WORK_NEW,'window batch synchronization')
    for s in ('scatter join sync','c.active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()','if(need>capArena){if(arena)cudaFree(arena)'):
        if s in t:raise SystemExit(f'per-group synchronization/allocation artifact remains: {s}')
    for r in ('void reserve_arena(size_t need)','prealloc_arena_bytes','realloc_per_group=0','record main scatter done','block wait main scatter','window batch sync','ctx[d].active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_t0).count()'):
        if r not in t:raise SystemExit(f'missing window-batch artifact: {r}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(t);print(f'lowered {a.out} per_group_host_syncs=0 window_syncs_per_gpu=56 expected_total_window_host_syncs=448 scratch_preallocated=1 scratch_realloc_per_group=0 group_completion_by_stream_events=1')
if __name__=='__main__':main()
