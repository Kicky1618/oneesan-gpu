#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
OLD='''if(d==0){for(int q=1;q<ng;++q)ck(cudaStreamWaitEvent(ctx[0].sMain,ctx[q].copyDone,0),"gpu0 wait peer window");ck(cudaStreamSynchronize(ctx[0].sMain),"gpu0 global window sync");int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}host_barrier.wait();ctx[d].active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_t0).count();'''
NEW='''if(d==0){for(int q=1;q<ng;++q)ck(cudaStreamWaitEvent(ctx[0].sMain,ctx[q].copyDone,0),"gpu0 wait peer window");ck(cudaEventRecord(ctx[0].clearDone,ctx[0].sMain),"record global window done");int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}host_barrier.wait();ck(cudaStreamWaitEvent(ctx[d].sMain,ctx[0].clearDone,0),"local main wait global window");ck(cudaStreamWaitEvent(ctx[d].sBlock,ctx[0].clearDone,0),"local block wait global window");if(wi+1==schedule.size()){host_barrier.wait();if(d==0)ck(cudaStreamSynchronize(ctx[0].sMain),"gpu0 row end sync");host_barrier.wait();}ctx[d].active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_t0).count();'''
def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();t=a.src.read_text();n=t.count(OLD)
    if n!=1:raise SystemExit(f'row-end sync: expected one cross-window match, got {n}')
    t=t.replace(OLD,NEW,1)
    if 'gpu0 global window sync' in t:raise SystemExit('per-window host GPU sync remains')
    for r in ('record global window done','local main wait global window','local block wait global window','gpu0 row end sync','if(wi+1==schedule.size())'):
        if r not in t:raise SystemExit(f'missing row-end window artifact: {r}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(t);print(f'lowered {a.out} cross_device_window_chain=1 host_gpu_syncs_per_row=1 expected_rows=28 expected_host_gpu_sync_calls=28 crosswindow_old_calls=56 windowbatch_old_calls=448')
if __name__=='__main__':main()
