#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
OLD='''for(int row=0;row<b300_row_limit;++row){for(size_t wi=0;wi<schedule.size();++wi){auto const&pw=schedule[wi];auto window_t0=std::chrono::steady_clock::now();for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);ck(cudaStreamSynchronize(ctx[d].sMain),"window batch sync");ctx[d].active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_t0).count();host_barrier.wait();if(d==0){int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}}}'''
NEW='''for(int row=0;row<b300_row_limit;++row){for(size_t wi=0;wi<schedule.size();++wi){auto const&pw=schedule[wi];auto window_t0=std::chrono::steady_clock::now();for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);ck(cudaEventRecord(ctx[d].copyDone,ctx[d].sMain),"record GPU window done");host_barrier.wait();if(d==0){for(int q=1;q<ng;++q)ck(cudaStreamWaitEvent(ctx[0].sMain,ctx[q].copyDone,0),"gpu0 wait peer window");ck(cudaStreamSynchronize(ctx[0].sMain),"gpu0 global window sync");int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}host_barrier.wait();ctx[d].active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-window_t0).count();}}'''
def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();t=a.src.read_text();n=t.count(OLD)
    if n!=1:raise SystemExit(f'cross-device window barrier: expected one windowbatch match, got {n}')
    t=t.replace(OLD,NEW,1)
    if 'window batch sync' in t:raise SystemExit('per-GPU window host sync remains')
    for r in ('record GPU window done','gpu0 wait peer window','gpu0 global window sync','for(int q=1;q<ng;++q)','host_barrier.wait();ctx[d].active+='):
        if r not in t:raise SystemExit(f'missing cross-device barrier artifact: {r}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(t);print(f'lowered {a.out} cross_device_window_events=1 host_gpu_syncs_per_window=1 expected_windows=56 expected_host_gpu_sync_calls=56 old_windowbatch_host_gpu_sync_calls=448 sync_call_reduction=8x')
if __name__=='__main__':main()
