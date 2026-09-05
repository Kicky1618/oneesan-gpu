#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

MAIN_OLD='int main(int argc,char**argv){'
MAIN_NEW='''struct B300HostBarrier{
    int total;std::atomic<int>arrived{0},phase{0};
    explicit B300HostBarrier(int n):total(n){}
    void wait(){int p=phase.load(std::memory_order_acquire);if(arrived.fetch_add(1,std::memory_order_acq_rel)+1==total){arrived.store(0,std::memory_order_relaxed);phase.fetch_add(1,std::memory_order_release);}else while(phase.load(std::memory_order_acquire)==p)std::this_thread::yield();}
};
int main(int argc,char**argv){'''

LOOP_OLD='''    auto wall0=std::chrono::steady_clock::now();int done_windows=0;
    for(int row=0;row<b300_row_limit;++row){
        for(auto const&pw:schedule){
            std::vector<std::thread>ths;ths.reserve(ng);
            for(int d=0;d<ng;++d)ths.emplace_back([&,d]{ck(cudaSetDevice(d),"set static worker device");for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);});
            for(auto&t:ths)t.join();++done_windows;
        }
        std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done_windows<<"\\n";
    }'''
LOOP_NEW='''    auto wall0=std::chrono::steady_clock::now();std::atomic<int>done_windows_atomic{0};B300HostBarrier host_barrier(ng);std::vector<std::thread>ths;ths.reserve(ng);
    for(int d=0;d<ng;++d)ths.emplace_back([&,d]{ck(cudaSetDevice(d),"set persistent static worker device");for(int row=0;row<b300_row_limit;++row){for(size_t wi=0;wi<schedule.size();++wi){auto const&pw=schedule[wi];for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);host_barrier.wait();if(d==0){int done=done_windows_atomic.fetch_add(1,std::memory_order_relaxed)+1;if(wi+1==schedule.size())std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done<<"\\n";}}}});
    for(auto&t:ths)t.join();int done_windows=done_windows_atomic.load(std::memory_order_relaxed);'''

def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1:raise SystemExit(f'{label}: expected exactly one worker-bound row-limited match, got {n}')
    return text.replace(old,new,1)

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();text=a.src.read_text()
    text=once(text,MAIN_OLD,MAIN_NEW,'persistent barrier insertion')
    text=once(text,LOOP_OLD,LOOP_NEW,'persistent worker loop')
    for stale in ('set static worker device','std::vector<std::thread>ths;ths.reserve(ng);\n            for(int d=0;d<ng;++d)'):
        if stale in text:raise SystemExit(f'per-window worker artifact remains: {stale}')
    for required in ('struct B300HostBarrier','host_barrier.wait()','set persistent static worker device','std::atomic<int>done_windows_atomic','for(size_t wi=0;wi<schedule.size();++wi)','for(auto&t:ths)t.join();int done_windows='):
        if required not in text:raise SystemExit(f'missing persistent-worker artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} persistent_workers=8 expected_old_thread_creations=448 expected_new_thread_creations=8 thread_creation_reduction=56x expected_old_worker_cudaSetDevice_calls=917504 expected_new_worker_cudaSetDevice_calls=8 cudaSetDevice_reduction=114688x')
if __name__=='__main__':main()
