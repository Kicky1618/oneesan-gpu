#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Double-buffered event pipeline for raw bucket transpose.
//
// For chunk k on every GPU:
//   fetch_stream: peer -> staging[k&1], record fetch_done
//   commit_stream: wait own fetch_done + peer fetch_done, staging -> final,
//                  record commit_done
//
// Because k+1 uses the other staging buffer, its bidirectional peer fetch may
// overlap the local HBM commit of k.  The two directions of the peer fetch
// remain on separate device streams, preserving any full-duplex NVLink
// capability.  Reusing a staging buffer waits on its preceding commit event.
// Round boundaries wait on the previous round's peer and local completion so
// no new source is read while a prior matching is still writing it.

struct BucketTransposePipelineCtx {
    static constexpr int N=BUCKET_TRANSPOSE_MAX_GPU;
    int ngpu=0;
    size_t chunk_bytes=0;
    size_t max_chunks=0;
    std::array<uint8_t*,N> base{};
    std::array<std::array<uint8_t*,2>,N> staging{};
    std::array<cudaStream_t,N> fetch_stream{};
    std::array<cudaStream_t,N> commit_stream{};
    std::array<std::array<cudaEvent_t,N-1>,N> round_done{};
    std::array<std::vector<cudaEvent_t>,N> fetch_done;
    std::array<std::vector<cudaEvent_t>,N> commit_done;
    std::vector<std::array<std::pair<int,int>,N/2>> rounds;
    double peer_gib=0.0;
    double local_gib=0.0;
    uint64_t transposes=0;

    static void ckp(cudaError_t e,const char*what){
        if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<'\n';std::exit(230);}
    }
    size_t event_index(size_t round,size_t chunk)const{return round*max_chunks+chunk;}

    void init(const BucketTransposePlan&plan,const std::array<Count*,N>&ptrs,size_t chunk){
        ngpu=plan.ngpu;chunk_bytes=chunk;rounds=bucket_transpose_rounds(ngpu);
        if(!chunk_bytes){std::cerr<<"bucket pipeline transpose zero chunk\n";std::exit(231);}
        uint64_t max_cap=0;
        for(int a=0;a<ngpu;++a)for(int b=0;b<ngpu;++b)if(a!=b)
            max_cap=std::max(max_cap,plan.slot[a][b].capacity_bytes);
        max_chunks=size_t((max_cap+chunk_bytes-1)/chunk_bytes);if(!max_chunks)max_chunks=1;
        size_t nevents=rounds.size()*max_chunks;

        for(int g=0;g<ngpu;++g){
            base[g]=reinterpret_cast<uint8_t*>(ptrs[g]);
            ckp(cudaSetDevice(g),"bucket pipeline set init device");
            for(int peer=0;peer<ngpu;++peer)if(peer!=g){
                int can=0;ckp(cudaDeviceCanAccessPeer(&can,g,peer),"bucket pipeline peer query");
                if(!can){std::cerr<<"bucket pipeline no P2P "<<g<<"->"<<peer<<'\n';std::exit(232);}
                cudaError_t e=cudaDeviceEnablePeerAccess(peer,0);
                if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)ckp(e,"bucket pipeline enable peer");
                if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
            }
            for(int b=0;b<2;++b)ckp(cudaMalloc(&staging[g][b],chunk_bytes),"bucket pipeline staging alloc");
            ckp(cudaStreamCreateWithFlags(&fetch_stream[g],cudaStreamNonBlocking),"bucket pipeline fetch stream");
            ckp(cudaStreamCreateWithFlags(&commit_stream[g],cudaStreamNonBlocking),"bucket pipeline commit stream");
            for(size_t ri=0;ri<rounds.size();++ri)
                ckp(cudaEventCreateWithFlags(&round_done[g][ri],cudaEventDisableTiming),"bucket pipeline round event");
            fetch_done[g].resize(nevents,nullptr);commit_done[g].resize(nevents,nullptr);
            for(size_t i=0;i<nevents;++i){
                ckp(cudaEventCreateWithFlags(&fetch_done[g][i],cudaEventDisableTiming),"bucket pipeline fetch event");
                ckp(cudaEventCreateWithFlags(&commit_done[g][i],cudaEventDisableTiming),"bucket pipeline commit event");
            }
        }
    }

    void synchronize(){
        if(!ngpu||rounds.empty())return;
        size_t last=rounds.size()-1;
        for(int g=0;g<ngpu;++g){
            ckp(cudaSetDevice(g),"bucket pipeline set synchronize");
            ckp(cudaEventSynchronize(round_done[g][last]),"bucket pipeline synchronize");
        }
    }

    void transpose(const BucketTransposePlan&plan){
        if(plan.ngpu!=ngpu)std::exit(233);
        for(size_t ri=0;ri<rounds.size();++ri){
            auto const&round=rounds[ri];
            if(ri){
                // New fetches must see all writes from the preceding matching,
                // and both staging buffers must have finished their old commits.
                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];
                    ckp(cudaSetDevice(a),"bucket pipeline set A round wait");
                    ckp(cudaStreamWaitEvent(fetch_stream[a],round_done[a][ri-1],0),"bucket pipeline A waits own round");
                    ckp(cudaStreamWaitEvent(fetch_stream[a],round_done[b][ri-1],0),"bucket pipeline A waits peer round");
                    ckp(cudaSetDevice(b),"bucket pipeline set B round wait");
                    ckp(cudaStreamWaitEvent(fetch_stream[b],round_done[b][ri-1],0),"bucket pipeline B waits own round");
                    ckp(cudaStreamWaitEvent(fetch_stream[b],round_done[a][ri-1],0),"bucket pipeline B waits peer round");
                }
            }

            uint64_t max_cap=0;
            for(int i=0;i<ngpu/2;++i){auto[a,b]=round[size_t(i)];max_cap=std::max(max_cap,plan.slot[a][b].capacity_bytes);}
            size_t k=0;
            for(uint64_t off=0;off<max_cap;off+=chunk_bytes,++k){
                if(k>=max_chunks)std::exit(234);
                size_t ei=event_index(ri,k);int buf=int(k&1u);
                std::array<size_t,N> nbytes{};

                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];uint64_t cap=plan.slot[a][b].capacity_bytes;
                    if(off>=cap)continue;size_t n=size_t(std::min<uint64_t>(chunk_bytes,cap-off));nbytes[a]=nbytes[b]=n;
                    if(k>=2){
                        size_t old=event_index(ri,k-2);
                        ckp(cudaSetDevice(a),"bucket pipeline set A buffer reuse");
                        ckp(cudaStreamWaitEvent(fetch_stream[a],commit_done[a][old],0),"bucket pipeline A waits buffer reuse");
                        ckp(cudaSetDevice(b),"bucket pipeline set B buffer reuse");
                        ckp(cudaStreamWaitEvent(fetch_stream[b],commit_done[b][old],0),"bucket pipeline B waits buffer reuse");
                    }
                    ckp(cudaSetDevice(a),"bucket pipeline set A fetch");
                    ckp(cudaMemcpyPeerAsync(staging[a][buf],a,base[b]+plan.slot[b][a].off_bytes+off,b,n,fetch_stream[a]),"bucket pipeline fetch B to A");
                    ckp(cudaEventRecord(fetch_done[a][ei],fetch_stream[a]),"bucket pipeline record A fetch");
                    ckp(cudaSetDevice(b),"bucket pipeline set B fetch");
                    ckp(cudaMemcpyPeerAsync(staging[b][buf],b,base[a]+plan.slot[a][b].off_bytes+off,a,n,fetch_stream[b]),"bucket pipeline fetch A to B");
                    ckp(cudaEventRecord(fetch_done[b][ei],fetch_stream[b]),"bucket pipeline record B fetch");
                    peer_gib+=double(2*n)/double(1ULL<<30);
                }

                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];size_t n=nbytes[a];if(!n)continue;
                    ckp(cudaSetDevice(a),"bucket pipeline set A commit");
                    ckp(cudaStreamWaitEvent(commit_stream[a],fetch_done[a][ei],0),"bucket pipeline A waits own fetch");
                    ckp(cudaStreamWaitEvent(commit_stream[a],fetch_done[b][ei],0),"bucket pipeline A waits peer fetch");
                    ckp(cudaMemcpyAsync(base[a]+plan.slot[a][b].off_bytes+off,staging[a][buf],n,cudaMemcpyDeviceToDevice,commit_stream[a]),"bucket pipeline commit A");
                    ckp(cudaEventRecord(commit_done[a][ei],commit_stream[a]),"bucket pipeline record A commit");
                    ckp(cudaSetDevice(b),"bucket pipeline set B commit");
                    ckp(cudaStreamWaitEvent(commit_stream[b],fetch_done[b][ei],0),"bucket pipeline B waits own fetch");
                    ckp(cudaStreamWaitEvent(commit_stream[b],fetch_done[a][ei],0),"bucket pipeline B waits peer fetch");
                    ckp(cudaMemcpyAsync(base[b]+plan.slot[b][a].off_bytes+off,staging[b][buf],n,cudaMemcpyDeviceToDevice,commit_stream[b]),"bucket pipeline commit B");
                    ckp(cudaEventRecord(commit_done[b][ei],commit_stream[b]),"bucket pipeline record B commit");
                    local_gib+=double(2*n)/double(1ULL<<30);
                }
            }

            // round_done is recorded on commit_stream after every commit of the
            // round. The next matching waits on both its own and peer events.
            for(int g=0;g<ngpu;++g){
                ckp(cudaSetDevice(g),"bucket pipeline set round record");
                ckp(cudaEventRecord(round_done[g][ri],commit_stream[g]),"bucket pipeline record round done");
            }
        }
        synchronize();++transposes;
    }

    void release(){
        if(ngpu&&transposes)synchronize();
        for(int g=0;g<ngpu;++g){
            ckp(cudaSetDevice(g),"bucket pipeline set release");
            for(cudaEvent_t&e:fetch_done[g]){if(e)cudaEventDestroy(e);e=nullptr;}fetch_done[g].clear();
            for(cudaEvent_t&e:commit_done[g]){if(e)cudaEventDestroy(e);e=nullptr;}commit_done[g].clear();
            for(size_t ri=0;ri<rounds.size();++ri){if(round_done[g][ri])cudaEventDestroy(round_done[g][ri]);round_done[g][ri]=nullptr;}
            if(fetch_stream[g])cudaStreamDestroy(fetch_stream[g]);if(commit_stream[g])cudaStreamDestroy(commit_stream[g]);
            for(int b=0;b<2;++b){if(staging[g][b])cudaFree(staging[g][b]);staging[g][b]=nullptr;}
            fetch_stream[g]=nullptr;commit_stream[g]=nullptr;base[g]=nullptr;
        }
        ngpu=0;chunk_bytes=0;max_chunks=0;rounds.clear();
    }
};
