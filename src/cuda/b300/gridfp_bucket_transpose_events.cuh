#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Event-driven variant of BucketTransposeCtx.
// cudaStreamWaitEvent may wait on an event from another CUDA device.  Encode
// in-place swap safety entirely as GPU dependencies and avoid per-chunk host
// synchronization.  Fetch events are unique per (round,chunk), so correctness
// does not depend on event re-record semantics.

struct BucketTransposeEventCtx {
    static constexpr int N=BUCKET_TRANSPOSE_MAX_GPU;
    int ngpu=0;
    size_t chunk_bytes=0;
    size_t max_chunks=0;
    std::array<uint8_t*,N> base{};
    std::array<uint8_t*,N> staging{};
    std::array<cudaStream_t,N> stream{};
    std::array<cudaEvent_t,N> round_done{};
    std::array<std::vector<cudaEvent_t>,N> fetch_done;
    std::vector<std::array<std::pair<int,int>,N/2>> rounds;
    double peer_gib=0.0;
    double local_gib=0.0;
    uint64_t transposes=0;

    static void cke(cudaError_t e,const char*what){
        if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<'\n';std::exit(220);}
    }

    size_t fetch_index(size_t round,size_t chunk)const{return round*max_chunks+chunk;}

    void init(
        const BucketTransposePlan&plan,
        const std::array<Count*,N>&ptrs,
        size_t chunk
    ){
        ngpu=plan.ngpu;chunk_bytes=chunk;rounds=bucket_transpose_rounds(ngpu);
        if(!chunk_bytes){std::cerr<<"bucket event transpose zero chunk\n";std::exit(221);}
        uint64_t max_cap=0;
        for(int a=0;a<ngpu;++a)for(int b=0;b<ngpu;++b)
            if(a!=b)max_cap=std::max(max_cap,plan.slot[a][b].capacity_bytes);
        max_chunks=size_t((max_cap+chunk_bytes-1)/chunk_bytes);
        if(!max_chunks)max_chunks=1;
        size_t event_count=rounds.size()*max_chunks;

        for(int g=0;g<ngpu;++g){
            base[g]=reinterpret_cast<uint8_t*>(ptrs[g]);
            cke(cudaSetDevice(g),"bucket event transpose set init device");
            for(int peer=0;peer<ngpu;++peer)if(peer!=g){
                int can=0;cke(cudaDeviceCanAccessPeer(&can,g,peer),"bucket event transpose peer query");
                if(!can){std::cerr<<"bucket event transpose no P2P "<<g<<"->"<<peer<<'\n';std::exit(222);}
                cudaError_t e=cudaDeviceEnablePeerAccess(peer,0);
                if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)cke(e,"bucket event transpose enable peer");
                if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
            }
            cke(cudaMalloc(&staging[g],chunk_bytes),"bucket event transpose staging alloc");
            cke(cudaStreamCreateWithFlags(&stream[g],cudaStreamNonBlocking),"bucket event transpose stream");
            cke(cudaEventCreateWithFlags(&round_done[g],cudaEventDisableTiming),"bucket event transpose round event");
            fetch_done[g].resize(event_count,nullptr);
            for(size_t k=0;k<event_count;++k)
                cke(cudaEventCreateWithFlags(&fetch_done[g][k],cudaEventDisableTiming),"bucket event transpose fetch event");
        }
    }

    void transpose(const BucketTransposePlan&plan){
        if(plan.ngpu!=ngpu)std::exit(223);
        for(size_t ri=0;ri<rounds.size();++ri){
            auto const&round=rounds[ri];

            // A new pair must not read a source slot while its peer still has
            // writes pending from the preceding matching round.
            if(ri){
                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];
                    cke(cudaSetDevice(a),"bucket event transpose set A round wait");
                    cke(cudaStreamWaitEvent(stream[a],round_done[b],0),"bucket event transpose A round wait");
                    cke(cudaSetDevice(b),"bucket event transpose set B round wait");
                    cke(cudaStreamWaitEvent(stream[b],round_done[a],0),"bucket event transpose B round wait");
                }
            }

            uint64_t max_cap=0;
            for(int i=0;i<ngpu/2;++i){auto[a,b]=round[size_t(i)];max_cap=std::max(max_cap,plan.slot[a][b].capacity_bytes);}
            size_t chunk_index=0;
            for(uint64_t off=0;off<max_cap;off+=chunk_bytes,++chunk_index){
                if(chunk_index>=max_chunks)std::exit(224);
                size_t ei=fetch_index(ri,chunk_index);
                std::array<size_t,N> nbytes{};

                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];uint64_t cap=plan.slot[a][b].capacity_bytes;
                    if(off>=cap)continue;
                    size_t n=size_t(std::min<uint64_t>(chunk_bytes,cap-off));nbytes[a]=nbytes[b]=n;
                    cke(cudaSetDevice(a),"bucket event transpose set A fetch");
                    cke(cudaMemcpyPeerAsync(staging[a],a,base[b]+plan.slot[b][a].off_bytes+off,b,n,stream[a]),"bucket event transpose fetch B to A");
                    cke(cudaEventRecord(fetch_done[a][ei],stream[a]),"bucket event transpose record A fetch");
                    cke(cudaSetDevice(b),"bucket event transpose set B fetch");
                    cke(cudaMemcpyPeerAsync(staging[b],b,base[a]+plan.slot[a][b].off_bytes+off,a,n,stream[b]),"bucket event transpose fetch A to B");
                    cke(cudaEventRecord(fetch_done[b][ei],stream[b]),"bucket event transpose record B fetch");
                    peer_gib+=double(2*n)/double(1ULL<<30);
                }

                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];size_t n=nbytes[a];if(!n)continue;
                    cke(cudaSetDevice(a),"bucket event transpose set A commit");
                    cke(cudaStreamWaitEvent(stream[a],fetch_done[b][ei],0),"bucket event transpose A waits B fetch");
                    cke(cudaMemcpyAsync(base[a]+plan.slot[a][b].off_bytes+off,staging[a],n,cudaMemcpyDeviceToDevice,stream[a]),"bucket event transpose commit A");
                    cke(cudaSetDevice(b),"bucket event transpose set B commit");
                    cke(cudaStreamWaitEvent(stream[b],fetch_done[a][ei],0),"bucket event transpose B waits A fetch");
                    cke(cudaMemcpyAsync(base[b]+plan.slot[b][a].off_bytes+off,staging[b],n,cudaMemcpyDeviceToDevice,stream[b]),"bucket event transpose commit B");
                    local_gib+=double(2*n)/double(1ULL<<30);
                }
            }

            for(int g=0;g<ngpu;++g){
                cke(cudaSetDevice(g),"bucket event transpose set round record");
                cke(cudaEventRecord(round_done[g],stream[g]),"bucket event transpose record round done");
            }
        }

        // The next compute window uses the default stream. Queue a device-side
        // wait rather than blocking the host here.
        for(int g=0;g<ngpu;++g){
            cke(cudaSetDevice(g),"bucket event transpose set final wait");
            cke(cudaStreamWaitEvent(nullptr,round_done[g],0),"bucket event transpose default waits done");
        }
        ++transposes;
    }

    void synchronize(){
        for(int g=0;g<ngpu;++g){cke(cudaSetDevice(g),"bucket event transpose set synchronize");cke(cudaEventSynchronize(round_done[g]),"bucket event transpose synchronize");}
    }

    void release(){
        if(ngpu)synchronize();
        for(int g=0;g<ngpu;++g){
            cke(cudaSetDevice(g),"bucket event transpose set release");
            for(cudaEvent_t&e:fetch_done[g]){if(e)cudaEventDestroy(e);e=nullptr;}
            fetch_done[g].clear();
            if(round_done[g])cudaEventDestroy(round_done[g]);
            if(stream[g])cudaStreamDestroy(stream[g]);
            if(staging[g])cudaFree(staging[g]);
            round_done[g]=nullptr;stream[g]=nullptr;staging[g]=nullptr;base[g]=nullptr;
        }
        ngpu=0;chunk_bytes=0;max_chunks=0;rounds.clear();
    }
};
