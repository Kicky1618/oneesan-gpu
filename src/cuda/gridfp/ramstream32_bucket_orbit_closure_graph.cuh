#pragma once

#include "ramstream32_bucket_orbit_closure_stream.cuh"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// Four reusable one-pass window graphs per GPU. Kernel arguments p and grid
// geometry are captured once; bucket pointers and modulus live in device
// constant state and may be rebound/updated between launches.
enum BucketOnePassGraphKind : uint8_t {
    BKOC_GRAPH_FORWARD_LOW=0,
    BKOC_GRAPH_FORWARD_HIGH=1,
    BKOC_GRAPH_REVERSE_LOW=2,
    BKOC_GRAPH_REVERSE_HIGH=3,
    BKOC_GRAPH_COUNT=4,
};

struct BucketOnePassGraphs {
    cudaStream_t stream=nullptr;
    std::array<cudaGraphExec_t,BKOC_GRAPH_COUNT> exec{};
    int threads=0,gx=0,gy=0;

    template<class F>
    void capture_one(BucketOnePassGraphKind kind,F&&enqueue,const char*what){
        cudaGraph_t graph=nullptr;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),what);
        enqueue();
        ck(cudaStreamEndCapture(stream,&graph),what);
        if(!graph){std::cerr<<"one-pass graph capture returned null graph "<<what<<'\n';std::exit(390);}
        ck(cudaGraphInstantiate(&exec[size_t(kind)],graph,nullptr,nullptr,0),what);
        ck(cudaGraphDestroy(graph),what);
    }

    void init(const StorageLayout&layout,int t=256,int x=16,int y=8){
        threads=t;gx=x;gy=y;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"one-pass graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW,[&]{bucket_enqueue_low_orbit_closure_fused(layout,stream,threads,gx,gy);},"capture forward LOW graph");
        capture_one(BKOC_GRAPH_FORWARD_HIGH,[&]{bucket_enqueue_high_orbit_closure_fused(layout,stream,threads,gx,gy);},"capture forward HIGH graph");
        capture_one(BKOC_GRAPH_REVERSE_LOW,[&]{bucket_enqueue_reverse_low_orbit_closure_fused(layout,stream,threads,gx,gy);},"capture reverse LOW graph");
        capture_one(BKOC_GRAPH_REVERSE_HIGH,[&]{bucket_enqueue_reverse_high_orbit_closure_fused(layout,stream,threads,gx,gy);},"capture reverse HIGH graph");
    }
    void launch(BucketOnePassGraphKind kind){
        ck(cudaGraphLaunch(exec[size_t(kind)],stream),"one-pass graph launch");
    }
    void synchronize(){if(stream)ck(cudaStreamSynchronize(stream),"one-pass graph sync");}
    void release(){
        for(auto&e:exec){if(e)cudaGraphExecDestroy(e);e=nullptr;}
        if(stream)cudaStreamDestroy(stream);stream=nullptr;
    }
};

static void bucket_onepass_graph_sync_devices(
    std::array<BucketOnePassGraphs,BUCKET_NGPU>&graphs,int ngpu
){
    for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"one-pass graph sync set");graphs[g].synchronize();}
}
