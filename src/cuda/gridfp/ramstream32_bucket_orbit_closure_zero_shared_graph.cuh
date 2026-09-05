#pragma once
#include "ramstream32_bucket_orbit_closure_graph.cuh"
#include "ramstream32_bucket_orbit_closure_zero_stream.cuh"
#include "ramstream32_bucket_orbit_closure_zero_shared_high.cuh"

struct BucketClosureZeroSharedGraphs {
    cudaStream_t stream=nullptr;
    std::array<cudaGraphExec_t,BKOC_GRAPH_COUNT> exec{};
    int threads=0,gx=0,gy=0;
    template<class F>void capture_one(BucketOnePassGraphKind kind,F&&enqueue,const char*what){
        cudaGraph_t graph=nullptr;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),what);enqueue();ck(cudaStreamEndCapture(stream,&graph),what);
        if(!graph){std::cerr<<"closure-zero-shared graph capture returned null graph "<<what<<'\n';std::exit(551);}
        ck(cudaGraphInstantiate(&exec[size_t(kind)],graph,nullptr,nullptr,0),what);ck(cudaGraphDestroy(graph),what);
    }
    void init(const StorageLayout&layout,int t=256,int x=16,int y=8){
        threads=t;gx=x;gy=y;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"closure-zero-shared graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW,[&]{bucket_enqueue_low_orbit_closure_zero(layout,stream,threads,gx,gy);},"capture closure-zero-shared forward LOW graph");
        capture_one(BKOC_GRAPH_FORWARD_HIGH,[&]{bucket_enqueue_high_orbit_closure_zero_shared(layout,stream,threads,gx,gy);},"capture closure-zero-shared forward HIGH graph");
        capture_one(BKOC_GRAPH_REVERSE_LOW,[&]{bucket_enqueue_reverse_low_split54_zero(layout,stream,threads,gx,gy);},"capture closure-zero-shared reverse LOW graph");
        capture_one(BKOC_GRAPH_REVERSE_HIGH,[&]{bucket_enqueue_reverse_high_split54_zero_shared(layout,stream,threads,gx,gy);},"capture closure-zero-shared reverse HIGH graph");
    }
    void launch(BucketOnePassGraphKind kind){ck(cudaGraphLaunch(exec[size_t(kind)],stream),"closure-zero-shared graph launch");}
    void synchronize(){if(stream)ck(cudaStreamSynchronize(stream),"closure-zero-shared graph sync");}
    void release(){for(auto&e:exec){if(e)cudaGraphExecDestroy(e);e=nullptr;}if(stream)cudaStreamDestroy(stream);stream=nullptr;}
};
static void bucket_closure_zero_shared_graph_sync_devices(std::array<BucketClosureZeroSharedGraphs,BUCKET_NGPU>&graphs,int ngpu){for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"closure-zero-shared graph sync set");graphs[g].synchronize();}}
