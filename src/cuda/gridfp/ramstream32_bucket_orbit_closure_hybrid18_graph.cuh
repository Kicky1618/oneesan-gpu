#pragma once
#include "ramstream32_bucket_orbit_closure_graph.cuh"
#include "ramstream32_bucket_orbit_closure_packed18_stream.cuh"
#include "ramstream32_bucket_reverse_split18_stream.cuh"

struct BucketHybrid18Graphs {
    cudaStream_t stream=nullptr;
    std::array<cudaGraphExec_t,BKOC_GRAPH_COUNT> exec{};
    int threads=0,gx=0,gy=0;
    template<class F>void capture_one(BucketOnePassGraphKind kind,F&&enqueue,const char*what){cudaGraph_t graph=nullptr;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),what);enqueue();ck(cudaStreamEndCapture(stream,&graph),what);if(!graph){std::cerr<<"hybrid18 graph capture returned null graph "<<what<<'\n';std::exit(450);}ck(cudaGraphInstantiate(&exec[size_t(kind)],graph,nullptr,nullptr,0),what);ck(cudaGraphDestroy(graph),what);}
    void init(const StorageLayout&layout,int t=256,int x=16,int y=8){threads=t;gx=x;gy=y;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"hybrid18 graph stream");capture_one(BKOC_GRAPH_FORWARD_LOW,[&]{bucket_enqueue_low_orbit_closure18(layout,stream,threads,gx,gy);},"capture hybrid18 forward LOW graph");capture_one(BKOC_GRAPH_FORWARD_HIGH,[&]{bucket_enqueue_high_orbit_closure18(layout,stream,threads,gx,gy);},"capture hybrid18 forward HIGH graph");capture_one(BKOC_GRAPH_REVERSE_LOW,[&]{bucket_enqueue_reverse_low_split18(layout,stream,threads,gx,gy);},"capture hybrid18 reverse LOW graph");capture_one(BKOC_GRAPH_REVERSE_HIGH,[&]{bucket_enqueue_reverse_high_split18(layout,stream,threads,gx,gy);},"capture hybrid18 reverse HIGH graph");}
    void launch(BucketOnePassGraphKind kind){ck(cudaGraphLaunch(exec[size_t(kind)],stream),"hybrid18 graph launch");}
    void synchronize(){if(stream)ck(cudaStreamSynchronize(stream),"hybrid18 graph sync");}
    void release(){for(auto&e:exec){if(e)cudaGraphExecDestroy(e);e=nullptr;}if(stream)cudaStreamDestroy(stream);stream=nullptr;}
};
static void bucket_hybrid18_graph_sync_devices(std::array<BucketHybrid18Graphs,BUCKET_NGPU>&graphs,int ngpu){for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"hybrid18 graph sync set");graphs[g].synchronize();}}
