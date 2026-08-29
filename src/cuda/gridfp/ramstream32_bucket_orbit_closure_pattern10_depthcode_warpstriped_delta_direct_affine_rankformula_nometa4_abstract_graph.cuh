#pragma once

#include "ramstream32_bucket_orbit_closure_graph.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_stream.cuh"

static inline int p10dc_rankformula_grid_env(const char* name,int fallback){
    const char* s=std::getenv(name);
    if(!s||!*s)return fallback;
    int v=std::atoi(s);
    if(v<=0){std::cerr<<name<<" must be positive, got "<<s<<'\n';std::exit(775);}
    return v;
}

struct BucketPattern10DepthCodeWarpStripedDeltaDirectAffineRankFormulaNometa4AbstractGraphs {
    cudaStream_t stream = nullptr;
    std::array<cudaGraphExec_t, BKOC_GRAPH_COUNT> exec{};
    template<class F> void capture_one(BucketOnePassGraphKind kind,F&& enqueue,const char* what){cudaGraph_t graph=nullptr;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),what);enqueue();ck(cudaStreamEndCapture(stream,&graph),what);if(!graph){std::cerr<<"rankformula-nometa4-abstract graph capture returned null graph "<<what<<'\n';std::exit(776);}ck(cudaGraphInstantiate(&exec[size_t(kind)],graph,nullptr,nullptr,0),what);ck(cudaGraphDestroy(graph),what);}
    void init(const StorageLayout& layout,int threads=256,int gx=16,int gy=8){
        p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
        p10dc_install_rankformula_abstract_lut();
        const int low_gx=p10dc_rankformula_grid_env("BUCKET_LOW_GRID_X",gx);
        const int low_gy=p10dc_rankformula_grid_env("BUCKET_LOW_GRID_Y",gy);
        const int high_gx=p10dc_rankformula_grid_env("BUCKET_HIGH_GRID_X",gx);
        const int high_gy=p10dc_rankformula_grid_env("BUCKET_HIGH_GRID_Y",gy);
        std::cerr<<"rankformula_grid threads="<<threads
                 <<" low_gx="<<low_gx<<" low_gy="<<low_gy
                 <<" high_gx="<<high_gx<<" high_gy="<<high_gy
                 <<" split_geometry=1\n";
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"rankformula-nometa4-abstract graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW,[&]{bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(layout,stream,threads,low_gx,low_gy);},"capture nometa4 abstract forward LOW");
        capture_one(BKOC_GRAPH_FORWARD_HIGH,[&]{bucket_enqueue_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(layout,stream,threads,high_gx,high_gy);},"capture nometa4 abstract forward HIGH");
        capture_one(BKOC_GRAPH_REVERSE_LOW,[&]{bucket_enqueue_reverse_low_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(layout,stream,threads,low_gx,low_gy);},"capture nometa4 abstract reverse LOW");
        capture_one(BKOC_GRAPH_REVERSE_HIGH,[&]{bucket_enqueue_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(layout,stream,threads,high_gx,high_gy);},"capture nometa4 abstract reverse HIGH");
    }
    void launch(BucketOnePassGraphKind kind){ck(cudaGraphLaunch(exec[size_t(kind)],stream),"rankformula-nometa4-abstract graph launch");}
    void synchronize(){if(stream)ck(cudaStreamSynchronize(stream),"rankformula-nometa4-abstract graph sync");}
    void release(){for(auto&e:exec){if(e)cudaGraphExecDestroy(e);e=nullptr;}if(stream)cudaStreamDestroy(stream);stream=nullptr;}
};

static void bucket_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph_sync_devices(
    std::array<BucketPattern10DepthCodeWarpStripedDeltaDirectAffineRankFormulaNometa4AbstractGraphs,BUCKET_NGPU>& graphs,int ngpu
){for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"rankformula-nometa4-abstract graph sync set");graphs[g].synchronize();}}
