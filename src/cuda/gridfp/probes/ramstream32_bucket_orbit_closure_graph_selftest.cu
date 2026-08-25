#define main bucket_reverse_graph_reference_main
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#undef main

#include "../ramstream32_bucket_orbit_closure_preflight.cuh"
#include "../ramstream32_bucket_orbit_closure_graph.cuh"

static void bgoc_run(
    BucketHostGrid&g,bool low,bool reverse,
    const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt,BucketOnePassGraphs&graph
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){
        std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=low?g[fixed][s]:g[s][fixed];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"bgoc H2D");}
        dt.bind_owner(fixed,phy,d);
        BucketOnePassGraphKind k=low?(reverse?BKOC_GRAPH_REVERSE_LOW:BKOC_GRAPH_FORWARD_LOW):(reverse?BKOC_GRAPH_REVERSE_HIGH:BKOC_GRAPH_FORWARD_HIGH);
        graph.launch(k);graph.synchronize();
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=low?g[fixed][s]:g[s][fixed];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"bgoc D2H");}
        bkft_free_slots(d);
    }
}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W,L=LOW_LUT_K;static_assert(W<=12,"graph one-pass bucket selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"bucket-orbit-closure-graph-selftest SKIP no CUDA device\n";return 0;}ck(cudaSetDevice(0),"bgoc set device");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);BucketForwardOrbitClosureAttachHost fa=build_bucket_forward_orbit_closure_attach(layout,bo,bf);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);BucketReverseOrbitClosureAttachHost ra=build_bucket_reverse_orbit_closure_attach_checked(layout,bo,bf,rb,rf);
    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);std::mt19937_64 rng(0x1618c6a9ULL);std::vector<Count>im(ms.size()),ib(bs.size());for(auto&x:im)x=Count(rng()%mod);for(auto&x:ib)x=Count(rng()%mod);
    auto[flm,flb]=gdg_reference_window(W,L,1,mod,ms,bs,mi,di,im,ib);auto[fhm,fhb]=gdg_reference_window(W,W-1,L+1,mod,ms,bs,mi,di,im,ib);auto[rlm,rlb]=bra_reference(1,L,mod,ms,bs,mi,di,im,ib);auto[rhm,rhb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"bgoc modulus");BucketFusedDeviceTables dt;dt.install_metadata(layout,bo,bf);ReverseBucketFusedDeviceTables rt;rt.install(rb,rf);BucketForwardOrbitClosureAttachDeviceTables fat;fat.install(fa);BucketReverseOrbitClosureAttachDeviceTables rat;rat.install(ra);BucketOnePassGraphs graphs;graphs.init(layout,256,4,4);
    auto g1=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);bgoc_run(g1,true,false,layout,phy,dt,graphs);if(!bkft_compare("graph-forward-low",g1,ms,bs,flm,flb,storage,layout,owner,phy))return 10;
    auto g2=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);bgoc_run(g2,false,false,layout,phy,dt,graphs);if(!bkft_compare("graph-forward-high",g2,ms,bs,fhm,fhb,storage,layout,owner,phy))return 11;
    auto g3=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);bgoc_run(g3,true,true,layout,phy,dt,graphs);if(!bkft_compare("graph-reverse-low",g3,ms,bs,rlm,rlb,storage,layout,owner,phy))return 12;
    auto g4=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);bgoc_run(g4,false,true,layout,phy,dt,graphs);if(!bkft_compare("graph-reverse-high",g4,ms,bs,rhm,rhb,storage,layout,owner,phy))return 13;
    std::cout<<"bucket-orbit-closure-graph-selftest OK W="<<W<<" graphs_per_gpu=4 graph_launches_per_row=2 kernels_per_position=1 pm_accum="<<GPU_DIRECT_PM_ACCUM<<"\n";
    graphs.release();rat.release();fat.release();rt.release();dt.release();return 0;
}
