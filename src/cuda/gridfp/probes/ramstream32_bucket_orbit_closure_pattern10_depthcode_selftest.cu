#pragma push_macro("main")
#undef main
#define main pattern10_depthcode_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_bucket_orbit_closure_pattern10_depthcode_warpctx.cuh"

static void p10dc_run_low(
    BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt,bool rev
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&v=g[fixed][s];if(!v.empty())ck(cudaMemcpy(d[s],v.data(),v.size()*sizeof(Count),cudaMemcpyHostToDevice),"p10dc low H2D");}dt.bind_owner(fixed,phy,d);if(rev)bucket_launch_reverse_low_pattern10_depthcode(layout,256,4,4);else bucket_launch_low_orbit_closure_pattern10_depthcode(layout,256,4,4);ck(cudaDeviceSynchronize(),"p10dc low sync");for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&v=g[fixed][s];if(!v.empty())ck(cudaMemcpy(v.data(),d[s],v.size()*sizeof(Count),cudaMemcpyDeviceToHost),"p10dc low D2H");}bkft_free_slots(d);}
}

enum P10DCTestHighCtx { P10DC_TEST_THREAD=0, P10DC_TEST_RESOLVED=1, P10DC_TEST_WARP=2 };
static void p10dc_launch_high_ctx(const StorageLayout&layout,bool rev,P10DCTestHighCtx ctx){
    dim3 block(256),grid(4,4,unsigned(layout.main_blocks.size()));
    if(ctx==P10DC_TEST_THREAD){
        if(rev)bucket_launch_reverse_high_pattern10_depthcode(layout,256,4,4);
        else bucket_launch_high_orbit_closure_pattern10_depthcode(layout,256,4,4);
        return;
    }
    if(!rev){
        for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
            if(ctx==P10DC_TEST_RESOLVED)bucket_high_orbit_closure_pattern10_depthcode_resolved_kernel<<<grid,block>>>(p);
            else bucket_high_orbit_closure_pattern10_depthcode_warpctx_kernel<<<grid,block>>>(p);
            ck(cudaGetLastError(),ctx==P10DC_TEST_RESOLVED?"p10dc resolved forward high":"p10dc warp forward high");
        }
    }else{
        for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
            if(ctx==P10DC_TEST_RESOLVED)bucket_reverse_high_pattern10_depthcode_resolved_kernel<<<grid,block>>>(p);
            else bucket_reverse_high_pattern10_depthcode_warpctx_kernel<<<grid,block>>>(p);
            ck(cudaGetLastError(),ctx==P10DC_TEST_RESOLVED?"p10dc resolved reverse high":"p10dc warp reverse high");
        }
    }
}
static void p10dc_run_high_ctx(
    BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt,bool rev,P10DCTestHighCtx ctx
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&v=g[s][fixed];if(!v.empty())ck(cudaMemcpy(d[s],v.data(),v.size()*sizeof(Count),cudaMemcpyHostToDevice),"p10dc high H2D");}dt.bind_owner(fixed,phy,d);p10dc_launch_high_ctx(layout,rev,ctx);ck(cudaDeviceSynchronize(),"p10dc high ctx sync");for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&v=g[s][fixed];if(!v.empty())ck(cudaMemcpy(v.data(),d[s],v.size()*sizeof(Count),cudaMemcpyDeviceToHost),"p10dc high D2H");}bkft_free_slots(d);}
}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W,L=LOW_LUT_K;static_assert(W<=12,"pattern10 depthcode selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"bucket-closure-pattern10-depthcode-selftest SKIP no CUDA device\n";return 0;}ck(cudaSetDevice(0),"p10dc set device");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    auto fh=build_bucket_forward_pattern10_depthcode_placeholder(layout,bo,bf);auto rh=build_bucket_reverse_pattern10_depthcode_zero_checked(layout,bo,bf,rb,rf);

    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);std::mt19937_64 rng(0x1618dc0deULL);std::vector<Count>im(ms.size()),ib(bs.size());for(auto&x:im)x=Count(rng()%mod);for(auto&x:ib)x=Count(rng()%mod);
    auto[flm,flb]=gdg_reference_window(W,L,1,mod,ms,bs,mi,di,im,ib);auto[fhm,fhb]=gdg_reference_window(W,W-1,L+1,mod,ms,bs,mi,di,im,ib);auto[rlm,rlb]=bra_reference(1,L,mod,ms,bs,mi,di,im,ib);auto[rhm,rhb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"p10dc modulus");BucketFusedDeviceTables dt;dt.install_metadata(layout,bo,bf);BucketForwardPattern10DepthCodeDeviceTables fdt;fdt.install(fh);BucketReversePattern10DepthCodeDeviceTables rdt;rdt.install(rh);
    auto g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_low(g,layout,phy,dt,false);if(!bkft_compare("pattern10-depthcode-forward-low",g,ms,bs,flm,flb,storage,layout,owner,phy))return 10;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_high_ctx(g,layout,phy,dt,false,P10DC_TEST_THREAD);if(!bkft_compare("pattern10-depthcode-forward-high-thread",g,ms,bs,fhm,fhb,storage,layout,owner,phy))return 11;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_low(g,layout,phy,dt,true);if(!bkft_compare("pattern10-depthcode-reverse-low",g,ms,bs,rlm,rlb,storage,layout,owner,phy))return 12;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_high_ctx(g,layout,phy,dt,true,P10DC_TEST_THREAD);if(!bkft_compare("pattern10-depthcode-reverse-high-thread",g,ms,bs,rhm,rhb,storage,layout,owner,phy))return 13;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_high_ctx(g,layout,phy,dt,false,P10DC_TEST_RESOLVED);if(!bkft_compare("pattern10-depthcode-forward-high-resolved",g,ms,bs,fhm,fhb,storage,layout,owner,phy))return 14;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_high_ctx(g,layout,phy,dt,true,P10DC_TEST_RESOLVED);if(!bkft_compare("pattern10-depthcode-reverse-high-resolved",g,ms,bs,rhm,rhb,storage,layout,owner,phy))return 15;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_high_ctx(g,layout,phy,dt,false,P10DC_TEST_WARP);if(!bkft_compare("pattern10-depthcode-forward-high-warp",g,ms,bs,fhm,fhb,storage,layout,owner,phy))return 16;
    g=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);p10dc_run_high_ctx(g,layout,phy,dt,true,P10DC_TEST_WARP);if(!bkft_compare("pattern10-depthcode-reverse-high-warp",g,ms,bs,rhm,rhb,storage,layout,owner,phy))return 17;
    std::cout<<"bucket-closure-pattern10-depthcode-selftest OK W="<<W<<" mode="<<rh.codebook.mode<<" codebook_bytes="<<rh.codebook.bytes()<<" sidecar_bytes_per_orbit=0 temporary_depth_bytes=0 decode_unrank=0 payload_masks=1 high_ctx=thread,resolved,warp\n";rdt.release();fdt.release();dt.release();return 0;
}
