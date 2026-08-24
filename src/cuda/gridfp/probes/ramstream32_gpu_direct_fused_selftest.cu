#define main gpu_direct_gather_reference_main
#include "ramstream32_gpu_direct_gather_selftest.cu"
#undef main

#include "../ramstream32_gpu_direct_gather_cross.cuh"
#include "../ramstream32_gpu_direct_fused.cuh"

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W;
    static_assert(W==LOW_LUT_K+HIGH_LUT_K+1);static_assert(W<=12,"fused selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"gpu-direct-fused-selftest SKIP no CUDA device\n";return 0;}ck(cudaSetDevice(0),"gdf selftest set device");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectCrossHost forward=build_gpu_direct_cross(storage);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused(layout,ordinary,cross);
    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);if(ms.size()!=layout.main_size||bs.size()!=layout.block_size)return 2;
    std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);
    std::vector<Count>init_m(ms.size()),init_b(bs.size());std::mt19937_64 rng(1618);for(auto&x:init_m)x=Count(rng()%mod);for(auto&x:init_b)x=Count(rng()%mod);
    auto[low_m,low_b]=gdg_reference_window(W,LOW_LUT_K,1,mod,ms,bs,mi,di,init_m,init_b);auto[high_m,high_b]=gdg_reference_window(W,W-1,LOW_LUT_K+1,mod,ms,bs,mi,di,init_m,init_b);auto[row_m,row_b]=gdg_reference_window(W,LOW_LUT_K,1,mod,ms,bs,mi,di,high_m,high_b);
    RamCounts ma,ba;ma.alloc(layout.main_size,"gdf main");ba.alloc(layout.block_size,"gdf block");Count*dm=nullptr,*db=nullptr;ck(cudaMalloc(&dm,ma.bytes),"gdf alloc main");ck(cudaMalloc(&db,ba.bytes),"gdf alloc block");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"gdf modulus");
    GpuDirectDeviceTables base;base.install(storage,layout,lowdesc,loworbit,highdirect,forward);GpuDirectGatherDeviceTables ot;ot.install(ordinary);gpu_direct_gather_drop_redundant(base);GpuDirectCrossGatherDeviceTables xt;xt.install(cross);gpu_direct_cross_gather_drop_redundant(base,ot);GpuDirectFusedDeviceTables ft;ft.install(fused);gpu_direct_fused_drop_destination_tables(ot,xt);
    gdg_fill(ma,ba,ms,bs,init_m,init_b,storage,layout);gdg_to_device(dm,db,ma,ba);gpu_direct_run_low_fused(dm,db,layout,256,4,4);gdg_from_device(ma,ba,dm,db);if(!gdg_compare("fused-low",ma,ba,ms,bs,low_m,low_b,storage,layout))return 10;
    gdg_fill(ma,ba,ms,bs,init_m,init_b,storage,layout);gdg_to_device(dm,db,ma,ba);gpu_direct_run_high_fused(dm,db,layout,256,4,4);gdg_from_device(ma,ba,dm,db);if(!gdg_compare("fused-high",ma,ba,ms,bs,high_m,high_b,storage,layout))return 11;
    gdg_fill(ma,ba,ms,bs,init_m,init_b,storage,layout);gdg_to_device(dm,db,ma,ba);gpu_direct_run_high_fused(dm,db,layout,256,4,4);gpu_direct_run_low_fused(dm,db,layout,256,4,4);gdg_from_device(ma,ba,dm,db);if(!gdg_compare("fused-row",ma,ba,ms,bs,row_m,row_b,storage,layout))return 12;
    std::cout<<"gpu-direct-fused-selftest OK W="<<W<<" main="<<ms.size()<<" block="<<bs.size()<<" fused_mib="<<double(fused.bytes())/double(1<<20)<<" low_dst="<<fused.low_dst.size()<<" high_dst="<<fused.high_dst.size()<<" closure_atomic=0 low_launches="<<2*LOW_LUT_K<<" high_launches="<<2*HIGH_LUT_K<<" scratch_bytes=0\n";
    ft.release();xt.release();ot.release();base.release();cudaFree(dm);cudaFree(db);ma.release();ba.release();return 0;
}
