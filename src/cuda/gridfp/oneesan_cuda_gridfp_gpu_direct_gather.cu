#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN

#include "ramstream32_cpu_low_sparse.hpp"
#include "ramstream32_cpu_high.hpp"
#include "ramstream32_cpu_high_direct.hpp"
#include "ramstream32_gpu_direct.cuh"
#include "ramstream32_gpu_direct_gather.cuh"

static bool gdg_has_arg(int argc, char** argv, const char* needle) {
    for (int i=1;i<argc;++i) if (std::strcmp(argv[i],needle)==0) return true;
    return false;
}
static double gdg_seconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2],nullptr,10)) : 4294967291u;
    int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    int grid_x = argc > 4 ? std::atoi(argv[4]) : 16;
    int grid_y = argc > 5 ? std::atoi(argv[5]) : 8;
    bool plan_only = gdg_has_arg(argc,argv,"--plan-only");
    int W=n+1;
    if (W!=TARGET_W || n<2 || W>MAXW) return 1;
    if constexpr (LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W) return 1;
    if (threads<=0 || threads>1024 || grid_x<=0 || grid_y<=0) return 2;

    auto prep0=std::chrono::steady_clock::now();
    build_full_dp();
    G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectCrossHost cross=build_gpu_direct_cross(storage);
    GpuDirectGatherHost gather=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    double prepare_s=gdg_seconds(prep0);

    size_t auth_bytes=size_t(layout.main_size+layout.block_size)*sizeof(Count);
    size_t orbit_bytes=loworbit.rec.size()*sizeof(uint64_t)
        + highdirect.orbit_ops.nn.size()*sizeof(CpuHighOrbitOp)
        + highdirect.orbit_ops.nrnl.size()*sizeof(CpuHighOrbitOp);
    size_t cross_stream_bytes=highdirect.closure_ops.cross.size()*sizeof(CpuHighClosureOp);
    size_t offset_bytes=(highdirect.orbit_off.nn.size()+highdirect.orbit_off.nrnl.size()
        + highdirect.closure_off.cross.size())*sizeof(uint32_t);
    size_t cross_rank_bytes=(cross.high_rank.size()+cross.low_rank.size())*sizeof(uint32_t);
    size_t resident_meta=orbit_bytes+cross_stream_bytes+offset_bytes+cross_rank_bytes+gather.bytes();
    size_t resident_bytes=auth_bytes+resident_meta;

    // base.install transiently uploads metadata that gather execution later
    // drops. Account for the temporary peak until a gather-specific installer
    // is split out.
    size_t transient_extra=(lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)
        + highdirect.closure_ops.block.size()*sizeof(CpuHighClosureOp)
        + highdirect.closure_off.block.size()*sizeof(uint32_t);
    size_t transient_peak=resident_bytes+transient_extra;

    if (plan_only) {
        std::cout
            << "backend=gridfp-gpu-direct-gather-v0.1-plan"
            << " n=" << n
            << " main_states=" << layout.main_size
            << " blocked_states=" << layout.block_size
            << " authoritative_gib=" << double(auth_bytes)/double(1ULL<<30)
            << " gather_mib=" << double(gather.bytes())/double(1ULL<<20)
            << " resident_metadata_mib=" << double(resident_meta)/double(1ULL<<20)
            << " resident_total_gib=" << double(resident_bytes)/double(1ULL<<30)
            << " transient_peak_gib=" << double(transient_peak)/double(1ULL<<30)
            << " low_dst=" << gather.low_dst.size()
            << " low_edges=" << gather.low_src.size()
            << " low_cross=" << gather.low_cross.size()
            << " low_max_indegree=" << gather.low_max_indegree
            << " high_dst=" << gather.high_dst.size()
            << " high_edges=" << gather.high_src.size()
            << " high_max_indegree=" << gather.high_max_indegree
            << " low_launches_per_row=" << 3*LOW_LUT_K
            << " high_launches_per_row=" << 3*HIGH_LUT_K
            << " ordinary_closure_atomic=0"
            << " cross_closure_atomic=1"
            << " scratch_bytes=0"
            << " prepare_s=" << prepare_s << '\n';
        return 0;
    }

    int visible=0;
    ck(cudaGetDeviceCount(&visible),"gdg device count");
    if (visible<1) return 3;
    ck(cudaSetDevice(0),"gdg set device");
    size_t free_bytes=0,total_bytes=0;
    ck(cudaMemGetInfo(&free_bytes,&total_bytes),"gdg mem info");
    if (transient_peak>free_bytes) {
        std::cerr << "insufficient HBM: need_gib="
                  << double(transient_peak)/double(1ULL<<30)
                  << " free_gib=" << double(free_bytes)/double(1ULL<<30) << '\n';
        return 4;
    }

    Count *dmain=nullptr,*dblock=nullptr;
    ck(cudaMalloc(&dmain,size_t(layout.main_size)*sizeof(Count)),"gdg alloc main");
    ck(cudaMalloc(&dblock,size_t(layout.block_size)*sizeof(Count)),"gdg alloc block");
    ck(cudaMemset(dmain,0,size_t(layout.main_size)*sizeof(Count)),"gdg zero main");
    ck(cudaMemset(dblock,0,size_t(layout.block_size)*sizeof(Count)),"gdg zero block");
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"gdg modulus");

    GpuDirectDeviceTables base;
    base.install(storage,layout,lowdesc,loworbit,highdirect,cross);
    GpuDirectGatherDeviceTables gt;
    gt.install(gather);
    gpu_direct_gather_drop_redundant(base);

    MateID init=MateID(R)<<(2*(W-1));
    Code init_rank=storage_rank_main_host(init,storage,layout);
    Count one=1;
    ck(cudaMemcpy(dmain+init_rank,&one,sizeof(one),cudaMemcpyHostToDevice),"gdg init");

    double high_s=0,low_s=0;
    auto wall0=std::chrono::steady_clock::now();
    for (int row=0;row<W;++row) {
        auto t=std::chrono::steady_clock::now();
        gpu_direct_run_high_gather(dmain,dblock,layout,threads,grid_x,grid_y);
        high_s+=gdg_seconds(t);
        t=std::chrono::steady_clock::now();
        gpu_direct_run_low_gather(dmain,dblock,layout,threads,grid_x,grid_y);
        low_s+=gdg_seconds(t);
        std::cerr << "row " << row+1 << '/' << W
                  << " high_s=" << high_s << " low_s=" << low_s << '\n';
    }
    double wall_s=gdg_seconds(wall0);

    Code final_rank=storage_rank_main_host(MateID(R),storage,layout);
    Count answer=0;
    ck(cudaMemcpy(&answer,dmain+final_rank,sizeof(answer),cudaMemcpyDeviceToHost),"gdg answer");
    std::cout
        << "backend=gridfp-gpu-direct-gather-v0.1"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " authoritative_gib=" << double(auth_bytes)/double(1ULL<<30)
        << " resident_metadata_mib=" << double(resident_meta)/double(1ULL<<20)
        << " gather_mib=" << double(gather.bytes())/double(1ULL<<20)
        << " low_max_indegree=" << gather.low_max_indegree
        << " high_max_indegree=" << gather.high_max_indegree
        << " threads=" << threads
        << " grid_x=" << grid_x
        << " grid_y=" << grid_y
        << " high_s=" << high_s
        << " low_s=" << low_s
        << " prepare_s=" << prepare_s
        << " wall_s=" << wall_s
        << " ordinary_closure_atomic=0"
        << " cross_closure_atomic=1"
        << " scratch_bytes=0\n";

    gt.release();
    base.release();
    cudaFree(dmain); cudaFree(dblock);
    return 0;
}
