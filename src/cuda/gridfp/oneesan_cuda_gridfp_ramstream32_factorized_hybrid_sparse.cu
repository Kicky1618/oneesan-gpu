#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN

#include "ramstream32_cpu_low_sparse.hpp"

static void hybrid_sparse_release_dense_host(StorageFactorHost& storage) {
    storage.low_packed_rank.clear(); storage.low_packed_rank.shrink_to_fit();
    storage.high_packed_rank.clear(); storage.high_packed_rank.shrink_to_fit();
    storage.low_all_codes.clear(); storage.low_all_codes.shrink_to_fit();
    storage.high_all_codes.clear(); storage.high_all_codes.shrink_to_fit();
    G_FACTOR.low_packed_rank.clear(); G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear(); G_FACTOR.high_packed_rank.shrink_to_fit();
    G_FACTOR.low_all_codes.clear(); G_FACTOR.low_all_codes.shrink_to_fit();
    G_FACTOR.high_all_codes.clear(); G_FACTOR.high_all_codes.shrink_to_fit();
    G_FACTOR.high_main_base.clear(); G_FACTOR.high_main_base.shrink_to_fit();
    G_FACTOR.high_block_base.clear(); G_FACTOR.high_block_base.shrink_to_fit();
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int gpu_target_mib = argc > 3 ? std::atoi(argv[3]) : 12288;
    int cpu_workers = argc > 4 ? std::max(1, std::atoi(argv[4])) : 4;
    bool plan_only = argc > 5 && std::strcmp(argv[5], "--plan-only") == 0;
    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    auto meta0 = std::chrono::steady_clock::now();
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    double meta_build_s = ram_seconds_since(meta0);

    WindowPlan high_wp = make_direct2d_window(true);
    WindowPlan low_wp = make_direct2d_window(false);
    size_t gpu_target = size_t(gpu_target_mib) << 20;
    if (high_wp.max_bytes > gpu_target) {
        std::cerr << "hybrid-sparse high window does not fit: high_gib="
                  << double(high_wp.max_bytes) / double(1ULL << 30)
                  << " target_gib=" << double(gpu_target) / double(1ULL << 30) << '\n';
        return 4;
    }

    auto high_jobs = make_direct2d_jobs(high_wp);
    auto cpu_jobs = make_cpu_low_jobs(W, low_wp);
    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = storage_rank_main_host(init, storage, layout);
    Code answer_rank = storage_rank_main_host(MateID(R), storage, layout);

    double highdesc_mib = double((highdesc.main_desc.size()+highdesc.block_desc.size())*sizeof(uint32_t))/(1<<20);
    double sparse_orbit_mib = double(sparse.orbit_ops.size()*sizeof(CpuLowSparseOrbitOp))/(1<<20);
    double sparse_closure_mib = double(sparse.closure_ops.size()*sizeof(uint64_t))/(1<<20);
    double sparse_offsets_mib = double((sparse.orbit_off.size()+sparse.closure_off.size())*sizeof(uint32_t))/(1<<20);
    double sparse_cross_mib = double(sparse.high_cross_rank.size()*sizeof(uint16_t))/(1<<20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size()+G_FACTOR.low_mask_off.size()
        +G_FACTOR.high_mask_codes.size()+G_FACTOR.high_mask_off.size())*sizeof(uint32_t))/(1<<20);
    double dense_host_release_mib = double((storage.low_packed_rank.size()+storage.high_packed_rank.size()
        +G_FACTOR.low_packed_rank.size()+G_FACTOR.high_packed_rank.size())*sizeof(uint32_t))/(1<<20);
    double dense_cpu_meta_mib = double((lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)
        + orbit.rec.size()*sizeof(uint64_t))/(1<<20);

    double auth_bytes = double(layout.main_size + layout.block_size) * sizeof(Count);
    double pcie_bytes = 2.0 * W * auth_bytes;
    double pcie_tib = pcie_bytes / double(1ULL << 40);
    double pcie_50gib_s = pcie_bytes / (50.0 * double(1ULL << 30));

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-sparse-v4.5-plan"
            << " n=" << n
            << " gpu_high_desc_mib=" << highdesc_mib
            << " gpu_mask_mib=" << mask_mib
            << " cpu_sparse_orbit_mib=" << sparse_orbit_mib
            << " cpu_sparse_closure_mib=" << sparse_closure_mib
            << " cpu_sparse_offsets_mib=" << sparse_offsets_mib
            << " cpu_sparse_cross_rank_mib=" << sparse_cross_mib
            << " cpu_dense_meta_replaced_mib=" << dense_cpu_meta_mib
            << " dense_host_release_mib=" << dense_host_release_mib
            << " meta_build_s=" << meta_build_s
            << " gpu_high_window_max_gib=" << double(high_wp.max_bytes)/double(1ULL<<30)
            << " cpu_workers=" << cpu_workers
            << " cpu_scratch_gib=0"
            << " pcie_tib_per_residue=" << pcie_tib
            << " pcie_50gib_s=" << pcie_50gib_s
            << '\n';
        return 0;
    }

    int visible=0; ck(cudaGetDeviceCount(&visible),"cudaGetDeviceCount");
    if(visible<1)return 2;
    ck(cudaSetDevice(0),"cudaSetDevice");

    BidescMaskDeviceTables mask_tables; mask_tables.install(G_FACTOR);
    HighDescDeviceTables highdesc_tables; highdesc_tables.install(highdesc);

    hybrid_sparse_release_dense_host(storage);
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    orbit.rec.clear(); orbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size,"mmap hybrid-sparse main");
    block_auth.alloc(layout.block_size,"mmap hybrid-sparse block");
    main_auth.ptr[init_rank]=1;

    Direct2DCtx gpu; gpu.init(mod);
    CpuLowSparsePool cpu(cpu_workers);
    int gpu_threads=256;
    auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        for(const auto& job:high_jobs) if(job.work)
            process_group_bidesc_compact(gpu,main_auth,block_auth,storage,layout,W,high_wp,job.g,gpu_threads);
        cpu.run(cpu_jobs,main_auth,block_auth,storage,layout,sparse,mod);
        std::cerr<<"row "<<row+1<<'/'<<W<<" gpu_groups="<<gpu.groups<<" cpu_groups="<<cpu.groups()<<'\n';
    }

    double wall_s=ram_seconds_since(wall0);
    Count answer=main_auth.ptr[answer_rank];
    std::cout
        << "backend=gridfp-ramstream32-factorized-hybrid-sparse-v4.5"
        << " n="<<n<<" residue="<<answer<<" modulus="<<mod
        << " gpu_high_desc_mib="<<highdesc_mib<<" gpu_mask_mib="<<mask_mib
        << " cpu_sparse_orbit_mib="<<sparse_orbit_mib
        << " cpu_sparse_closure_mib="<<sparse_closure_mib
        << " cpu_sparse_cross_rank_mib="<<sparse_cross_mib
        << " gpu_groups="<<gpu.groups<<" cpu_groups="<<cpu.groups()
        << " cpu_workers="<<cpu_workers<<" cpu_scratch_gib=0"
        << " h2d_s="<<gpu.h2d_s<<" gpu_kernel_s="<<gpu.kernel_s<<" d2h_s="<<gpu.d2h_s
        << " cpu_kernel_sum_s="<<cpu.kernel_s()<<" cpu_wall_s="<<cpu.wall_s
        << " pcie_tib_per_residue="<<pcie_tib<<" wall_s="<<wall_s<<'\n';

    gpu.destroy(); highdesc_tables.release(); mask_tables.release();
    main_auth.release(); block_auth.release();
    return 0;
}
