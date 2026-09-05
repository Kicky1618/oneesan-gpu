#define main ramstream_cpu_low_selftest_reference_main
#include "ramstream32_cpu_low_selftest.cu"
#undef main

#include "../ramstream32_gpu_direct.cuh"

static void copy_factor_to_device(
    Count* dmain, Count* dblock,
    const RamCounts& main_auth, const RamCounts& block_auth
) {
    ck(cudaMemcpy(dmain, main_auth.ptr, main_auth.bytes, cudaMemcpyHostToDevice),
       "gpu direct selftest main H2D");
    ck(cudaMemcpy(dblock, block_auth.ptr, block_auth.bytes, cudaMemcpyHostToDevice),
       "gpu direct selftest block H2D");
}

static void copy_factor_from_device(
    RamCounts& main_auth, RamCounts& block_auth,
    const Count* dmain, const Count* dblock
) {
    ck(cudaMemcpy(main_auth.ptr, dmain, main_auth.bytes, cudaMemcpyDeviceToHost),
       "gpu direct selftest main D2H");
    ck(cudaMemcpy(block_auth.ptr, dblock, block_auth.bytes, cudaMemcpyDeviceToHost),
       "gpu direct selftest block D2H");
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "GPU direct selftest intentionally uses a small width");

    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "gpu-direct-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "gpu direct selftest set device");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectCrossHost cross = build_gpu_direct_cross(storage);

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size || block_states.size() != layout.block_size) {
        std::cerr << "FAIL gpu-direct state/layout size mismatch\n";
        return 2;
    }

    std::unordered_map<MateID, size_t> mi, di;
    mi.reserve(main_states.size() * 2);
    di.reserve(block_states.size() * 2);
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    std::vector<Count> init_m(main_states.size()), init_d(block_states.size());
    std::mt19937_64 rng(1618);
    for (auto& v : init_m) v = Count(rng() % mod);
    for (auto& v : init_d) v = Count(rng() % mod);

    auto [low_rm, low_rd] = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, init_m, init_d);
    auto [high_rm, high_rd] = reference_window(
        W, W - 1, LOW_LUT_K + 1, mod,
        main_states, block_states, mi, di, init_m, init_d);
    auto [row_rm, row_rd] = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, high_rm, high_rd);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "gpu direct selftest main");
    block_auth.alloc(layout.block_size, "gpu direct selftest block");

    Count* dmain = nullptr;
    Count* dblock = nullptr;
    ck(cudaMalloc(&dmain, main_auth.bytes), "gpu direct selftest alloc main");
    ck(cudaMalloc(&dblock, block_auth.bytes), "gpu direct selftest alloc block");
    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "gpu direct selftest modulus");

    GpuDirectDeviceTables tables;
    tables.install(storage, layout, lowdesc, loworbit, highdirect, cross);

    fill_factor(main_auth, block_auth, main_states, block_states,
                init_m, init_d, storage, layout);
    copy_factor_to_device(dmain, dblock, main_auth, block_auth);
    gpu_direct_run_low(dmain, dblock, layout, 256, 4, 4);
    copy_factor_from_device(main_auth, block_auth, dmain, dblock);
    if (!compare_factor("gpu-direct-low", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd,
                        storage, layout)) return 10;

    fill_factor(main_auth, block_auth, main_states, block_states,
                init_m, init_d, storage, layout);
    copy_factor_to_device(dmain, dblock, main_auth, block_auth);
    gpu_direct_run_high(dmain, dblock, layout, 256, 4, 4);
    copy_factor_from_device(main_auth, block_auth, dmain, dblock);
    if (!compare_factor("gpu-direct-high", main_auth, block_auth,
                        main_states, block_states, high_rm, high_rd,
                        storage, layout)) return 11;

    fill_factor(main_auth, block_auth, main_states, block_states,
                init_m, init_d, storage, layout);
    copy_factor_to_device(dmain, dblock, main_auth, block_auth);
    gpu_direct_run_high(dmain, dblock, layout, 256, 4, 4);
    gpu_direct_run_low(dmain, dblock, layout, 256, 4, 4);
    copy_factor_from_device(main_auth, block_auth, dmain, dblock);
    if (!compare_factor("gpu-direct-row", main_auth, block_auth,
                        main_states, block_states, row_rm, row_rd,
                        storage, layout)) return 12;

    double meta_mib = double(
        lowdesc.main_desc.size() * sizeof(uint32_t)
        + loworbit.rec.size() * sizeof(uint64_t)
        + highdirect.orbit_ops.size() * sizeof(CpuHighOrbitOp)
        + highdirect.closure_ops.size() * sizeof(CpuHighClosureOp)
        + cross.high_rank.size() * sizeof(uint32_t)
        + cross.low_rank.size() * sizeof(uint32_t)) / double(1 << 20);

    std::cout << "gpu-direct-selftest OK W=" << W
              << " main=" << main_states.size()
              << " block=" << block_states.size()
              << " metadata_mib=" << meta_mib
              << " low_launches=" << (2 * LOW_LUT_K)
              << " high_launches=" << (2 * HIGH_LUT_K)
              << " scratch_bytes=0"
              << '\n';

    tables.release();
    cudaFree(dmain);
    cudaFree(dblock);
    main_auth.release();
    block_auth.release();
    return 0;
}
