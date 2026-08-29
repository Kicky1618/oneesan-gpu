#pragma push_macro("main")
#undef main
#define main two_cell_forward_forced_main_unused_for_stage
#include "two_cell_fusion2_cluster_lut_forced_microprobe.cu"
#pragma pop_macro("main")

#include "two_cell_snake_stage_api.hpp"

namespace {

int run_forward_stage(
    std::uint32_t* d_values,
    int W,
    int start,
    int cluster_arg,
    std::uint64_t shared_limit_bytes,
    std::uint32_t mod,
    bool forced
) {
    if (!d_values || W < 6 || W > oneesan::twocell::kMaxWidth ||
        start < 0 || start + 1 > W - 4 || mod < 3)
        return 101;
    if (forced) {
        if (cluster_arg != 2 && cluster_arg != 4 && cluster_arg != 8) return 101;
    } else if (cluster_arg != 1 && cluster_arg != 2 &&
               cluster_arg != 4 && cluster_arg != 8) {
        return 101;
    }

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const PrimitiveLut host_lut = build_primitive_lut(W, rt);
    install_tables(rt);
    install_stationary_tables(st);

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "snake forward props");
    if (prop.major < 9) return 102;
    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(
           &attr, two_cell_fusion2_cluster_sliced_lut_kernel),
       "snake forward attrs");
    const Rank per_block_limit = std::min<Rank>(
        static_cast<Rank>(shared_limit_bytes),
        static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);
    if (static_shared >= per_block_limit) return 103;

    std::uint32_t* d_lut = nullptr;
    Rank* d_offset = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)),
       "snake forward alloc LUT");
    ck(cudaMalloc(&d_offset, sizeof(host_lut.offset)),
       "snake forward alloc offsets");
    ck(cudaMalloc(&d_error, sizeof(int)), "snake forward alloc error");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "snake forward copy LUT");
    ck(cudaMemcpy(d_offset, host_lut.offset, sizeof(host_lut.offset),
                  cudaMemcpyHostToDevice), "snake forward copy offsets");
    ck(cudaMemset(d_error, 0, sizeof(int)), "snake forward zero error");

    const Rank states = st.total[W];
    const int outer_bits = W - 5;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank support_count = rt.choose[outer_bits][o];
        const auto choice = forced
            ? forced_lut_choice(cluster_arg, o, support_count,
                                per_block_limit, static_shared, rt)
            : choose_lut_runtime_bucket(o, support_count,
                                        per_block_limit, static_shared,
                                        cluster_arg, rt);
        if (!choice.ok) {
            cudaFree(d_lut); cudaFree(d_offset); cudaFree(d_error);
            return 104;
        }
        launch_lut_runtime_bucket(
            d_values, d_lut, d_offset,
            W, start, o, support_count, states, mod, d_error, choice);
    }
    ck(cudaDeviceSynchronize(), "snake forward sync");
    int error = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "snake forward copy error");
    cudaFree(d_lut);
    cudaFree(d_offset);
    cudaFree(d_error);
    return error ? 105 : 0;
}

} // namespace

extern "C" int oneesan_two_cell_forward2_stage(
    std::uint32_t* d_values, int W, int start, int requested_max_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod
) {
    return run_forward_stage(
        d_values, W, start, requested_max_cluster,
        shared_limit_bytes, mod, false);
}

extern "C" int oneesan_two_cell_forward2_stage_forced(
    std::uint32_t* d_values, int W, int start, int forced_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod
) {
    return run_forward_stage(
        d_values, W, start, forced_cluster,
        shared_limit_bytes, mod, true);
}
