#pragma push_macro("main")
#undef main
#define main two_cell_reverse_runtime_main_unused_for_stage
#include "two_cell_reverse_fusion2_cluster_runtime_microprobe.cu"
#pragma pop_macro("main")

#include "two_cell_snake_stage_api.hpp"

namespace {

int run_reverse_stage(
    std::uint32_t* d_values,
    int W,
    int start,
    int cluster_arg,
    std::uint64_t shared_limit_bytes,
    std::uint32_t mod,
    bool forced
) {
    if (!d_values || W < 6 || W > oneesan::twocell::kMaxWidth ||
        start < 1 || start + 1 > W - 4 || mod < 3)
        return 121;
    if (forced) {
        if (cluster_arg != 2 && cluster_arg != 4 && cluster_arg != 8) return 121;
    } else if (cluster_arg != 1 && cluster_arg != 2 &&
               cluster_arg != 4 && cluster_arg != 8) {
        return 121;
    }

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const ReversePrimitiveLut host_lut = build_reverse_primitive_lut(W, rt);
    install_tables(rt);
    install_stationary_tables(st);

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "snake reverse props");
    if (prop.major < 9) return 122;
    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(
           &attr, two_cell_reverse_fusion2_cluster_sliced_kernel),
       "snake reverse attrs");
    const Rank per_block_limit = std::min<Rank>(
        static_cast<Rank>(shared_limit_bytes),
        static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);
    if (static_shared >= per_block_limit) return 123;

    std::uint32_t *d_left = nullptr, *d_reflection = nullptr;
    Rank* d_offset = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_left,
                  host_lut.label_left.size() * sizeof(std::uint32_t)),
       "snake reverse alloc left LUT");
    ck(cudaMalloc(&d_reflection,
                  host_lut.reflection_meta.size() * sizeof(std::uint32_t)),
       "snake reverse alloc reflection LUT");
    ck(cudaMalloc(&d_offset, sizeof(host_lut.offset)),
       "snake reverse alloc offsets");
    ck(cudaMalloc(&d_error, sizeof(int)), "snake reverse alloc error");
    ck(cudaMemcpy(d_left, host_lut.label_left.data(),
                  host_lut.label_left.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice),
       "snake reverse copy left LUT");
    ck(cudaMemcpy(d_reflection, host_lut.reflection_meta.data(),
                  host_lut.reflection_meta.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice),
       "snake reverse copy reflection LUT");
    ck(cudaMemcpy(d_offset, host_lut.offset, sizeof(host_lut.offset),
                  cudaMemcpyHostToDevice),
       "snake reverse copy offsets");
    ck(cudaMemset(d_error, 0, sizeof(int)), "snake reverse zero error");

    const Rank states = st.total[W];
    const int outer_bits = W - 5;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank support_count = rt.choose[outer_bits][o];
        const auto choice = forced
            ? forced_reverse_choice(cluster_arg, o, support_count,
                                    per_block_limit, static_shared, rt)
            : choose_reverse_runtime_bucket(o, support_count,
                                            per_block_limit, static_shared,
                                            cluster_arg, rt);
        if (!choice.ok) {
            cudaFree(d_left); cudaFree(d_reflection);
            cudaFree(d_offset); cudaFree(d_error);
            return 124;
        }
        launch_forced_reverse_bucket(
            d_values, d_left, d_reflection, d_offset,
            W, start, o, support_count, states, mod, d_error, choice);
    }
    ck(cudaDeviceSynchronize(), "snake reverse sync");
    int error = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "snake reverse copy error");
    cudaFree(d_left);
    cudaFree(d_reflection);
    cudaFree(d_offset);
    cudaFree(d_error);
    return error ? 125 : 0;
}

} // namespace

extern "C" int oneesan_two_cell_reverse2_stage(
    std::uint32_t* d_values, int W, int start, int requested_max_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod
) {
    return run_reverse_stage(
        d_values, W, start, requested_max_cluster,
        shared_limit_bytes, mod, false);
}

extern "C" int oneesan_two_cell_reverse2_stage_forced(
    std::uint32_t* d_values, int W, int start, int forced_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod
) {
    return run_reverse_stage(
        d_values, W, start, forced_cluster,
        shared_limit_bytes, mod, true);
}
