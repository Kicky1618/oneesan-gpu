#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_cluster_lut_runtime_microprobe_main_unused
#include "two_cell_fusion2_cluster_lut_runtime_microprobe.cu"
#pragma pop_macro("main")

namespace {

LutRuntimeChoice forced_lut_choice(
    int cluster,
    int outer_ones,
    Rank support_count,
    Rank per_block_limit,
    Rank static_shared,
    const RankTables& rt
) {
    LutRuntimeChoice out{};
    const Rank local_states = fusion2_slice_max_owner_states(
        outer_ones, cluster, rt);
    const Rank dynamic_bytes = local_states * sizeof(std::uint32_t);
    if (static_shared + dynamic_bytes > per_block_limit) return out;

    ck(cudaFuncSetAttribute(
           two_cell_fusion2_cluster_sliced_lut_kernel,
           cudaFuncAttributeMaxDynamicSharedMemorySize,
           static_cast<int>(dynamic_bytes)),
       "forced DSM optin shared");

    const unsigned clusters = static_cast<unsigned>(
        std::max<Rank>(1, std::min<Rank>(support_count, 256)));
    cudaLaunchConfig_t config = make_lut_cluster_config(
        cluster, dynamic_bytes, clusters);
    int max_potential = 0;
    ck(cudaOccupancyMaxPotentialClusterSize(
           &max_potential,
           reinterpret_cast<const void*>(two_cell_fusion2_cluster_sliced_lut_kernel),
           &config),
       "forced DSM max potential");
    if (max_potential < cluster) return out;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    int active = 0;
    ck(cudaOccupancyMaxActiveClusters(
           &active,
           reinterpret_cast<const void*>(two_cell_fusion2_cluster_sliced_lut_kernel),
           &config),
       "forced DSM active clusters");
    if (active <= 0) return out;

    out.cluster = cluster;
    out.local_states = local_states;
    out.dynamic_bytes = dynamic_bytes;
    out.max_potential_cluster = max_potential;
    out.max_active_clusters = active;
    out.ok = true;
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const int forced_cluster = argc > 2 ? std::atoi(argv[2]) : 2;
    const Rank shared_kib = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 4
        ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > 10 ||
        (forced_cluster != 2 && forced_cluster != 4 && forced_cluster != 8) ||
        mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const PrimitiveLut host_lut = build_primitive_lut(W, rt);
    if (plan_only) {
        std::cout << "forced_DSM_plan W=" << W
                  << " cluster=" << forced_cluster
                  << " primitive_lut_KiB="
                  << double(host_lut.value.size() * sizeof(std::uint32_t)) / 1024.0
                  << " remote_DSM_path_forced=1\n";
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "forced DSM device count");
    if (visible < 1) return 3;
    ck(cudaSetDevice(0), "forced DSM set device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "forced DSM props");
    if (prop.major < 9) {
        std::cerr << "forced DSM requires compute capability >= 9.0\n";
        return 4;
    }
    install_tables(rt);
    install_stationary_tables(st);

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_fusion2_cluster_sliced_lut_kernel),
       "forced DSM function attrs");
    const Rank per_block_limit = std::min<Rank>(
        shared_kib * 1024ULL,
        static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);

    std::uint32_t* d_lut = nullptr;
    Rank* d_offset = nullptr;
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)),
       "forced DSM alloc LUT");
    ck(cudaMalloc(&d_offset, sizeof(host_lut.offset)), "forced DSM alloc offsets");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "forced DSM copy LUT");
    ck(cudaMemcpy(d_offset, host_lut.offset, sizeof(host_lut.offset),
                  cudaMemcpyHostToDevice), "forced DSM copy offsets");

    const Rank states = st.total[W];
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 521ULL) % (mod - 1ULL)));

    for (int start = 1; start + 1 <= W - 4; start += 2) {
        const auto reference = fusion2_reference(input, W, start, rt, st, mod);
        std::uint32_t* d_values = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)),
           "forced DSM alloc values");
        ck(cudaMalloc(&d_error, sizeof(int)), "forced DSM alloc error");
        ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "forced DSM copy input");
        ck(cudaMemset(d_error, 0, sizeof(int)), "forced DSM zero error");

        const int outer_bits = W - 5;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank support_count = rt.choose[outer_bits][o];
            const auto choice = forced_lut_choice(
                forced_cluster, o, support_count,
                per_block_limit, static_shared, rt);
            if (!choice.ok) {
                std::cerr << "SKIP forced DSM unsupported bucket"
                          << " W=" << W << " outer=" << o
                          << " cluster=" << forced_cluster << '\n';
                return 5;
            }
            launch_lut_runtime_bucket(
                d_values, d_lut, d_offset,
                W, start, o, support_count, states, mod, d_error, choice);
        }
        ck(cudaDeviceSynchronize(), "forced DSM sync");

        std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
        int error = 0;
        ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "forced DSM copy output");
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
           "forced DSM copy error");
        if (error || output != reference) {
            std::cerr << "FAIL forced DSM arithmetic"
                      << " W=" << W << " start=" << start
                      << " cluster=" << forced_cluster
                      << " error=" << error << '\n';
            return 6;
        }

        cudaFree(d_values);
        cudaFree(d_error);
        std::cout << "forced-DSM arithmetic=OK"
                  << " W=" << W
                  << " start=" << start
                  << " cluster=" << forced_cluster
                  << " remote_path_exercised=1\n";
    }

    cudaFree(d_lut);
    cudaFree(d_offset);
    return 0;
}
