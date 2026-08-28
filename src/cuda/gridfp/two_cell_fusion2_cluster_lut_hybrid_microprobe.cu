#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_cluster_hybrid_microprobe_main_unused
#include "two_cell_fusion2_cluster_hybrid_microprobe.cu"
#pragma pop_macro("main")

#include "two_cell_fusion2_cluster_sliced_lut_microprobe.cu"

namespace {

struct LutClusterChoice {
    int cluster = 0;
    Rank local_states = 0;
    Rank dynamic_bytes = 0;
    int max_potential_cluster = 0;
    int max_active_clusters = 0;
    bool ok = false;
};

LutClusterChoice choose_lut_cluster_bucket(
    int outer_ones,
    Rank support_count,
    Rank per_block_limit,
    Rank static_shared,
    int requested_max_cluster,
    const RankTables& rt
) {
    LutClusterChoice out{};
    for (int cluster : {1, 2, 4, 8}) {
        if (cluster > requested_max_cluster || cluster > MAX_CLUSTER_BLOCKS) break;
        const Rank local_states = fusion2_slice_max_owner_states(
            outer_ones, cluster, rt);
        const Rank dynamic_bytes = local_states * sizeof(std::uint32_t);
        if (static_shared + dynamic_bytes > per_block_limit) continue;
        if (dynamic_bytes > static_cast<Rank>(std::numeric_limits<int>::max())) continue;

        ck(cudaFuncSetAttribute(
               two_cell_fusion2_cluster_sliced_lut_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(dynamic_bytes)),
           "cluster LUT hybrid optin shared");

        const unsigned clusters = static_cast<unsigned>(
            std::max<Rank>(1, std::min<Rank>(support_count, 256)));
        cudaLaunchConfig_t config = cluster_config(
            cluster, dynamic_bytes, clusters);

        int max_potential = 0;
        ck(cudaOccupancyMaxPotentialClusterSize(
               &max_potential,
               reinterpret_cast<const void*>(
                   two_cell_fusion2_cluster_sliced_lut_kernel),
               &config),
           "cluster LUT hybrid max potential");
        if (max_potential < cluster) continue;

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
               reinterpret_cast<const void*>(
                   two_cell_fusion2_cluster_sliced_lut_kernel),
               &config),
           "cluster LUT hybrid active clusters");
        if (active <= 0) continue;

        out.cluster = cluster;
        out.local_states = local_states;
        out.dynamic_bytes = dynamic_bytes;
        out.max_potential_cluster = max_potential;
        out.max_active_clusters = active;
        out.ok = true;
        return out;
    }
    return out;
}

void launch_lut_bucket(
    std::uint32_t* d_values,
    const std::uint32_t* d_lut,
    const Rank* d_offset,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank states,
    std::uint32_t mod,
    int* d_error,
    const LutClusterChoice& choice
) {
    const unsigned clusters = static_cast<unsigned>(
        std::max<Rank>(1, std::min<Rank>(
            support_count,
            Rank(std::max(1, choice.max_active_clusters * 8)))));
    cudaLaunchConfig_t config = cluster_config(
        choice.cluster, choice.dynamic_bytes, clusters);
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = choice.cluster;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        two_cell_fusion2_cluster_sliced_lut_kernel,
        d_values, d_lut, d_offset,
        W, start, outer_ones, support_count, states,
        choice.local_states, mod, d_error);
    ck(cudaGetLastError(), "cluster LUT hybrid launch");
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const int requested_max_cluster = argc > 2 ? std::atoi(argv[2]) : 8;
    const Rank shared_kib = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 4
        ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth ||
        requested_max_cluster < 1 || requested_max_cluster > 8 || mod < 3)
        return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const PrimitiveLut host_lut = build_primitive_lut(W, rt);

    if (plan_only) {
        print_capacity_only_plan(
            W, shared_kib * 1024ULL, 4096ULL,
            requested_max_cluster, rt);
        print_cluster_lut_plan(W, rt);
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cluster LUT hybrid device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "cluster LUT hybrid set device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cluster LUT hybrid props");
    if (prop.major < 9) {
        std::cerr << "cluster LUT hybrid requires compute capability >= 9.0\n";
        return 5;
    }
    install_tables(rt);
    install_stationary_tables(st);

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_fusion2_cluster_sliced_lut_kernel),
       "cluster LUT hybrid function attributes");
    const Rank per_block_limit = std::min<Rank>(
        shared_kib * 1024ULL,
        static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);
    if (static_shared >= per_block_limit) return 6;

    std::uint32_t* d_lut = nullptr;
    Rank* d_offset = nullptr;
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)),
       "cluster LUT alloc primitive LUT");
    ck(cudaMalloc(&d_offset, sizeof(host_lut.offset)),
       "cluster LUT alloc offsets");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "cluster LUT copy primitive LUT");
    ck(cudaMemcpy(d_offset, host_lut.offset, sizeof(host_lut.offset),
                  cudaMemcpyHostToDevice), "cluster LUT copy offsets");

    const Rank states = st.total[W];
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 503ULL) % (mod - 1ULL)));

    for (int start = 1; start + 1 <= W - 4; start += 2) {
        const auto reference = fusion2_reference(input, W, start, rt, st, mod);
        std::uint32_t* d_values = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)),
           "cluster LUT alloc values");
        ck(cudaMalloc(&d_error, sizeof(int)), "cluster LUT alloc error");
        ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "cluster LUT copy input");
        ck(cudaMemset(d_error, 0, sizeof(int)), "cluster LUT zero error");

        const int outer_bits = W - 5;
        Rank fused_weight = 0;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank support_count = rt.choose[outer_bits][o];
            const auto choice = choose_lut_cluster_bucket(
                o, support_count, per_block_limit, static_shared,
                requested_max_cluster, rt);
            if (!choice.ok) {
                std::cerr << "FAIL cluster LUT no executable bucket"
                          << " W=" << W << " start=" << start
                          << " outer=" << o << '\n';
                return 7;
            }
            launch_lut_bucket(
                d_values, d_lut, d_offset,
                W, start, o, support_count, states, mod, d_error, choice);
            fused_weight += support_count *
                oneesan::twocell::fusion_block_size(2, o, rt);
            std::cout << "bucket outer=" << o
                      << " cluster=" << choice.cluster
                      << " local_states=" << choice.local_states
                      << " dynamic_shared=" << choice.dynamic_bytes
                      << " active_clusters=" << choice.max_active_clusters
                      << '\n';
        }
        ck(cudaDeviceSynchronize(), "cluster LUT hybrid sync");

        std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
        int error = 0;
        ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "cluster LUT copy output");
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
           "cluster LUT copy error");
        if (error || fused_weight != states || output != reference) {
            std::cerr << "FAIL cluster LUT arithmetic"
                      << " W=" << W << " start=" << start
                      << " error=" << error
                      << " fused=" << fused_weight << '/' << states << '\n';
            return 8;
        }
        cudaFree(d_values);
        cudaFree(d_error);
        std::cout << "two-cell-fusion2-cluster-LUT-hybrid"
                  << " W=" << W
                  << " start=" << start
                  << " static_shared=" << static_shared
                  << " primitive_lut_KiB="
                  << double(host_lut.value.size() * sizeof(std::uint32_t)) / 1024.0
                  << " arithmetic=OK\n";
    }

    cudaFree(d_lut);
    cudaFree(d_offset);
    std::cout << "ALL_OK two_cell_fusion2_cluster_LUT_hybrid=1 W=" << W << '\n';
    return 0;
}
