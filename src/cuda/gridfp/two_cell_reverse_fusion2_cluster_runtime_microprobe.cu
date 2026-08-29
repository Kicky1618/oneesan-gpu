#pragma push_macro("main")
#undef main
#define main two_cell_reverse_fusion2_cluster_forced_microprobe_main_unused
#include "two_cell_reverse_fusion2_cluster_forced_microprobe.cu"
#pragma pop_macro("main")

namespace {

ReverseClusterChoice choose_reverse_runtime_bucket(
    int outer_ones,
    Rank support_count,
    Rank per_block_limit,
    Rank static_shared,
    int requested_max_cluster,
    const RankTables& rt
) {
    ReverseClusterChoice out{};
    for (int cluster : {1, 2, 4, 8}) {
        if (cluster > requested_max_cluster || cluster > REVERSE_CLUSTER_MAX) break;
        const Rank local_states = reverse_slice_max_owner_states(
            outer_ones, cluster, rt);
        const Rank dynamic_bytes = local_states * sizeof(std::uint32_t);
        if (static_shared + dynamic_bytes > per_block_limit ||
            dynamic_bytes > static_cast<Rank>(std::numeric_limits<int>::max()))
            continue;

        ck(cudaFuncSetAttribute(
               two_cell_reverse_fusion2_cluster_sliced_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(dynamic_bytes)),
           "reverse runtime optin shared");

        const unsigned clusters = static_cast<unsigned>(
            std::max<Rank>(1, std::min<Rank>(support_count, 256)));
        cudaLaunchConfig_t config = make_reverse_cluster_config(
            cluster, dynamic_bytes, clusters);
        int max_potential = 0;
        ck(cudaOccupancyMaxPotentialClusterSize(
               &max_potential,
               reinterpret_cast<const void*>(
                   two_cell_reverse_fusion2_cluster_sliced_kernel),
               &config),
           "reverse runtime max potential");
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
                   two_cell_reverse_fusion2_cluster_sliced_kernel),
               &config),
           "reverse runtime active clusters");
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
        (requested_max_cluster != 1 && requested_max_cluster != 2 &&
         requested_max_cluster != 4 && requested_max_cluster != 8) ||
        mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const ReversePrimitiveLut host_lut = build_reverse_primitive_lut(W, rt);

    if (plan_only) {
        const int outer_bits = W - 5;
        Rank total = 0, fused = 0;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank support_count = rt.choose[outer_bits][o];
            const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
            total += support_count * n;
            for (int c : {1, 2, 4, 8}) {
                if (c > requested_max_cluster) break;
                if (reverse_slice_max_owner_states(o, c, rt) * sizeof(std::uint32_t) +
                        4096ULL <= shared_kib * 1024ULL) {
                    fused += support_count * n;
                    break;
                }
            }
        }
        std::cout << "reverse-runtime-plan"
                  << " W=" << W
                  << " max_cluster=" << requested_max_cluster
                  << " fused_state_fraction="
                  << (total ? double(fused) / double(total) : 0.0)
                  << " reflection_root_scan=0"
                  << " reflected_primitive_rescan=0\n";
        return 0;
    }
    if (W > 10) return 3;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "reverse runtime device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "reverse runtime set device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "reverse runtime props");
    if (prop.major < 9) {
        std::cerr << "reverse runtime DSM requires compute capability >= 9.0\n";
        return 5;
    }
    install_tables(rt);
    install_stationary_tables(st);

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(
           &attr, two_cell_reverse_fusion2_cluster_sliced_kernel),
       "reverse runtime function attrs");
    const Rank per_block_limit = std::min<Rank>(
        shared_kib * 1024ULL,
        static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);

    std::uint32_t *d_left = nullptr, *d_reflection = nullptr;
    Rank* d_offset = nullptr;
    ck(cudaMalloc(&d_left,
                  host_lut.label_left.size() * sizeof(std::uint32_t)),
       "reverse runtime alloc left LUT");
    ck(cudaMalloc(&d_reflection,
                  host_lut.reflection_meta.size() * sizeof(std::uint32_t)),
       "reverse runtime alloc reflection LUT");
    ck(cudaMalloc(&d_offset, sizeof(host_lut.offset)),
       "reverse runtime alloc offsets");
    ck(cudaMemcpy(d_left, host_lut.label_left.data(),
                  host_lut.label_left.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice),
       "reverse runtime copy left LUT");
    ck(cudaMemcpy(d_reflection, host_lut.reflection_meta.data(),
                  host_lut.reflection_meta.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice),
       "reverse runtime copy reflection LUT");
    ck(cudaMemcpy(d_offset, host_lut.offset, sizeof(host_lut.offset),
                  cudaMemcpyHostToDevice),
       "reverse runtime copy offsets");

    const Rank states = st.total[W];
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 809ULL) % (mod - 1ULL)));

    const int outer_bits = W - 5;
    for (int start = 0; start <= W - 5; ++start) {
        const auto reference = reverse_fusion2_reference(
            input, W, start, rt, st, mod);
        std::uint32_t* d_values = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)),
           "reverse runtime alloc values");
        ck(cudaMalloc(&d_error, sizeof(int)),
           "reverse runtime alloc error");
        ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice),
           "reverse runtime copy input");
        ck(cudaMemset(d_error, 0, sizeof(int)),
           "reverse runtime zero error");

        Rank fallback = 0;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank support_count = rt.choose[outer_bits][o];
            const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
            const auto choice = choose_reverse_runtime_bucket(
                o, support_count, per_block_limit, static_shared,
                requested_max_cluster, rt);
            if (!choice.ok) {
                fallback += support_count * n;
                continue;
            }
            std::cout << "reverse-runtime-bucket"
                      << " start=" << start
                      << " outer=" << o
                      << " cluster=" << choice.cluster
                      << " local_states=" << choice.local_states
                      << " active_clusters=" << choice.max_active_clusters
                      << '\n';
            launch_forced_reverse_bucket(
                d_values, d_left, d_reflection, d_offset,
                W, start, o, support_count, states, mod, d_error, choice);
        }
        if (fallback) {
            std::cerr << "reverse runtime small-width correctness requires all buckets fused;"
                      << " fallback_states=" << fallback << '\n';
            return 6;
        }
        ck(cudaDeviceSynchronize(), "reverse runtime sync");

        std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
        int error = 0;
        ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost),
           "reverse runtime copy output");
        ck(cudaMemcpy(&error, d_error, sizeof(error),
                      cudaMemcpyDeviceToHost),
           "reverse runtime copy error");
        if (error || output != reference) {
            std::cerr << "FAIL reverse runtime arithmetic"
                      << " W=" << W << " start=" << start
                      << " error=" << error << '\n';
            return 7;
        }
        cudaFree(d_values);
        cudaFree(d_error);
    }

    cudaFree(d_left);
    cudaFree(d_reflection);
    cudaFree(d_offset);
    std::cout << "reverse-runtime arithmetic=OK W=" << W << '\n';
    return 0;
}
