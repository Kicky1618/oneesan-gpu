#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_turn_owner_subwarp_microprobe_main_unused
#include "gridfp_reduced_production_turn_owner_subwarp_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_turn_high_device.cuh"

namespace {

__device__ __forceinline__ bool turn_high_owner_small_step(
    DeviceKey src, int W, bool expand, SmallTerms& edge
) {
    return expand ? turn_small_expand_high_step(src, W, edge)
                  : turn_small_compress_high_step(src, W, edge);
}

__global__ void turn_high_owner_subwarp_inplace_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 local_components,
    int W,
    int K,
    bool expand,
    int gpu_id,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ component_prefix,
    const Rank64* __restrict__ component_sr_begin,
    const Rank64* __restrict__ component_group,
    std::uint32_t mod,
    unsigned long long* __restrict__ processed,
    int* error
) {
    __shared__ DeviceKey sh_src[WARPS_PER_BLOCK][SUBGROUPS_PER_WARP][SUB_MAX_PAIRS];
    __shared__ DeviceKey sh_dst[WARPS_PER_BLOCK][SUBGROUPS_PER_WARP][SUB_MAX_PAIRS];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][SUBGROUPS_PER_WARP][SUB_MAX_PAIRS];
    __shared__ GroupedComponentContextDevice sh_ctx[WARPS_PER_BLOCK][SUBGROUPS_PER_WARP];
    __shared__ int sh_ns[WARPS_PER_BLOCK][SUBGROUPS_PER_WARP];
    __shared__ int sh_nd[WARPS_PER_BLOCK][SUBGROUPS_PER_WARP];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane / SUBGROUP_WIDTH;
    const int sublane = lane & (SUBGROUP_WIDTH - 1);
    const unsigned subgroup_mask = 0xffu << (subgroup * SUBGROUP_WIDTH);
    const Rank64 warp_global = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 first = warp_global * SUBGROUPS_PER_WARP + Rank64(subgroup);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK * SUBGROUPS_PER_WARP;
    const OwnerComponentPlanDevice plan{
        component_prefix, component_sr_begin, component_group};
    const int q = W - 1;
    const bool reverse = !expand;
    const int tile_start = expand ? W - 1 : K + 1;

    for (Rank64 local_rank = first; local_rank < local_components; local_rank += stride) {
        if (sublane == 0) {
            sh_ns[warp][subgroup] = 0;
            sh_nd[warp][subgroup] = 0;
            const MateID label = expand
                ? owner_component_label_unrank_planned_device(
                    W, W - 1, false, W - 1, K, plan, local_rank)
                : turn_compress_high_label_unrank_planned_device(W, K, plan, local_rank);
            const DeviceKey seed = expand
                ? turn_expand_high_seed(label, W)
                : turn_compress_high_seed(label, W);
            const GroupedComponentContextDevice ctx = grouped_component_context_device(
                seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
            sh_ctx[warp][subgroup] = ctx;
            if (ctx.owner != gpu_id) {
                set_error(error, 271);
            } else {
                sh_src[warp][subgroup][0] = seed;
                sh_ns[warp][subgroup] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp][subgroup]) {
                    SmallTerms edge;
                    if (!turn_high_owner_small_step(
                            sh_src[warp][subgroup][cursor++], W, expand, edge)) {
                        set_error(error, 272); break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp][subgroup], sh_nd[warp][subgroup], d) >= 0)
                            continue;
                        if (sh_nd[warp][subgroup] >= SUB_MAX_PAIRS) {
                            set_error(error, 273); break;
                        }
                        sh_dst[warp][subgroup][sh_nd[warp][subgroup]++] = d;
                        if (!turn_high_discover_inverse_to_set(
                                d, W, expand,
                                sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                SUB_MAX_PAIRS)) {
                            set_error(error, 274); break;
                        }
                    }
                    if (*error) break;
                }
            }
        }
        __syncwarp(subgroup_mask);

        const int ns = sh_ns[warp][subgroup];
        const int nd = sh_nd[warp][subgroup];
        const GroupedComponentContextDevice ctx = sh_ctx[warp][subgroup];
        for (int i = sublane; i < ns; i += SUBGROUP_WIDTH) {
            const GroupedDeviceRank gr = grouped_rank_in_component_device(
                sh_src[warp][subgroup][i], W, q, reverse, ctx);
            if (gr.owner != gpu_id) {
                set_error(error, 275);
                sh_value[warp][subgroup][i] = 0;
            } else {
                sh_value[warp][subgroup][i] = state[gr.local];
            }
        }
        __syncwarp(subgroup_mask);

        const int dst_q = expand ? W - 2 : W - 1;
        for (int di = sublane; di < nd; di += SUBGROUP_WIDTH) {
            const DeviceKey mine = sh_dst[warp][subgroup][di];
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, dst_q, reverse, ctx);
            if (dgr.owner != gpu_id) {
                set_error(error, 276); continue;
            }
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!turn_high_owner_small_step(sh_src[warp][subgroup][si], W, expand, edge)) {
                    set_error(error, 277); continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(edge.v[ei].coef) *
                               static_cast<long long>(sh_value[warp][subgroup][si]);
                }
            }
            long long z = acc % static_cast<long long>(mod);
            if (z < 0) z += mod;
            state[dgr.local] = static_cast<std::uint32_t>(z);
        }
        __syncwarp(subgroup_mask);
        if (sublane == 0) atomicAdd(processed, 1ULL);
    }
}

void build_high_turn_grouped_reference(
    int W,
    int K,
    int ngpu,
    std::uint32_t mod,
    const ProductionFactorTables& tables,
    const HostTilePlan& tile,
    std::vector<std::uint32_t>& input,
    std::vector<std::uint32_t>& expected
) {
    const OwnerPlan owner_plan{tile.owner_begin, tile.owner_size};
    const auto main_words = gen_words(W);
    const auto block_words = gen_words(W - 1);
    const auto qh = layout(main_words, block_words, W - 1);
    input.assign(static_cast<std::size_t>(tables.size()), 0);
    expected.assign(static_cast<std::size_t>(tables.size()), 0);
    std::map<Key,std::uint32_t> main_values;
    Rank64 serial = 0;
    for (Key k : qh) {
        const std::uint32_t value = static_cast<std::uint32_t>(
            1 + (serial++ * 2654435761ULL) % (mod - 1ULL));
        const GroupedRank gr = grouped_rank(
            k, tables, W, W - 1, true, K + 1, K, ngpu, owner_plan);
        input[static_cast<std::size_t>(tile.shard_base[gr.owner] + gr.local)] = value;
        for (const auto& [d,c] : step_basis(k, W, W - 1, true))
            add_mod_signed(main_values[d], value, int(c), mod);
    }
    for (const auto& [s,value] : main_values) {
        const Vec col = project_vec(step_basis(s, W, W - 1, false), W, W - 2, false);
        for (const auto& [d,c] : col) {
            const GroupedRank gr = grouped_rank(
                d, tables, W, W - 2, false, W - 1, K, ngpu, owner_plan);
            add_mod_signed(expected[static_cast<std::size_t>(tile.shard_base[gr.owner] + gr.local)],
                           value, int(c), mod);
        }
    }
    if (serial != tables.size()) fail("high turn owner input dimension");
}

void run_high_turn_owner_subwarp(
    int W, int K, int ngpu, unsigned blocks, std::uint32_t mod
) {
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    std::vector<std::uint32_t> input, expected;
    build_high_turn_grouped_reference(W, K, ngpu, mod, tables, tile, input, expected);

    double total_ms = 0.0;
    Rank64 compress_total = 0, expand_total = 0;
    for (int g = 0; g < ngpu; ++g) {
        const HostOwnerComponentPlan cp = make_host_turn_compress_plan(tables, K, g, ngpu);
        const HostOwnerComponentPlan ep = make_host_owner_component_plan(tables, K, g, ngpu);
        compress_total += cp.prefix.back();
        expand_total += ep.prefix.back();
        const Rank64 local_states = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];

        std::uint32_t* d_state = nullptr;
        Rank64* d_owner_begin = nullptr;
        unsigned long long* d_processed = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_state, local_states * sizeof(std::uint32_t)), "high turn alloc state");
        ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "high turn alloc owner begin");
        ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "high turn alloc processed");
        ck(cudaMalloc(&d_error, sizeof(int)), "high turn alloc error");
        ck(cudaMemcpy(d_state, input.data() + base,
                      local_states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "high turn copy state");
        ck(cudaMemcpy(d_owner_begin, tile.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "high turn copy owner begin");

        for (int phase = 0; phase < 2; ++phase) {
            const bool expand = phase != 0;
            const HostOwnerComponentPlan& hp = expand ? ep : cp;
            Rank64 *d_prefix = nullptr, *d_sr_begin = nullptr, *d_cg = nullptr;
            upload_plan(hp, d_prefix, d_sr_begin, d_cg);
            ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "high turn zero processed");
            ck(cudaMemset(d_error, 0, sizeof(int)), "high turn zero error");
            const Rank64 n = hp.prefix.back();
            const Rank64 one_pass =
                (n + WARPS_PER_BLOCK * SUBGROUPS_PER_WARP - 1) /
                (WARPS_PER_BLOCK * SUBGROUPS_PER_WARP);
            const unsigned launch_blocks = static_cast<unsigned>(
                std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
            cudaEvent_t a{}, b{};
            ck(cudaEventCreate(&a), "high turn event a");
            ck(cudaEventCreate(&b), "high turn event b");
            ck(cudaEventRecord(a), "high turn record a");
            turn_high_owner_subwarp_inplace_kernel<<<launch_blocks, THREADS>>>(
                d_state, n, W, K, expand, g, ngpu,
                d_owner_begin, d_prefix, d_sr_begin, d_cg,
                mod, d_processed, d_error);
            ck(cudaGetLastError(), "high turn launch");
            ck(cudaEventRecord(b), "high turn record b");
            ck(cudaEventSynchronize(b), "high turn sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, a, b), "high turn elapsed");
            total_ms += ms;
            int error = 0;
            unsigned long long processed = 0;
            ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "high turn copy error");
            ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "high turn copy processed");
            if (error || processed != n)
                fail("high turn accounting g=" + std::to_string(g) + " phase=" + std::to_string(phase));
            cudaEventDestroy(a); cudaEventDestroy(b);
            cudaFree(d_cg); cudaFree(d_sr_begin); cudaFree(d_prefix);
        }

        std::vector<std::uint32_t> got(static_cast<std::size_t>(local_states));
        ck(cudaMemcpy(got.data(), d_state, local_states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "high turn copy output");
        if (!std::equal(got.begin(), got.end(), expected.begin() + static_cast<std::ptrdiff_t>(base)))
            fail("high turn arithmetic g=" + std::to_string(g));
        cudaFree(d_error); cudaFree(d_processed); cudaFree(d_owner_begin); cudaFree(d_state);
    }
    if (compress_total != motzkin_count(W - 1)) fail("high turn compress total");
    if (expand_total != motzkin_count(W - 1) - motzkin_count(W - 3))
        fail("high turn expand total");
    std::cout << "gridfp-reduced-high-turn-owner-subwarp"
              << " W=" << W << " K=" << K << " ngpu=" << ngpu
              << " compress_components=" << compress_total
              << " expand_components=" << expand_total
              << " duplicate_component_scans=0 components_per_warp=4 subgroup_width=8"
              << " second_state_buffer_bytes=0"
              << " summed_sequential_gpu_ms=" << total_ms
              << " exact=OK\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : 4;
    const unsigned blocks = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    const std::uint32_t mod = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 7 || W > RP_MAX_W || K < 2 || K > W - 3 || !blocks || ngpu < 2 || ngpu > 16 || mod < 3)
        return 2;
    ProductionFactorTables tables(W);
    if (plan_only) {
        Rank64 csum = 0, esum = 0;
        for (int g = 0; g < ngpu; ++g) {
            csum += make_host_turn_compress_plan(tables, K, g, ngpu).prefix.back();
            esum += make_host_owner_component_plan(tables, K, g, ngpu).prefix.back();
        }
        std::cout << "gridfp-reduced-high-turn-owner-subwarp-plan"
                  << " W=" << W << " K=" << K << " ngpu=" << ngpu
                  << " compress_components=" << csum
                  << " expand_components=" << esum
                  << " duplicate_component_scans=0 components_per_warp=4"
                  << " second_state_buffer_bytes=0 component_table_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "high turn device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "high turn set device");
    install_tables(tables);
    run_high_turn_owner_subwarp(W, K, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_high_turn_owner_subwarp=1 W=" << W << '\n';
    return 0;
}
