#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_owner_component_lean_microprobe_main_unused
#include "gridfp_reduced_production_owner_component_lean_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_group_context_device.cuh"

namespace {

static constexpr int SUBGROUPS_PER_WARP = 4;
static constexpr int SUBGROUP_WIDTH = 8;
static constexpr int SUB_MAX_PAIRS = 20;

__global__ void owner_component_subwarp_inplace_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 local_components,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
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
    const int next = reverse ? q + 1 : q - 1;
    const OwnerComponentPlanDevice plan{
        component_prefix, component_sr_begin, component_group};

    for (Rank64 local_rank = first; local_rank < local_components; local_rank += stride) {
        if (sublane == 0) {
            sh_ns[warp][subgroup] = 0;
            sh_nd[warp][subgroup] = 0;
            const MateID label = owner_component_label_unrank_planned_device(
                W, q, reverse, tile_start, K, plan, local_rank);
            bool eligible = false;
            const DeviceKey seed = component_seed_direction(label, W, q, reverse, eligible);
            if (!eligible) {
                set_error(error, 231);
            } else {
                const GroupedComponentContextDevice ctx = grouped_component_context_device(
                    seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
                sh_ctx[warp][subgroup] = ctx;
                if (ctx.owner != gpu_id) {
                    set_error(error, 232);
                } else {
                    sh_src[warp][subgroup][0] = seed;
                    sh_ns[warp][subgroup] = 1;
                    int cursor = 0;
                    while (cursor < sh_ns[warp][subgroup]) {
                        SmallTerms edge;
                        if (!small_step(
                                sh_src[warp][subgroup][cursor++], W, q, reverse, edge)) {
                            set_error(error, 233);
                            break;
                        }
                        for (int ei = 0; ei < edge.n; ++ei) {
                            if (!edge.v[ei].coef) continue;
                            const DeviceKey d = edge.v[ei].key;
                            if (find_key(sh_dst[warp][subgroup], sh_nd[warp][subgroup], d) >= 0)
                                continue;
                            if (sh_nd[warp][subgroup] >= SUB_MAX_PAIRS) {
                                set_error(error, 234);
                                break;
                            }
                            sh_dst[warp][subgroup][sh_nd[warp][subgroup]++] = d;
                            if (!discover_inverse_direction_to_set(
                                    d, W, q, reverse,
                                    sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                    SUB_MAX_PAIRS)) {
                                set_error(error, 235);
                                break;
                            }
                        }
                        if (*error) break;
                    }
                    if (sh_ns[warp][subgroup] != sh_nd[warp][subgroup])
                        set_error(error, 236);
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
                set_error(error, 237);
                sh_value[warp][subgroup][i] = 0;
            } else {
                sh_value[warp][subgroup][i] = state[gr.local];
            }
        }
        __syncwarp(subgroup_mask);

        for (int di = sublane; di < nd; di += SUBGROUP_WIDTH) {
            const DeviceKey mine = sh_dst[warp][subgroup][di];
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, next, reverse, ctx);
            if (dgr.owner != gpu_id) {
                set_error(error, 238);
                continue;
            }
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!small_step(sh_src[warp][subgroup][si], W, q, reverse, edge)) {
                    set_error(error, 239);
                    continue;
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

void run_owner_subwarp_position(
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    unsigned blocks,
    std::uint32_t mod
) {
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const int next = reverse ? q + 1 : q - 1;
    ProductionFactorCodec src_codec(tables, q - 1);
    ProductionFactorCodec dst_codec(tables, next - 1);

    std::vector<std::uint32_t> global(static_cast<std::size_t>(tables.size()));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(tables.size()));
    for (Rank64 r = 0; r < tables.size(); ++r)
        global[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + (r * 2654435761ULL) % (mod - 1ULL));
    for (Rank64 s = 0; s < tables.size(); ++s) {
        const Key k = src_codec.unrank(s);
        const std::uint32_t v = global[static_cast<std::size_t>(s)];
        for (const auto& [d,c] : reduced_step_basis(k, W, q, reverse))
            add_mod_signed(reference[static_cast<std::size_t>(dst_codec.rank(d))], v, int(c), mod);
    }
    const std::vector<std::uint32_t> grouped_in = host_grouped_layout(
        global, tables, q, reverse, tile_start, K, ngpu, tile);
    const std::vector<std::uint32_t> grouped_out = host_grouped_layout(
        reference, tables, next, reverse, tile_start, K, ngpu, tile);

    Rank64 sum_components = 0;
    double total_ms = 0.0;
    for (int g = 0; g < ngpu; ++g) {
        const HostOwnerComponentPlan hp = make_host_owner_component_plan(tables, K, g, ngpu);
        const Rank64 local_components = hp.prefix.back();
        const Rank64 local_states = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        sum_components += local_components;

        std::uint32_t* d_state = nullptr;
        Rank64 *d_owner_begin = nullptr, *d_prefix = nullptr, *d_sr_begin = nullptr, *d_cg = nullptr;
        unsigned long long* d_processed = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_state, local_states * sizeof(std::uint32_t)), "subwarp alloc state");
        ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "subwarp alloc owner begin");
        ck(cudaMalloc(&d_prefix, hp.prefix.size() * sizeof(Rank64)), "subwarp alloc prefix");
        ck(cudaMalloc(&d_sr_begin, hp.sr_begin.size() * sizeof(Rank64)), "subwarp alloc sr begin");
        ck(cudaMalloc(&d_cg, hp.component_group.size() * sizeof(Rank64)), "subwarp alloc component group");
        ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "subwarp alloc processed");
        ck(cudaMalloc(&d_error, sizeof(int)), "subwarp alloc error");
        ck(cudaMemcpy(d_state, grouped_in.data() + base,
                      local_states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "subwarp copy state");
        ck(cudaMemcpy(d_owner_begin, tile.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "subwarp copy owner begin");
        ck(cudaMemcpy(d_prefix, hp.prefix.data(), hp.prefix.size() * sizeof(Rank64), cudaMemcpyHostToDevice), "subwarp copy prefix");
        ck(cudaMemcpy(d_sr_begin, hp.sr_begin.data(), hp.sr_begin.size() * sizeof(Rank64), cudaMemcpyHostToDevice), "subwarp copy sr begin");
        ck(cudaMemcpy(d_cg, hp.component_group.data(), hp.component_group.size() * sizeof(Rank64), cudaMemcpyHostToDevice), "subwarp copy component group");
        ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "subwarp zero processed");
        ck(cudaMemset(d_error, 0, sizeof(int)), "subwarp zero error");

        const Rank64 one_pass =
            (local_components + WARPS_PER_BLOCK * SUBGROUPS_PER_WARP - 1) /
            (WARPS_PER_BLOCK * SUBGROUPS_PER_WARP);
        const unsigned launch_blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
        cudaEvent_t a{}, b{};
        ck(cudaEventCreate(&a), "subwarp event a");
        ck(cudaEventCreate(&b), "subwarp event b");
        ck(cudaEventRecord(a), "subwarp record a");
        owner_component_subwarp_inplace_kernel<<<launch_blocks, THREADS>>>(
            d_state, local_components, W, q, reverse, tile_start, K, g, ngpu,
            d_owner_begin, d_prefix, d_sr_begin, d_cg, mod, d_processed, d_error);
        ck(cudaGetLastError(), "subwarp launch");
        ck(cudaEventRecord(b), "subwarp record b");
        ck(cudaEventSynchronize(b), "subwarp sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "subwarp elapsed");
        total_ms += ms;

        int error = 0;
        unsigned long long processed = 0;
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "subwarp copy error");
        ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "subwarp copy processed");
        if (error || processed != local_components)
            fail("subwarp accounting g=" + std::to_string(g) + " error=" + std::to_string(error));
        std::vector<std::uint32_t> got(static_cast<std::size_t>(local_states));
        ck(cudaMemcpy(got.data(), d_state, local_states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "subwarp copy output");
        if (!std::equal(got.begin(), got.end(), grouped_out.begin() + static_cast<std::ptrdiff_t>(base)))
            fail("subwarp arithmetic g=" + std::to_string(g));

        cudaEventDestroy(a); cudaEventDestroy(b);
        cudaFree(d_error); cudaFree(d_processed); cudaFree(d_cg); cudaFree(d_sr_begin);
        cudaFree(d_prefix); cudaFree(d_owner_begin); cudaFree(d_state);
    }

    const Rank64 expected = motzkin_count(W - 1) - motzkin_count(W - 3);
    if (sum_components != expected) fail("subwarp global component accounting");
    std::cout << "gridfp-reduced-owner-subwarp"
              << " W=" << W << " q=" << q
              << " direction=" << (reverse ? "reverse" : "forward")
              << " K=" << K << " ngpu=" << ngpu
              << " components=" << sum_components
              << " components_per_warp=4 subgroup_width=8"
              << " max_pairs_capacity=" << SUB_MAX_PAIRS
              << " outer_context_once_per_component=1"
              << " inverse_temp_terms=0 local_position_array=0"
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
    if (W < 7 || W > RP_MAX_W || K < 2 || K > W - 3 || !blocks ||
        ngpu < 2 || ngpu > 16 || mod < 3) return 2;

    ProductionFactorTables tables(W);
    if (plan_only) {
        Rank64 lo = std::numeric_limits<Rank64>::max(), hi = 0, sum = 0;
        for (int g = 0; g < ngpu; ++g) {
            const HostOwnerComponentPlan hp = make_host_owner_component_plan(tables, K, g, ngpu);
            const Rank64 n = hp.prefix.back();
            lo = std::min(lo, n); hi = std::max(hi, n); sum += n;
        }
        std::cout << "gridfp-reduced-owner-subwarp-plan"
                  << " W=" << W << " K=" << K << " ngpu=" << ngpu
                  << " components=" << sum
                  << " min_local_components=" << lo
                  << " max_local_components=" << hi
                  << " components_per_warp=4 subgroup_width=8"
                  << " max_pairs_capacity=" << SUB_MAX_PAIRS
                  << " component_table_bytes=0 inverse_table_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "subwarp device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "subwarp set device");
    install_tables(tables);

    const int fstart = W - 1;
    for (int q = W - 1; q >= std::max(3, W - K); --q)
        run_owner_subwarp_position(W, q, false, fstart, K, ngpu, blocks, mod);
    const int rstart = 1;
    for (int q = 1; q <= std::min(W - 3, K); ++q)
        run_owner_subwarp_position(W, q, true, rstart, K, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_owner_subwarp=1 W=" << W << '\n';
    return 0;
}
