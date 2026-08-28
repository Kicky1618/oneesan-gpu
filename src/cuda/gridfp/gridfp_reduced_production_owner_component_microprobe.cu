#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_inplace_microprobe_main_unused
#include "gridfp_reduced_production_grouped_inplace_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_owner_component_device.cuh"

namespace {

Rank64 host_owner_component_group_size(const ProductionFactorTables& t, int L, int r) {
    Rank64 total = 0;
    for (int l = 0; l <= L - 1; ++l) {
        const int occupied = r + l;
        if (!(occupied & 1)) continue;
        const Rank64 pc = t.primitive[static_cast<std::size_t>(occupied)][1];
        total += (t.binom(L - 1, l) - t.binom(L - 3, l)) * pc;
    }
    return total;
}

Rank64 host_owner_component_count(
    const ProductionFactorTables& t, int K, int owner, int ngpu
) {
    const int L = K + 2;
    const int O = t.W - L;
    Rank64 total = 0;
    const HostTilePlan plan = make_host_tile_plan(t, K, ngpu);
    (void)plan;
    Rank64 state_prefix = 0;
    const Rank64 state_total = t.size();
    for (int r = 0; r <= O; ++r) {
        const Rank64 count = t.binom(O, r);
        const Rank64 group = host_group_size(t, L, r);
        Rank64 owned = 0;
        for (Rank64 sr = 0; sr < count; ++sr) {
            const Rank64 midpoint = state_prefix + sr * group + group / 2;
            int g = int((__uint128_t(midpoint) * ngpu) / state_total);
            if (g >= ngpu) g = ngpu - 1;
            owned += g == owner;
        }
        total += owned * host_owner_component_group_size(t, L, r);
        state_prefix += count * group;
    }
    return total;
}

__global__ void owner_component_step_inplace_kernel(
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
    std::uint32_t mod,
    unsigned long long* __restrict__ processed,
    int* error
) {
    __shared__ DeviceKey sh_src[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ DeviceKey sh_dst[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int next = reverse ? q + 1 : q - 1;

    for (Rank64 local_rank = first; local_rank < local_components; local_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = owner_component_label_unrank_device(
                W, q, reverse, tile_start, K, gpu_id, ngpu, local_rank);
            bool eligible = false;
            const DeviceKey seed = component_seed_direction(label, W, q, reverse, eligible);
            if (!eligible) {
                set_error(error, 191);
            } else {
                const GroupedDeviceRank sgr = grouped_rank_device(
                    seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
                if (sgr.owner != gpu_id) {
                    set_error(error, 192);
                } else {
                    sh_src[warp][0] = seed;
                    sh_ns[warp] = 1;
                    int cursor = 0;
                    while (cursor < sh_ns[warp]) {
                        SmallTerms edge;
                        if (!small_step(sh_src[warp][cursor++], W, q, reverse, edge)) {
                            set_error(error, 193); break;
                        }
                        for (int ei = 0; ei < edge.n; ++ei) {
                            if (!edge.v[ei].coef) continue;
                            const DeviceKey d = edge.v[ei].key;
                            if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                            if (sh_nd[warp] >= MAX_PAIRS) { set_error(error, 194); break; }
                            sh_dst[warp][sh_nd[warp]++] = d;
                            DeviceTerm pre[RP_MAX_TERMS]{};
                            const int np = inverse_direction(d, W, q, reverse, pre);
                            if (np < 0) { set_error(error, 195); break; }
                            for (int pi = 0; pi < np; ++pi) {
                                if (!pre[pi].coef) continue;
                                if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                                if (sh_ns[warp] >= MAX_PAIRS) { set_error(error, 196); break; }
                                sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                            }
                        }
                        if (*error) break;
                    }
                    if (sh_ns[warp] != sh_nd[warp]) set_error(error, 197);
                }
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp], nd = sh_nd[warp];
        if (lane < ns) {
            const GroupedDeviceRank gr = grouped_rank_device(
                sh_src[warp][lane], W, q, reverse, tile_start, K, ngpu, owner_begin);
            if (gr.owner != gpu_id) {
                set_error(error, 198);
                sh_value[warp][lane] = 0;
            } else {
                sh_value[warp][lane] = state[gr.local];
            }
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            const GroupedDeviceRank dgr = grouped_rank_device(
                mine, W, next, reverse, tile_start, K, ngpu, owner_begin);
            if (dgr.owner != gpu_id) {
                set_error(error, 199);
            } else {
                long long acc = 0;
                for (int si = 0; si < ns; ++si) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][si], W, q, reverse, edge)) {
                        set_error(error, 200); continue;
                    }
                    for (int ei = 0; ei < edge.n; ++ei)
                        if (key_equal(edge.v[ei].key, mine))
                            acc += static_cast<long long>(edge.v[ei].coef) *
                                   static_cast<long long>(sh_value[warp][si]);
                }
                long long z = acc % static_cast<long long>(mod);
                if (z < 0) z += mod;
                state[dgr.local] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
        if (lane == 0) atomicAdd(processed, 1ULL);
    }
}

void run_owner_position(
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
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    const OwnerPlan owner_plan{plan.owner_begin, plan.owner_size};
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
        global, tables, q, reverse, tile_start, K, ngpu, plan);
    const std::vector<std::uint32_t> grouped_out = host_grouped_layout(
        reference, tables, next, reverse, tile_start, K, ngpu, plan);

    Rank64 total_components = 0;
    double total_ms = 0.0;
    for (int g = 0; g < ngpu; ++g) {
        const Rank64 local_states = plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        const Rank64 local_components = host_owner_component_count(tables, K, g, ngpu);
        total_components += local_components;

        std::uint32_t* d_state = nullptr;
        Rank64* d_owner_begin = nullptr;
        unsigned long long* d_processed = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_state, local_states * sizeof(std::uint32_t)), "owner component alloc state");
        ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "owner component alloc owner begin");
        ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "owner component alloc processed");
        ck(cudaMalloc(&d_error, sizeof(int)), "owner component alloc error");
        ck(cudaMemcpy(d_state, grouped_in.data() + base,
                      local_states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "owner component copy state");
        ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "owner component copy owner begin");
        ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "owner component zero processed");
        ck(cudaMemset(d_error, 0, sizeof(int)), "owner component zero error");

        const Rank64 one_pass = (local_components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned launch_blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
        cudaEvent_t a{}, b{};
        ck(cudaEventCreate(&a), "owner component event a");
        ck(cudaEventCreate(&b), "owner component event b");
        ck(cudaEventRecord(a), "owner component record a");
        owner_component_step_inplace_kernel<<<launch_blocks, THREADS>>>(
            d_state, local_components, W, q, reverse, tile_start, K, g, ngpu,
            d_owner_begin, mod, d_processed, d_error);
        ck(cudaGetLastError(), "owner component launch");
        ck(cudaEventRecord(b), "owner component record b");
        ck(cudaEventSynchronize(b), "owner component sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "owner component elapsed");
        total_ms += ms;

        int error = 0;
        unsigned long long processed = 0;
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "owner component copy error");
        ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "owner component copy processed");
        if (error || processed != local_components)
            fail("owner component device accounting g=" + std::to_string(g));
        std::vector<std::uint32_t> got(static_cast<std::size_t>(local_states));
        ck(cudaMemcpy(got.data(), d_state, local_states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "owner component copy output");
        if (!std::equal(got.begin(), got.end(), grouped_out.begin() + static_cast<std::ptrdiff_t>(base)))
            fail("owner component arithmetic g=" + std::to_string(g));

        cudaEventDestroy(a); cudaEventDestroy(b);
        cudaFree(d_error); cudaFree(d_processed); cudaFree(d_owner_begin); cudaFree(d_state);
    }

    const Rank64 expected_components = motzkin_count(W - 1) - motzkin_count(W - 3);
    if (total_components != expected_components) fail("owner component global accounting");
    std::cout << "gridfp-reduced-owner-component-microprobe"
              << " W=" << W << " q=" << q
              << " direction=" << (reverse ? "reverse" : "forward")
              << " K=" << K << " ngpu=" << ngpu
              << " global_components=" << total_components
              << " duplicate_component_scans=0"
              << " summed_sequential_gpu_ms=" << total_ms
              << " component_table_bytes=0 exact=OK\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : 4;
    const unsigned blocks = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    const std::uint32_t mod = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 7 || W > RP_MAX_W || K < 2 || K > W - 3 || !blocks || ngpu < 2 || ngpu > 16 || mod < 3) return 2;

    ProductionFactorTables tables(W);
    if (plan_only) {
        Rank64 lo = std::numeric_limits<Rank64>::max(), hi = 0, sum = 0;
        for (int g = 0; g < ngpu; ++g) {
            const Rank64 n = host_owner_component_count(tables, K, g, ngpu);
            lo = std::min(lo, n); hi = std::max(hi, n); sum += n;
        }
        std::cout << "gridfp-reduced-owner-component-plan"
                  << " W=" << W << " K=" << K << " ngpu=" << ngpu
                  << " components=" << sum
                  << " min_local_components=" << lo
                  << " max_local_components=" << hi
                  << " duplicate_component_scans=0"
                  << " component_table_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "owner component device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "owner component set device");
    install_tables(tables);

    const int fstart = W - 1;
    for (int q = W - 1; q >= std::max(3, W - K); --q)
        run_owner_position(W, q, false, fstart, K, ngpu, blocks, mod);
    const int rstart = 1;
    for (int q = 1; q <= std::min(W - 3, K); ++q)
        run_owner_position(W, q, true, rstart, K, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_owner_component_cuda=1\n";
    return 0;
}
