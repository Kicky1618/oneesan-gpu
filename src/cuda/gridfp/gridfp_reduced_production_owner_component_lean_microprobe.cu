#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_owner_component_microprobe_main_unused
#include "gridfp_reduced_production_owner_component_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_owner_component_plan_device.cuh"
#include "gridfp_reduced_production_discovery_device.cuh"

namespace {

struct HostOwnerComponentPlan {
    std::vector<Rank64> prefix;
    std::vector<Rank64> sr_begin;
    std::vector<Rank64> component_group;
};

HostOwnerComponentPlan make_host_owner_component_plan(
    const ProductionFactorTables& t,
    int K,
    int owner,
    int ngpu
) {
    const int L = K + 2;
    const int O = t.W - L;
    HostOwnerComponentPlan out;
    out.prefix.assign(static_cast<std::size_t>(O + 2), 0);
    out.sr_begin.assign(static_cast<std::size_t>(O + 1), 0);
    out.component_group.assign(static_cast<std::size_t>(O + 1), 0);

    Rank64 state_prefix = 0;
    const Rank64 total = t.size();
    for (int r = 0; r <= O; ++r) {
        const Rank64 count = t.binom(O, r);
        const Rank64 group = host_group_size(t, L, r);
        const Rank64 cg = host_owner_component_group_size(t, L, r);
        out.component_group[static_cast<std::size_t>(r)] = cg;

        Rank64 first = count, last = count;
        bool seen = false, ended = false;
        for (Rank64 sr = 0; sr < count; ++sr) {
            const Rank64 midpoint = state_prefix + sr * group + group / 2;
            int g = int((__uint128_t(midpoint) * ngpu) / total);
            if (g >= ngpu) g = ngpu - 1;
            if (g == owner) {
                if (ended) fail("owner component plan noncontiguous support range");
                if (!seen) first = sr;
                last = sr + 1;
                seen = true;
            } else if (seen) {
                ended = true;
            }
        }
        if (!seen) first = last = 0;
        out.sr_begin[static_cast<std::size_t>(r)] = first;
        out.prefix[static_cast<std::size_t>(r + 1)] =
            out.prefix[static_cast<std::size_t>(r)] + (last - first) * cg;
        state_prefix += count * group;
    }
    if (state_prefix != total) fail("owner component planned state total");
    if (out.prefix.back() != host_owner_component_count(t, K, owner, ngpu))
        fail("owner component planned count mismatch");
    return out;
}

__global__ void owner_component_lean_inplace_kernel(
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
    const OwnerComponentPlanDevice plan{
        component_prefix, component_sr_begin, component_group};

    for (Rank64 local_rank = first; local_rank < local_components; local_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = owner_component_label_unrank_planned_device(
                W, q, reverse, tile_start, K, plan, local_rank);
            bool eligible = false;
            const DeviceKey seed = component_seed_direction(label, W, q, reverse, eligible);
            if (!eligible) {
                set_error(error, 211);
            } else {
                const GroupedDeviceRank sgr = grouped_rank_device(
                    seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
                if (sgr.owner != gpu_id) {
                    set_error(error, 212);
                } else {
                    sh_src[warp][0] = seed;
                    sh_ns[warp] = 1;
                    int cursor = 0;
                    while (cursor < sh_ns[warp]) {
                        SmallTerms edge;
                        if (!small_step(sh_src[warp][cursor++], W, q, reverse, edge)) {
                            set_error(error, 213);
                            break;
                        }
                        for (int ei = 0; ei < edge.n; ++ei) {
                            if (!edge.v[ei].coef) continue;
                            const DeviceKey d = edge.v[ei].key;
                            if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                            if (sh_nd[warp] >= MAX_PAIRS) {
                                set_error(error, 214);
                                break;
                            }
                            sh_dst[warp][sh_nd[warp]++] = d;
                            if (!discover_inverse_direction_to_set(
                                    d, W, q, reverse,
                                    sh_src[warp], sh_ns[warp], MAX_PAIRS)) {
                                set_error(error, 215);
                                break;
                            }
                        }
                        if (*error) break;
                    }
                    if (sh_ns[warp] != sh_nd[warp]) set_error(error, 216);
                }
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        if (lane < ns) {
            const GroupedDeviceRank gr = grouped_rank_device(
                sh_src[warp][lane], W, q, reverse, tile_start, K, ngpu, owner_begin);
            if (gr.owner != gpu_id) {
                set_error(error, 217);
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
                set_error(error, 218);
            } else {
                long long acc = 0;
                for (int si = 0; si < ns; ++si) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][si], W, q, reverse, edge)) {
                        set_error(error, 219);
                        continue;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (key_equal(edge.v[ei].key, mine))
                            acc += static_cast<long long>(edge.v[ei].coef) *
                                   static_cast<long long>(sh_value[warp][si]);
                    }
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

void run_owner_lean_position(
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
        ck(cudaMalloc(&d_state, local_states * sizeof(std::uint32_t)), "owner lean alloc state");
        ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "owner lean alloc owner begin");
        ck(cudaMalloc(&d_prefix, hp.prefix.size() * sizeof(Rank64)), "owner lean alloc prefix");
        ck(cudaMalloc(&d_sr_begin, hp.sr_begin.size() * sizeof(Rank64)), "owner lean alloc sr begin");
        ck(cudaMalloc(&d_cg, hp.component_group.size() * sizeof(Rank64)), "owner lean alloc component group");
        ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "owner lean alloc processed");
        ck(cudaMalloc(&d_error, sizeof(int)), "owner lean alloc error");
        ck(cudaMemcpy(d_state, grouped_in.data() + base,
                      local_states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "owner lean copy state");
        ck(cudaMemcpy(d_owner_begin, tile.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "owner lean copy owner begin");
        ck(cudaMemcpy(d_prefix, hp.prefix.data(), hp.prefix.size() * sizeof(Rank64), cudaMemcpyHostToDevice), "owner lean copy prefix");
        ck(cudaMemcpy(d_sr_begin, hp.sr_begin.data(), hp.sr_begin.size() * sizeof(Rank64), cudaMemcpyHostToDevice), "owner lean copy sr begin");
        ck(cudaMemcpy(d_cg, hp.component_group.data(), hp.component_group.size() * sizeof(Rank64), cudaMemcpyHostToDevice), "owner lean copy component group");
        ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "owner lean zero processed");
        ck(cudaMemset(d_error, 0, sizeof(int)), "owner lean zero error");

        const Rank64 one_pass = (local_components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned launch_blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
        cudaEvent_t a{}, b{};
        ck(cudaEventCreate(&a), "owner lean event a");
        ck(cudaEventCreate(&b), "owner lean event b");
        ck(cudaEventRecord(a), "owner lean record a");
        owner_component_lean_inplace_kernel<<<launch_blocks, THREADS>>>(
            d_state, local_components, W, q, reverse, tile_start, K, g, ngpu,
            d_owner_begin, d_prefix, d_sr_begin, d_cg, mod, d_processed, d_error);
        ck(cudaGetLastError(), "owner lean launch");
        ck(cudaEventRecord(b), "owner lean record b");
        ck(cudaEventSynchronize(b), "owner lean sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "owner lean elapsed");
        total_ms += ms;

        int error = 0;
        unsigned long long processed = 0;
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "owner lean copy error");
        ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "owner lean copy processed");
        if (error || processed != local_components)
            fail("owner lean accounting g=" + std::to_string(g) + " error=" + std::to_string(error));
        std::vector<std::uint32_t> got(static_cast<std::size_t>(local_states));
        ck(cudaMemcpy(got.data(), d_state, local_states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "owner lean copy output");
        if (!std::equal(got.begin(), got.end(), grouped_out.begin() + static_cast<std::ptrdiff_t>(base)))
            fail("owner lean arithmetic g=" + std::to_string(g));

        cudaEventDestroy(a); cudaEventDestroy(b);
        cudaFree(d_error); cudaFree(d_processed); cudaFree(d_cg); cudaFree(d_sr_begin);
        cudaFree(d_prefix); cudaFree(d_owner_begin); cudaFree(d_state);
    }

    const Rank64 expected = motzkin_count(W - 1) - motzkin_count(W - 3);
    if (sum_components != expected) fail("owner lean global component accounting");
    std::cout << "gridfp-reduced-owner-component-lean"
              << " W=" << W << " q=" << q
              << " direction=" << (reverse ? "reverse" : "forward")
              << " K=" << K << " ngpu=" << ngpu
              << " components=" << sum_components
              << " duplicate_component_scans=0"
              << " inverse_temp_terms=0 local_position_array=0"
              << " owner_boundary_divisions_per_component=0"
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
        std::size_t plan_bytes = 0;
        for (int g = 0; g < ngpu; ++g) {
            const HostOwnerComponentPlan hp = make_host_owner_component_plan(tables, K, g, ngpu);
            const Rank64 n = hp.prefix.back();
            lo = std::min(lo, n); hi = std::max(hi, n); sum += n;
            plan_bytes += (hp.prefix.size() + hp.sr_begin.size() + hp.component_group.size()) * sizeof(Rank64);
        }
        std::cout << "gridfp-reduced-owner-component-lean-plan"
                  << " W=" << W << " K=" << K << " ngpu=" << ngpu
                  << " components=" << sum
                  << " min_local_components=" << lo
                  << " max_local_components=" << hi
                  << " total_plan_bytes=" << plan_bytes
                  << " inverse_temp_terms=0 local_position_array=0"
                  << " owner_boundary_divisions_per_component=0"
                  << " component_table_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "owner lean device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "owner lean set device");
    install_tables(tables);

    const int fstart = W - 1;
    for (int q = W - 1; q >= std::max(3, W - K); --q)
        run_owner_lean_position(W, q, false, fstart, K, ngpu, blocks, mod);
    const int rstart = 1;
    for (int q = 1; q <= std::min(W - 3, K); ++q)
        run_owner_lean_position(W, q, true, rstart, K, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_owner_component_lean=1 W=" << W << '\n';
    return 0;
}
