#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_tile_microprobe_main_unused
#include "gridfp_reduced_production_grouped_tile_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void grouped_component_step_inplace_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 components,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ shard_base,
    std::uint32_t mod,
    int* error
) {
    __shared__ DeviceKey sh_src[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ DeviceKey sh_dst[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];
    __shared__ int sh_owner[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int next = reverse ? q + 1 : q - 1;

    for (Rank64 component_rank = first; component_rank < components; component_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            sh_owner[warp] = -1;
            const MateID label = component_label_unrank_device(W, q, reverse, component_rank);
            bool eligible = false;
            const DeviceKey seed = component_seed_direction(label, W, q, reverse, eligible);
            if (!eligible) {
                set_error(error, 141);
            } else {
                const GroupedDeviceRank sgr = grouped_rank_device(
                    seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
                sh_owner[warp] = sgr.owner;
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][cursor++], W, q, reverse, edge)) {
                        set_error(error, 142);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 143);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, q, reverse, pre);
                        if (np < 0) {
                            set_error(error, 144);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 145);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 146);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        const int component_owner = sh_owner[warp];

        // The CPU in-place probe proves that the source and destination slot
        // sets of a component are identical in this grouped layout. Every
        // source value is therefore captured before any lane overwrites a slot.
        if (lane < ns) {
            const GroupedDeviceRank gr = grouped_rank_device(
                sh_src[warp][lane], W, q, reverse, tile_start, K, ngpu, owner_begin);
            if (gr.owner != component_owner) {
                set_error(error, 147);
                sh_value[warp][lane] = 0;
            } else {
                sh_value[warp][lane] = state[shard_base[gr.owner] + gr.local];
            }
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            const GroupedDeviceRank dgr = grouped_rank_device(
                mine, W, next, reverse, tile_start, K, ngpu, owner_begin);
            if (dgr.owner != component_owner) {
                set_error(error, 148);
            } else {
                long long acc = 0;
                for (int si = 0; si < ns; ++si) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][si], W, q, reverse, edge)) {
                        set_error(error, 149);
                        continue;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (key_equal(edge.v[ei].key, mine)) {
                            acc += static_cast<long long>(edge.v[ei].coef) *
                                   static_cast<long long>(sh_value[warp][si]);
                        }
                    }
                }
                long long z = acc % static_cast<long long>(mod);
                if (z < 0) z += mod;
                state[shard_base[dgr.owner] + dgr.local] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
    }
}

std::vector<std::uint32_t> host_grouped_layout(
    const std::vector<std::uint32_t>& global,
    const ProductionFactorTables& tables,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const HostTilePlan& plan
) {
    OwnerPlan owner_plan{plan.owner_begin, plan.owner_size};
    ProductionFactorCodec codec(tables, q - 1);
    std::vector<std::uint32_t> out(global.size());
    for (Rank64 r = 0; r < tables.size(); ++r) {
        const Key key = codec.unrank(r);
        const GroupedRank gr = grouped_rank(
            key, tables, tables.W, q, reverse, tile_start, K, ngpu, owner_plan);
        out[static_cast<std::size_t>(plan.shard_base[gr.owner] + gr.local)] =
            global[static_cast<std::size_t>(r)];
    }
    return out;
}

void run_grouped_inplace_tile(
    int W,
    int tile_start,
    int K,
    bool reverse,
    int ngpu,
    unsigned blocks,
    std::uint32_t mod
) {
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    const Rank64 components = motzkin_count(W - 1) - motzkin_count(W - 3);

    std::vector<std::uint32_t> global(static_cast<std::size_t>(tables.size()));
    for (Rank64 r = 0; r < tables.size(); ++r)
        global[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + (r * 2654435761ULL) % (mod - 1ULL));
    std::vector<std::uint32_t> reference = global;
    cpu_tile_reference(reference, tables, tile_start, K, reverse, mod);
    const int final_q = tile_start + (reverse ? K : -K);
    const std::vector<std::uint32_t> input = host_grouped_layout(
        global, tables, tile_start, reverse, tile_start, K, ngpu, plan);
    const std::vector<std::uint32_t> expected = host_grouped_layout(
        reference, tables, final_q, reverse, tile_start, K, ngpu, plan);

    std::uint32_t* d_state = nullptr;
    Rank64* d_owner_begin = nullptr;
    Rank64* d_shard_base = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, tables.size() * sizeof(std::uint32_t)), "inplace alloc state");
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "inplace alloc owner begin");
    ck(cudaMalloc(&d_shard_base, ngpu * sizeof(Rank64)), "inplace alloc shard base");
    ck(cudaMalloc(&d_error, sizeof(int)), "inplace alloc error");
    ck(cudaMemcpy(d_state, input.data(), tables.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "inplace copy state");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "inplace copy owner begin");
    ck(cudaMemcpy(d_shard_base, plan.shard_base.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "inplace copy shard base");
    ck(cudaMemset(d_error, 0, sizeof(int)), "inplace zero error");

    const Rank64 one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));
    int q = tile_start;
    const auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < K; ++step) {
        grouped_component_step_inplace_kernel<<<launch_blocks, THREADS>>>(
            d_state, components, W, q, reverse, tile_start, K, ngpu,
            d_owner_begin, d_shard_base, mod, d_error);
        ck(cudaGetLastError(), "inplace component launch");
        q += reverse ? 1 : -1;
    }
    ck(cudaDeviceSynchronize(), "inplace tile sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "inplace copy error");
    if (error) fail("inplace CUDA device error=" + std::to_string(error));
    std::vector<std::uint32_t> output(expected.size());
    ck(cudaMemcpy(output.data(), d_state, output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "inplace copy output");
    if (output != expected) fail("inplace grouped tile mismatch");

    std::cout << "W=" << W
              << " tile_start=" << tile_start
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << tables.size()
              << " components=" << components
              << " blocks=" << launch_blocks
              << " ms=" << ms
              << " state_buffers=1"
              << " second_state_buffer_bytes=0"
              << " per_state_metadata_bytes=0"
              << " exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_shard_base);
    cudaFree(d_owner_begin);
    cudaFree(d_state);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 9;
    const int K = argc > 2 ? std::atoi(argv[2]) : 3;
    const unsigned blocks = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    const std::uint32_t mod = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    if (W < 7 || W > 11 || K < 2 || K > W - 3 || blocks == 0 || ngpu != 8 || mod < 3) return 2;

    run_grouped_inplace_tile(W, W - 1, K, false, ngpu, blocks, mod);
    run_grouped_inplace_tile(W, 1, K, true, ngpu, blocks, mod);
    std::cout << "ALL_OK grouped_component_inplace_cuda=1\n";
    return 0;
}
