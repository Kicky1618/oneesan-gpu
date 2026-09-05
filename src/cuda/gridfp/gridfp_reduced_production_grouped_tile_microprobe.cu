#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_dense_microprobe_main_unused
#include "gridfp_reduced_production_component_dense_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_grouped_device.cuh"

namespace {

struct HostTilePlan {
    std::vector<Rank64> owner_begin;
    std::vector<Rank64> owner_size;
    std::vector<Rank64> shard_base;
};

Rank64 host_group_size(const ProductionFactorTables& t, int L, int outer_ones) {
    Rank64 total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        const Rank64 pc = t.primitive[static_cast<std::size_t>(occupied)][1];
        total += (t.binom(L, local) + t.binom(L - 2, local - 1)) * pc;
    }
    return total;
}

HostTilePlan make_host_tile_plan(
    const ProductionFactorTables& t,
    int K,
    int ngpu
) {
    const int L = K + 2;
    const int O = t.W - L;
    HostTilePlan out;
    out.owner_begin.assign(static_cast<std::size_t>(ngpu), std::numeric_limits<Rank64>::max());
    out.owner_size.assign(static_cast<std::size_t>(ngpu), 0);
    out.shard_base.assign(static_cast<std::size_t>(ngpu), 0);

    Rank64 total = 0;
    for (int r = 0; r <= O; ++r)
        total += t.binom(O, r) * host_group_size(t, L, r);
    if (total != t.size()) fail("grouped CUDA host plan dimension");

    Rank64 prefix = 0;
    int last_owner = -1;
    for (int r = 0; r <= O; ++r) {
        const Rank64 group = host_group_size(t, L, r);
        const Rank64 count = t.binom(O, r);
        for (Rank64 sr = 0; sr < count; ++sr) {
            const Rank64 base = prefix + sr * group;
            const Rank64 midpoint = base + group / 2;
            int owner = int((__uint128_t(midpoint) * ngpu) / total);
            if (owner >= ngpu) owner = ngpu - 1;
            if (owner < last_owner) fail("grouped CUDA owner monotonicity");
            last_owner = owner;
            auto& begin = out.owner_begin[static_cast<std::size_t>(owner)];
            if (begin == std::numeric_limits<Rank64>::max()) begin = base;
            out.owner_size[static_cast<std::size_t>(owner)] += group;
        }
        prefix += count * group;
    }
    Rank64 base = 0;
    for (int g = 0; g < ngpu; ++g) {
        if (out.owner_begin[static_cast<std::size_t>(g)] == std::numeric_limits<Rank64>::max())
            out.owner_begin[static_cast<std::size_t>(g)] = 0;
        out.shard_base[static_cast<std::size_t>(g)] = base;
        base += out.owner_size[static_cast<std::size_t>(g)];
    }
    if (base != total) fail("grouped CUDA shard sum");
    return out;
}

__global__ void global_to_grouped_kernel(
    const std::uint32_t* __restrict__ global,
    std::uint32_t* __restrict__ grouped,
    Rank64 states,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* owner_begin,
    const Rank64* shard_base,
    int* error
) {
    const Rank64 r = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= states) return;
    const DeviceKey k = factor_unrank_device(r, W, q - 1);
    const GroupedDeviceRank gr = grouped_rank_device(
        k, W, q, reverse, tile_start, K, ngpu, owner_begin);
    if (gr.owner < 0 || gr.owner >= ngpu) {
        set_error(error, 111);
        return;
    }
    grouped[shard_base[gr.owner] + gr.local] = global[r];
}

__global__ void grouped_to_global_kernel(
    const std::uint32_t* __restrict__ grouped,
    std::uint32_t* __restrict__ global,
    Rank64 states,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* owner_begin,
    const Rank64* shard_base,
    int* error
) {
    const Rank64 r = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= states) return;
    const DeviceKey k = factor_unrank_device(r, W, q - 1);
    const GroupedDeviceRank gr = grouped_rank_device(
        k, W, q, reverse, tile_start, K, ngpu, owner_begin);
    if (gr.owner < 0 || gr.owner >= ngpu) {
        set_error(error, 112);
        return;
    }
    global[r] = grouped[shard_base[gr.owner] + gr.local];
}

__global__ void grouped_component_step_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ writer,
    Rank64 components,
    Rank64 states,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* owner_begin,
    const Rank64* shard_base,
    std::uint32_t mod,
    unsigned long long* written,
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
                set_error(error, 113);
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
                        set_error(error, 114);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 115);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, q, reverse, pre);
                        if (np < 0) {
                            set_error(error, 116);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 117);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 118);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        const int component_owner = sh_owner[warp];
        if (lane < ns) {
            const GroupedDeviceRank gr = grouped_rank_device(
                sh_src[warp][lane], W, q, reverse, tile_start, K, ngpu, owner_begin);
            if (gr.owner != component_owner) {
                set_error(error, 119);
                sh_value[warp][lane] = 0;
            } else {
                sh_value[warp][lane] = input[shard_base[gr.owner] + gr.local];
            }
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            const GroupedDeviceRank dgr = grouped_rank_device(
                mine, W, next, reverse, tile_start, K, ngpu, owner_begin);
            if (dgr.owner != component_owner) {
                set_error(error, 120);
            } else {
                long long acc = 0;
                for (int si = 0; si < ns; ++si) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][si], W, q, reverse, edge)) {
                        set_error(error, 121);
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
                const Rank64 out_rank = shard_base[dgr.owner] + dgr.local;
                const unsigned long long empty = ~0ULL;
                const unsigned long long prev = atomicCAS(
                    writer + out_rank, empty, static_cast<unsigned long long>(component_rank));
                if (prev != empty && prev != static_cast<unsigned long long>(component_rank))
                    set_error(error, 122);
                output[out_rank] = static_cast<std::uint32_t>(z);
                atomicAdd(written, 1ULL);
            }
        }
        __syncwarp();
    }
}

void cpu_tile_reference(
    std::vector<std::uint32_t>& values,
    const ProductionFactorTables& tables,
    int tile_start,
    int K,
    bool reverse,
    std::uint32_t mod
) {
    int q = tile_start;
    for (int step = 0; step < K; ++step) {
        const int next = reverse ? q + 1 : q - 1;
        ProductionFactorCodec src(tables, q - 1);
        ProductionFactorCodec dst(tables, next - 1);
        std::vector<std::uint32_t> out(values.size());
        for (Rank64 s = 0; s < tables.size(); ++s) {
            const Key k = src.unrank(s);
            const std::uint32_t v = values[static_cast<std::size_t>(s)];
            for (const auto& [d, c] : reduced_step_basis(k, tables.W, q, reverse))
                add_mod_signed(out[static_cast<std::size_t>(dst.rank(d))], v, int(c), mod);
        }
        values.swap(out);
        q = next;
    }
}

void run_grouped_tile(
    int W,
    int tile_start,
    int K,
    bool reverse,
    int ngpu,
    unsigned blocks,
    std::uint32_t mod
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    const Rank64 states = tables.size();
    const Rank64 components = motzkin_count(W - 1) - motzkin_count(W - 3);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank64 r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>((1 + (r * 2654435761ULL) % (mod - 1ULL)) % mod);
    std::vector<std::uint32_t> reference = input;
    cpu_tile_reference(reference, tables, tile_start, K, reverse, mod);

    Rank64* d_owner_begin = nullptr;
    Rank64* d_shard_base = nullptr;
    std::uint32_t* d_global = nullptr;
    std::uint32_t* d_a = nullptr;
    std::uint32_t* d_b = nullptr;
    unsigned long long* d_writer = nullptr;
    unsigned long long* d_written = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "grouped alloc owner begin");
    ck(cudaMalloc(&d_shard_base, ngpu * sizeof(Rank64)), "grouped alloc shard base");
    ck(cudaMalloc(&d_global, states * sizeof(std::uint32_t)), "grouped alloc global");
    ck(cudaMalloc(&d_a, states * sizeof(std::uint32_t)), "grouped alloc a");
    ck(cudaMalloc(&d_b, states * sizeof(std::uint32_t)), "grouped alloc b");
    ck(cudaMalloc(&d_writer, states * sizeof(unsigned long long)), "grouped alloc writer");
    ck(cudaMalloc(&d_written, sizeof(unsigned long long)), "grouped alloc written");
    ck(cudaMalloc(&d_error, sizeof(int)), "grouped alloc error");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "grouped copy owner begin");
    ck(cudaMemcpy(d_shard_base, plan.shard_base.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "grouped copy shard base");
    ck(cudaMemcpy(d_global, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "grouped copy input");
    ck(cudaMemset(d_a, 0, states * sizeof(std::uint32_t)), "grouped zero a");
    ck(cudaMemset(d_b, 0, states * sizeof(std::uint32_t)), "grouped zero b");
    ck(cudaMemset(d_error, 0, sizeof(int)), "grouped zero error");

    const int threads = 256;
    const unsigned convert_blocks = static_cast<unsigned>((states + threads - 1) / threads);
    global_to_grouped_kernel<<<convert_blocks, threads>>>(
        d_global, d_a, states, W, tile_start, reverse, tile_start, K, ngpu,
        d_owner_begin, d_shard_base, d_error);
    ck(cudaGetLastError(), "grouped global-to-local launch");
    ck(cudaDeviceSynchronize(), "grouped global-to-local sync");

    const Rank64 one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned step_blocks = static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));
    int q = tile_start;
    double total_ms = 0.0;
    for (int step = 0; step < K; ++step) {
        ck(cudaMemset(d_writer, 0xff, states * sizeof(unsigned long long)), "grouped clear writer");
        ck(cudaMemset(d_written, 0, sizeof(unsigned long long)), "grouped clear written");
        cudaEvent_t aev{}, bev{};
        ck(cudaEventCreate(&aev), "grouped event a");
        ck(cudaEventCreate(&bev), "grouped event b");
        ck(cudaEventRecord(aev), "grouped record a");
        grouped_component_step_kernel<<<step_blocks, THREADS>>>(
            d_a, d_b, d_writer, components, states, W, q, reverse, tile_start, K,
            ngpu, d_owner_begin, d_shard_base, mod, d_written, d_error);
        ck(cudaGetLastError(), "grouped step launch");
        ck(cudaEventRecord(bev), "grouped record b");
        ck(cudaEventSynchronize(bev), "grouped step sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, aev, bev), "grouped step elapsed");
        total_ms += ms;
        unsigned long long written = 0;
        ck(cudaMemcpy(&written, d_written, sizeof(written), cudaMemcpyDeviceToHost), "grouped copy written");
        int error = 0;
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "grouped copy step error");
        if (error || written != states) {
            std::cerr << "FAIL grouped tile step=" << step << " q=" << q
                      << " error=" << error << " written=" << written
                      << " states=" << states << '\n';
            std::exit(123);
        }
        cudaEventDestroy(aev);
        cudaEventDestroy(bev);
        std::swap(d_a, d_b);
        q += reverse ? 1 : -1;
    }

    grouped_to_global_kernel<<<convert_blocks, threads>>>(
        d_a, d_global, states, W, q, reverse, tile_start, K, ngpu,
        d_owner_begin, d_shard_base, d_error);
    ck(cudaGetLastError(), "grouped local-to-global launch");
    ck(cudaDeviceSynchronize(), "grouped local-to-global sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    int error = 0;
    ck(cudaMemcpy(output.data(), d_global, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "grouped copy final");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "grouped copy final error");
    if (error) {
        std::cerr << "FAIL grouped final device error=" << error << '\n';
        std::exit(124);
    }
    if (output != reference) {
        for (Rank64 r = 0; r < states; ++r) {
            if (output[static_cast<std::size_t>(r)] == reference[static_cast<std::size_t>(r)]) continue;
            std::cerr << "FAIL grouped tile arithmetic rank=" << r
                      << " gpu=" << output[static_cast<std::size_t>(r)]
                      << " cpu=" << reference[static_cast<std::size_t>(r)] << '\n';
            break;
        }
        std::exit(125);
    }

    Rank64 min_states = states, max_states = 0;
    for (Rank64 z : plan.owner_size) {
        min_states = std::min(min_states, z);
        max_states = std::max(max_states, z);
    }
    std::cout << "gridfp-reduced-grouped-tile-microprobe"
              << " W=" << W
              << " tile_start=" << tile_start
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " logical_gpus=" << ngpu
              << " min_shard_states=" << min_states
              << " max_shard_states=" << max_states
              << " step_blocks=" << step_blocks
              << " tile_kernel_ms=" << total_ms
              << " intra_tile_shard_crossings=0"
              << " redistribution_bytes=0"
              << " arithmetic=OK\n";

    cudaFree(d_owner_begin);
    cudaFree(d_shard_base);
    cudaFree(d_global);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_writer);
    cudaFree(d_written);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : 4;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const unsigned blocks = argc > 4 ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 1024u;
    const std::uint32_t mod = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 7 || W > RP_MAX_W || K < 1 || K > W - 3 || ngpu < 1 || ngpu > 64 || !blocks || mod < 3) return 2;

    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    if (plan_only) {
        Rank64 min_states = tables.size(), max_states = 0;
        for (Rank64 z : plan.owner_size) {
            min_states = std::min(min_states, z);
            max_states = std::max(max_states, z);
        }
        std::cout << "gridfp-reduced-grouped-tile-microprobe-plan"
                  << " W=" << W
                  << " K=" << K
                  << " logical_gpus=" << ngpu
                  << " states=" << tables.size()
                  << " min_shard_states=" << min_states
                  << " max_shard_states=" << max_states
                  << " owner_begin_u64=" << ngpu
                  << " grouped_stream_total_GiB=" << double(tables.size()) * 4.0 / double(1ULL << 30)
                  << " intra_tile_redistribution_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "grouped tile device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "grouped tile set device");
    install_tables(tables);

    for (int start = K + 2; start <= W - 1; ++start)
        run_grouped_tile(W, start, K, false, ngpu, blocks, mod);
    for (int start = 1; start <= W - K - 2; ++start)
        run_grouped_tile(W, start, K, true, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_grouped_tile=1 W=" << W
              << " K=" << K << '\n';
    return 0;
}
