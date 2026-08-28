#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_microprobe_main_unused
#include "gridfp_reduced_production_shift_cycle_microprobe.cu"
#pragma pop_macro("main")

#include <array>
#include <vector>

namespace {

__global__ void peer_shifted_tile_cycle_kernel(
    std::uint32_t* const* __restrict__ shards,
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int worker_id,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ cycles,
    unsigned long long* __restrict__ rotated_values,
    unsigned long long* __restrict__ peer_values,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 local_warp = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 first = local_warp * Rank64(ngpu) + Rank64(worker_id);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK * Rank64(ngpu);
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            if (cycle_len < 0) {
                if (lane == 0) set_error(error, 171);
                continue;
            }
            if (cycle_len <= 1) continue;

            const DeviceKey leader = equal_run_key0_device(
                run.support, blocked, W, q, reverse);
            const GroupedDeviceRank lr = grouped_rank_device(
                leader, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
            if (lr.owner < 0 || lr.owner >= ngpu) {
                if (lane == 0) set_error(error, 172);
                continue;
            }

            const int occupied = __popc(run.support);
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            unsigned long long local_peer = 0;

            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = shards[lr.owner][lr.local + i];
                int prev_owner = lr.owner;
                std::uint32_t cur_support = shift_next_support_device(
                    run.support, blocked, W, q, Kwin, S, reverse);
                int hops = 1;
                while (cur_support != run.support) {
                    const DeviceKey cur = equal_run_key0_device(
                        cur_support, blocked, W, q, reverse);
                    const GroupedDeviceRank cr = grouped_rank_device(
                        cur, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
                    if (cr.owner < 0 || cr.owner >= ngpu) {
                        set_error(error, 173);
                        break;
                    }
                    const std::uint32_t next_value = shards[cr.owner][cr.local + i];
                    shards[cr.owner][cr.local + i] = temp;
                    if (cr.owner != prev_owner) ++local_peer;
                    temp = next_value;
                    prev_owner = cr.owner;
                    cur_support = shift_next_support_device(
                        cur_support, blocked, W, q, Kwin, S, reverse);
                    ++hops;
                    if (hops > cycle_len) {
                        set_error(error, 174);
                        break;
                    }
                }
                shards[lr.owner][lr.local + i] = temp;
                if (lr.owner != prev_owner) ++local_peer;
            }
            __syncwarp();
            if (lane == 0) {
                atomicAdd(cycles, 1ULL);
                atomicAdd(rotated_values, static_cast<unsigned long long>(pc) * cycle_len);
            }
            const unsigned long long warp_peer =
                __reduce_add_sync(0xffffffffu, local_peer);
            if (lane == 0 && warp_peer)
                atomicAdd(peer_values, warp_peer);
        }
    }
}

void enable_full_peer_mesh(int ngpu) {
    for (int src = 0; src < ngpu; ++src) {
        ck(cudaSetDevice(src), "peer mesh set source");
        for (int dst = 0; dst < ngpu; ++dst) {
            if (src == dst) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, src, dst), "peer mesh capability");
            if (!can) {
                std::cerr << "peer access unavailable src=" << src << " dst=" << dst << '\n';
                std::exit(175);
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "peer mesh enable");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

void run_peer_shift_cycle_probe(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    if (W < 7 || W > 11 || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || ngpu < 2 || ngpu > 8 || requested_blocks == 0)
        fail("peer shift execution geometry");

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "peer device count");
    if (visible < ngpu) {
        std::cerr << "need " << ngpu << " CUDA devices, visible=" << visible << '\n';
        std::exit(176);
    }
    enable_full_peer_mesh(ngpu);

    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> reference;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, input, reference);

    std::vector<std::uint32_t*> d_shard(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::uint32_t**> d_shard_ptrs(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<Rank64*> d_owner_begin(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<unsigned long long*> d_cycles(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<unsigned long long*> d_rotated(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<unsigned long long*> d_peer(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<int*> d_error(static_cast<std::size_t>(ngpu), nullptr);

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer alloc set device");
        install_tables(tables);
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d_shard[static_cast<std::size_t>(g)], n * sizeof(std::uint32_t)),
           "peer alloc shard");
        ck(cudaMemcpy(
               d_shard[static_cast<std::size_t>(g)],
               input.data() + plan.shard_base[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "peer copy shard input");
    }

    // UVA pointer values are valid on every peer-enabled worker.  Keep a tiny
    // local pointer table on each GPU so the kernel can select the owner shard.
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer metadata set device");
        ck(cudaMalloc(&d_shard_ptrs[static_cast<std::size_t>(g)],
                      ngpu * sizeof(std::uint32_t*)), "peer alloc pointer table");
        ck(cudaMemcpy(d_shard_ptrs[static_cast<std::size_t>(g)], d_shard.data(),
                      ngpu * sizeof(std::uint32_t*), cudaMemcpyHostToDevice),
           "peer copy pointer table");
        ck(cudaMalloc(&d_owner_begin[static_cast<std::size_t>(g)],
                      ngpu * sizeof(Rank64)), "peer alloc owner begin");
        ck(cudaMemcpy(d_owner_begin[static_cast<std::size_t>(g)], plan.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice),
           "peer copy owner begin");
        ck(cudaMalloc(&d_cycles[static_cast<std::size_t>(g)], sizeof(unsigned long long)),
           "peer alloc cycles");
        ck(cudaMalloc(&d_rotated[static_cast<std::size_t>(g)], sizeof(unsigned long long)),
           "peer alloc rotated");
        ck(cudaMalloc(&d_peer[static_cast<std::size_t>(g)], sizeof(unsigned long long)),
           "peer alloc peer count");
        ck(cudaMalloc(&d_error[static_cast<std::size_t>(g)], sizeof(int)),
           "peer alloc error");
        ck(cudaMemset(d_cycles[static_cast<std::size_t>(g)], 0, sizeof(unsigned long long)),
           "peer zero cycles");
        ck(cudaMemset(d_rotated[static_cast<std::size_t>(g)], 0, sizeof(unsigned long long)),
           "peer zero rotated");
        ck(cudaMemset(d_peer[static_cast<std::size_t>(g)], 0, sizeof(unsigned long long)),
           "peer zero peer count");
        ck(cudaMemset(d_error[static_cast<std::size_t>(g)], 0, sizeof(int)),
           "peer zero error");
    }

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 warps_needed = (base_supports + Rank64(ngpu) - 1) / Rank64(ngpu);
    const Rank64 one_pass_blocks =
        (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, one_pass_blocks)));

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer launch set device");
        peer_shifted_tile_cycle_kernel<<<blocks, THREADS>>>(
            d_shard_ptrs[static_cast<std::size_t>(g)], base_supports,
            W, Kwin, S, reverse, ngpu, g,
            d_owner_begin[static_cast<std::size_t>(g)],
            d_cycles[static_cast<std::size_t>(g)],
            d_rotated[static_cast<std::size_t>(g)],
            d_peer[static_cast<std::size_t>(g)],
            d_error[static_cast<std::size_t>(g)]);
        ck(cudaGetLastError(), "peer shift launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer sync set device");
        ck(cudaDeviceSynchronize(), "peer shift sync");
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    unsigned long long cycles = 0, rotated = 0, peer_values = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer counters set device");
        int error = 0;
        unsigned long long c = 0, r = 0, p = 0;
        ck(cudaMemcpy(&error, d_error[static_cast<std::size_t>(g)], sizeof(error),
                      cudaMemcpyDeviceToHost), "peer copy error");
        ck(cudaMemcpy(&c, d_cycles[static_cast<std::size_t>(g)], sizeof(c),
                      cudaMemcpyDeviceToHost), "peer copy cycles");
        ck(cudaMemcpy(&r, d_rotated[static_cast<std::size_t>(g)], sizeof(r),
                      cudaMemcpyDeviceToHost), "peer copy rotated");
        ck(cudaMemcpy(&p, d_peer[static_cast<std::size_t>(g)], sizeof(p),
                      cudaMemcpyDeviceToHost), "peer copy peer count");
        if (error) fail("peer shift device error=" + std::to_string(error) +
                        " gpu=" + std::to_string(g));
        cycles += c;
        rotated += r;
        peer_values += p;
    }

    std::vector<std::uint32_t> output(reference.size());
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer output set device");
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               d_shard[static_cast<std::size_t>(g)], n * sizeof(std::uint32_t),
               cudaMemcpyDeviceToHost), "peer copy output shard");
    }
    if (output != reference) fail("peer shift redistribution mismatch");

    std::cout << "W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " blocks_per_gpu=" << blocks
              << " cycles=" << cycles
              << " rotated_values=" << rotated
              << " peer_hops=" << peer_values
              << " peer_GiB=" << double(peer_values) * 4.0 / double(1ULL << 30)
              << " elapsed_ms=" << ms
              << " real_peer_allocations=1"
              << " base_support_partitioned=1"
              << " second_state_buffer_bytes=0"
              << " run_table_bytes=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "peer free set device");
        cudaFree(d_error[static_cast<std::size_t>(g)]);
        cudaFree(d_peer[static_cast<std::size_t>(g)]);
        cudaFree(d_rotated[static_cast<std::size_t>(g)]);
        cudaFree(d_cycles[static_cast<std::size_t>(g)]);
        cudaFree(d_owner_begin[static_cast<std::size_t>(g)]);
        cudaFree(d_shard_ptrs[static_cast<std::size_t>(g)]);
        cudaFree(d_shard[static_cast<std::size_t>(g)]);
    }
}

void print_peer_plan(int W, int Kwin, int S, int ngpu) {
    if (W < 7 || W > RP_MAX_W || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || ngpu < 2 || ngpu > 8)
        fail("peer shift plan geometry");
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    Rank64 max_states = 0;
    for (Rank64 n : plan.owner_size) max_states = std::max(max_states, n);
    std::cout << "peer-shift-plan"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " max_shard_GiB=" << double(max_states) * 4.0 / double(1ULL << 30)
              << " base_supports=" << (Rank64(1) << (W - 2))
              << " main_cycle_order="
              << (Kwin + S + 2) / std::gcd(Kwin + S + 2, S)
              << " blocked_cycle_order="
              << (Kwin + S) / std::gcd(Kwin + S, S)
              << " requires_full_peer_mesh=1"
              << " state_buffers_per_gpu=1"
              << " run_table_bytes=0 visited_bytes=0\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 3;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 2;
    const bool plan_only = has_arg(argc, argv, "--plan-only");

    if (plan_only) {
        print_peer_plan(W, Kwin, S, ngpu);
        return 0;
    }
    run_peer_shift_cycle_probe(W, Kwin, S, false, ngpu, blocks);
    run_peer_shift_cycle_probe(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK grouped_shift_cycle_real_peer=1\n";
    return 0;
}
