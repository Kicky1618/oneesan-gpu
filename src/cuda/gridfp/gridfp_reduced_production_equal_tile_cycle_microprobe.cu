#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_tile_microprobe_main_unused
#include "gridfp_reduced_production_grouped_tile_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_equal_tile_cycle_device.cuh"

namespace {

__global__ void equal_tile_cycle_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int gpu_id,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ shard_base,
    unsigned long long* __restrict__ cycles,
    unsigned long long* __restrict__ moved_values,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int q = reverse ? 1 + K : W - 1 - K;
    const int old_start = reverse ? 1 : W - 1;

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const int cycle_len = equal_cycle_leader_length_device(
                run.support, run.blocked != 0, W, K, reverse);
            if (cycle_len < 0) {
                if (lane == 0) set_error(error, 131);
                continue;
            }
            if (cycle_len <= 1) continue;

            const DeviceKey leader = equal_run_key0_device(
                run.support, run.blocked != 0, W, q, reverse);
            const GroupedDeviceRank lr = grouped_rank_device(
                leader, W, q, reverse, old_start, K, ngpu, owner_begin);
            if (lr.owner != gpu_id) continue;

            const int occupied = __popc(run.support);
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            const Rank64 leader_pos = shard_base[lr.owner] + lr.local;

            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = state[leader_pos + i];
                std::uint32_t cur_support = equal_next_support_device(
                    run.support, run.blocked != 0, W, K, reverse);
                int hops = 1;
                while (cur_support != run.support) {
                    const DeviceKey cur = equal_run_key0_device(
                        cur_support, run.blocked != 0, W, q, reverse);
                    const GroupedDeviceRank cr = grouped_rank_device(
                        cur, W, q, reverse, old_start, K, ngpu, owner_begin);
                    const Rank64 pos = shard_base[cr.owner] + cr.local + i;
                    const std::uint32_t next_value = state[pos];
                    state[pos] = temp;
                    temp = next_value;
                    cur_support = equal_next_support_device(
                        cur_support, run.blocked != 0, W, K, reverse);
                    ++hops;
                    if (hops > cycle_len) {
                        set_error(error, 132);
                        break;
                    }
                }
                state[leader_pos + i] = temp;
            }
            __syncwarp();
            if (lane == 0) {
                atomicAdd(cycles, 1ULL);
                atomicAdd(moved_values, static_cast<unsigned long long>(pc) * cycle_len);
            }
        }
    }
}

void build_grouped_boundary_vectors(
    int W,
    int K,
    bool reverse,
    int ngpu,
    const ProductionFactorTables& tables,
    const HostTilePlan& plan,
    std::vector<std::uint32_t>& old_layout,
    std::vector<std::uint32_t>& new_layout
) {
    const int q = reverse ? 1 + K : W - 1 - K;
    const int old_start = reverse ? 1 : W - 1;
    const int new_start = q;
    old_layout.assign(static_cast<std::size_t>(tables.size()), 0);
    new_layout.assign(static_cast<std::size_t>(tables.size()), 0);
    ProductionFactorCodec codec(tables, q - 1);

    for (Rank64 r = 0; r < tables.size(); ++r) {
        const Key key = codec.unrank(r);
        const std::uint32_t value = static_cast<std::uint32_t>(1 + (r * 2654435761ULL) % 4294967290ULL);
        const GroupedRank sr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu,
            OwnerPlan{plan.owner_begin, plan.owner_size});
        const GroupedRank dr = grouped_rank(
            key, tables, W, q, reverse, new_start, K, ngpu,
            OwnerPlan{plan.owner_begin, plan.owner_size});
        old_layout[static_cast<std::size_t>(plan.shard_base[sr.owner] + sr.local)] = value;
        new_layout[static_cast<std::size_t>(plan.shard_base[dr.owner] + dr.local)] = value;
    }
}

void run_equal_tile_cycle_probe(
    int W,
    int K,
    bool reverse,
    int ngpu,
    unsigned blocks
) {
    if (2 * K > W - 3) fail("cycle CUDA equal tile range");
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> reference;
    build_grouped_boundary_vectors(W, K, reverse, ngpu, tables, plan, input, reference);

    std::uint32_t* d_state = nullptr;
    Rank64* d_owner_begin = nullptr;
    Rank64* d_shard_base = nullptr;
    unsigned long long* d_cycles = nullptr;
    unsigned long long* d_moved = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, tables.size() * sizeof(std::uint32_t)), "cycle alloc state");
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "cycle alloc owner begin");
    ck(cudaMalloc(&d_shard_base, ngpu * sizeof(Rank64)), "cycle alloc shard base");
    ck(cudaMalloc(&d_cycles, sizeof(unsigned long long)), "cycle alloc count");
    ck(cudaMalloc(&d_moved, sizeof(unsigned long long)), "cycle alloc moved");
    ck(cudaMalloc(&d_error, sizeof(int)), "cycle alloc error");
    ck(cudaMemcpy(d_state, input.data(), tables.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "cycle copy input");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "cycle copy owner begin");
    ck(cudaMemcpy(d_shard_base, plan.shard_base.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "cycle copy shard base");
    ck(cudaMemset(d_cycles, 0, sizeof(unsigned long long)), "cycle zero count");
    ck(cudaMemset(d_moved, 0, sizeof(unsigned long long)), "cycle zero moved");
    ck(cudaMemset(d_error, 0, sizeof(int)), "cycle zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 one_pass_blocks = (base_supports + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        equal_tile_cycle_kernel<<<launch_blocks, THREADS>>>(
            d_state, base_supports, W, K, reverse, ngpu, g,
            d_owner_begin, d_shard_base, d_cycles, d_moved, d_error);
        ck(cudaGetLastError(), "cycle kernel launch");
    }
    ck(cudaDeviceSynchronize(), "cycle kernel sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long cycles = 0, moved = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "cycle copy error");
    ck(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost), "cycle copy count");
    ck(cudaMemcpy(&moved, d_moved, sizeof(moved), cudaMemcpyDeviceToHost), "cycle copy moved");
    if (error) fail("cycle CUDA device error=" + std::to_string(error));

    std::vector<std::uint32_t> output(static_cast<std::size_t>(tables.size()));
    ck(cudaMemcpy(output.data(), d_state, tables.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cycle copy output");
    if (output != reference) fail("cycle CUDA redistribution mismatch");

    std::cout << "W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << tables.size()
              << " base_supports=" << base_supports
              << " cycles=" << cycles
              << " rotated_values=" << moved
              << " blocks=" << launch_blocks
              << " ms=" << ms
              << " in_place=1 second_state_buffer_bytes=0"
              << " run_table_bytes=0 visited_bytes=0"
              << " exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_moved);
    cudaFree(d_cycles);
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
    if (W < 7 || W > 11 || K < 2 || 2 * K > W - 3 || blocks == 0 || ngpu != 8) return 2;

    run_equal_tile_cycle_probe(W, K, false, ngpu, blocks);
    run_equal_tile_cycle_probe(W, K, true, ngpu, blocks);
    std::cout << "ALL_OK grouped_equal_tile_cycle_cuda=1\n";
    return 0;
}
