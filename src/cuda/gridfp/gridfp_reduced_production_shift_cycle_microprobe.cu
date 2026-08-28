#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_tile_microprobe_main_unused
#include "gridfp_reduced_production_grouped_tile_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_shift_cycle_device.cuh"

namespace {

__global__ void shifted_tile_cycle_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int gpu_id,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ shard_base,
    unsigned long long* __restrict__ cycles,
    unsigned long long* __restrict__ rotated_values,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
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
                if (lane == 0) set_error(error, 151);
                continue;
            }
            if (cycle_len <= 1) continue;

            const DeviceKey leader = equal_run_key0_device(
                run.support, blocked, W, q, reverse);
            const GroupedDeviceRank lr = grouped_rank_device(
                leader, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
            if (lr.owner != gpu_id) continue;

            const int occupied = __popc(run.support);
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            const Rank64 leader_pos = shard_base[lr.owner] + lr.local;

            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = state[leader_pos + i];
                std::uint32_t cur_support = shift_next_support_device(
                    run.support, blocked, W, q, Kwin, S, reverse);
                int hops = 1;
                while (cur_support != run.support) {
                    const DeviceKey cur = equal_run_key0_device(
                        cur_support, blocked, W, q, reverse);
                    const GroupedDeviceRank cr = grouped_rank_device(
                        cur, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
                    const Rank64 pos = shard_base[cr.owner] + cr.local + i;
                    const std::uint32_t next_value = state[pos];
                    state[pos] = temp;
                    temp = next_value;
                    cur_support = shift_next_support_device(
                        cur_support, blocked, W, q, Kwin, S, reverse);
                    ++hops;
                    if (hops > cycle_len) {
                        set_error(error, 152);
                        break;
                    }
                }
                state[leader_pos + i] = temp;
            }
            __syncwarp();
            if (lane == 0) {
                atomicAdd(cycles, 1ULL);
                atomicAdd(rotated_values, static_cast<unsigned long long>(pc) * cycle_len);
            }
        }
    }
}

void build_shift_boundary_vectors(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const ProductionFactorTables& tables,
    const HostTilePlan& plan,
    std::vector<std::uint32_t>& old_layout,
    std::vector<std::uint32_t>& new_layout
) {
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);
    const int new_start = q;
    old_layout.assign(static_cast<std::size_t>(tables.size()), 0);
    new_layout.assign(static_cast<std::size_t>(tables.size()), 0);
    ProductionFactorCodec codec(tables, q - 1);
    const OwnerPlan owner_plan{plan.owner_begin, plan.owner_size};

    for (Rank64 r = 0; r < tables.size(); ++r) {
        const Key key = codec.unrank(r);
        const std::uint32_t value = static_cast<std::uint32_t>(
            1 + (r * 2654435761ULL) % 4294967290ULL);
        const GroupedRank sr = grouped_rank(
            key, tables, W, q, reverse, old_start, Kwin, ngpu, owner_plan);
        const GroupedRank dr = grouped_rank(
            key, tables, W, q, reverse, new_start, Kwin, ngpu, owner_plan);
        old_layout[static_cast<std::size_t>(plan.shard_base[sr.owner] + sr.local)] = value;
        new_layout[static_cast<std::size_t>(plan.shard_base[dr.owner] + dr.local)] = value;
    }
}

void run_shift_cycle_probe(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned blocks
) {
    if (Kwin < 1 || S < 1 || S > Kwin || Kwin + S + 2 > W)
        fail("shift CUDA geometry");
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> reference;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, input, reference);

    std::uint32_t* d_state = nullptr;
    Rank64* d_owner_begin = nullptr;
    Rank64* d_shard_base = nullptr;
    unsigned long long* d_cycles = nullptr;
    unsigned long long* d_rotated = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, tables.size() * sizeof(std::uint32_t)), "shift alloc state");
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "shift alloc owner begin");
    ck(cudaMalloc(&d_shard_base, ngpu * sizeof(Rank64)), "shift alloc shard base");
    ck(cudaMalloc(&d_cycles, sizeof(unsigned long long)), "shift alloc cycles");
    ck(cudaMalloc(&d_rotated, sizeof(unsigned long long)), "shift alloc rotated");
    ck(cudaMalloc(&d_error, sizeof(int)), "shift alloc error");
    ck(cudaMemcpy(d_state, input.data(), tables.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "shift copy input");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "shift copy owner begin");
    ck(cudaMemcpy(d_shard_base, plan.shard_base.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "shift copy shard base");
    ck(cudaMemset(d_cycles, 0, sizeof(unsigned long long)), "shift zero cycles");
    ck(cudaMemset(d_rotated, 0, sizeof(unsigned long long)), "shift zero rotated");
    ck(cudaMemset(d_error, 0, sizeof(int)), "shift zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 one_pass_blocks = (base_supports + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        shifted_tile_cycle_kernel<<<launch_blocks, THREADS>>>(
            d_state, base_supports, W, Kwin, S, reverse, ngpu, g,
            d_owner_begin, d_shard_base, d_cycles, d_rotated, d_error);
        ck(cudaGetLastError(), "shift cycle launch");
    }
    ck(cudaDeviceSynchronize(), "shift cycle sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long cycles = 0, rotated = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "shift copy error");
    ck(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost), "shift copy cycles");
    ck(cudaMemcpy(&rotated, d_rotated, sizeof(rotated), cudaMemcpyDeviceToHost), "shift copy rotated");
    if (error) fail("shift CUDA device error=" + std::to_string(error));

    std::vector<std::uint32_t> output(static_cast<std::size_t>(tables.size()));
    ck(cudaMemcpy(output.data(), d_state, tables.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "shift copy output");
    if (output != reference) fail("shift CUDA redistribution mismatch");

    std::cout << "W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << tables.size()
              << " base_supports=" << base_supports
              << " cycles=" << cycles
              << " rotated_values=" << rotated
              << " main_order=" << (Kwin + S + 2) / std::gcd(Kwin + S + 2, S)
              << " blocked_order=" << (Kwin + S) / std::gcd(Kwin + S, S)
              << " blocks=" << launch_blocks
              << " ms=" << ms
              << " in_place=1 second_state_buffer_bytes=0"
              << " run_table_bytes=0 visited_bytes=0 exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_rotated);
    cudaFree(d_cycles);
    cudaFree(d_shard_base);
    cudaFree(d_owner_begin);
    cudaFree(d_state);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 3;
    const unsigned blocks = argc > 4 ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 8;
    if (W < 7 || W > 11 || Kwin < 2 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || blocks == 0 || ngpu != 8) return 2;

    run_shift_cycle_probe(W, Kwin, S, false, ngpu, blocks);
    run_shift_cycle_probe(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK grouped_shift_cycle_cuda=1\n";
    return 0;
}
