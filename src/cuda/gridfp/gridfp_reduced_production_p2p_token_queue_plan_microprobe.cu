#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_traffic_microprobe_main_unused
#include "gridfp_reduced_production_p2p_traffic_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int TOKEN_PLAN_CHOICES = 4;
static constexpr unsigned TOKEN_PLAN_WORDS[TOKEN_PLAN_CHOICES] = {
    256u, 1024u, 4096u, 16384u
};
static constexpr unsigned TOKEN_PLAN_METADATA_BYTES = 32u;

__global__ void p2p_token_queue_plan_kernel(
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const std::uint8_t* __restrict__ owner_lut,
    unsigned long long* __restrict__ logical_values,
    unsigned long long* __restrict__ messages,
    unsigned long long* __restrict__ cross_cycles,
    int* error
) {
    unsigned long long local_values = 0;
    unsigned long long local_messages[TOKEN_PLAN_CHOICES]{};
    unsigned long long local_cross_cycles = 0;

    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
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
                set_error(error, 251);
                continue;
            }
            if (cycle_len <= 1) continue;

            const int leader_owner = traffic_owner_from_lut_device(
                run.support, W, Kwin, reverse, owner_lut);
            int prev_owner = leader_owner;
            unsigned cross_edges = 0;
            std::uint32_t support = shift_next_support_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            int hops = 1;
            while (support != run.support && hops < cycle_len) {
                const int owner = traffic_owner_from_lut_device(
                    support, W, Kwin, reverse, owner_lut);
                if (owner < 0 || owner >= ngpu) {
                    set_error(error, 252);
                    break;
                }
                cross_edges += owner != prev_owner;
                prev_owner = owner;
                support = shift_next_support_device(
                    support, blocked, W, q, Kwin, S, reverse);
                ++hops;
            }
            if (support != run.support || hops != cycle_len) {
                set_error(error, 253);
                continue;
            }
            cross_edges += prev_owner != leader_owner;
            if (!cross_edges) continue;

            const int occupied = __popc(run.support);
            const unsigned long long pc = RP_PRIMITIVE[occupied][1];
            local_values += pc * static_cast<unsigned long long>(cross_edges);
            ++local_cross_cycles;
            for (int c = 0; c < TOKEN_PLAN_CHOICES; ++c) {
                const unsigned long long chunk = TOKEN_PLAN_WORDS[c];
                const unsigned long long chunks = (pc + chunk - 1) / chunk;
                local_messages[c] +=
                    chunks * static_cast<unsigned long long>(cross_edges);
            }
        }
    }

    if (local_values) atomicAdd(logical_values, local_values);
    if (local_cross_cycles) atomicAdd(cross_cycles, local_cross_cycles);
    for (int c = 0; c < TOKEN_PLAN_CHOICES; ++c)
        if (local_messages[c]) atomicAdd(messages + c, local_messages[c]);
}

void run_token_queue_plan(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const auto owner_lut = build_traffic_owner_lut(tables, Kwin, ngpu);
    ck(cudaSetDevice(0), "token plan set device");
    install_tables(tables);

    std::uint8_t* d_owner_lut = nullptr;
    unsigned long long* d_logical = nullptr;
    unsigned long long* d_messages = nullptr;
    unsigned long long* d_cross_cycles = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_owner_lut, owner_lut.size()), "token plan alloc LUT");
    ck(cudaMemcpy(d_owner_lut, owner_lut.data(), owner_lut.size(),
                  cudaMemcpyHostToDevice), "token plan copy LUT");
    ck(cudaMalloc(&d_logical, sizeof(unsigned long long)),
       "token plan alloc logical");
    ck(cudaMalloc(&d_messages,
                  TOKEN_PLAN_CHOICES * sizeof(unsigned long long)),
       "token plan alloc messages");
    ck(cudaMalloc(&d_cross_cycles, sizeof(unsigned long long)),
       "token plan alloc cross cycles");
    ck(cudaMalloc(&d_error, sizeof(int)), "token plan alloc error");
    ck(cudaMemset(d_logical, 0, sizeof(unsigned long long)),
       "token plan zero logical");
    ck(cudaMemset(d_messages, 0,
                  TOKEN_PLAN_CHOICES * sizeof(unsigned long long)),
       "token plan zero messages");
    ck(cudaMemset(d_cross_cycles, 0, sizeof(unsigned long long)),
       "token plan zero cross cycles");
    ck(cudaMemset(d_error, 0, sizeof(int)), "token plan zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 needed_blocks =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));

    const auto t0 = std::chrono::steady_clock::now();
    p2p_token_queue_plan_kernel<<<blocks, THREADS>>>(
        base_supports, W, Kwin, S, reverse, ngpu, d_owner_lut,
        d_logical, d_messages, d_cross_cycles, d_error);
    ck(cudaGetLastError(), "token plan launch");
    ck(cudaDeviceSynchronize(), "token plan sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long logical = 0, cross_cycles = 0;
    unsigned long long messages[TOKEN_PLAN_CHOICES]{};
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "token plan copy error");
    ck(cudaMemcpy(&logical, d_logical, sizeof(logical), cudaMemcpyDeviceToHost),
       "token plan copy logical");
    ck(cudaMemcpy(messages, d_messages, sizeof(messages), cudaMemcpyDeviceToHost),
       "token plan copy messages");
    ck(cudaMemcpy(&cross_cycles, d_cross_cycles, sizeof(cross_cycles),
                  cudaMemcpyDeviceToHost), "token plan copy cross cycles");
    if (error) fail("token queue plan device error=" + std::to_string(error));

    const double logical_bytes = double(logical) * sizeof(std::uint32_t);
    std::cout << "gridfp-p2p-token-queue-plan"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " cross_cycles=" << cross_cycles
              << " logical_peer_GiB="
              << logical_bytes / double(1ULL << 30)
              << " blocks=" << blocks
              << " wall_ms=" << ms
              << " state_allocation_bytes=0 exact=OK\n";

    for (int c = 0; c < TOKEN_PLAN_CHOICES; ++c) {
        const double metadata_bytes =
            double(messages[c]) * TOKEN_PLAN_METADATA_BYTES;
        const double avg_payload = messages[c]
            ? logical_bytes / double(messages[c]) : 0.0;
        std::cout << "token-chunk-plan"
                  << " direction=" << (reverse ? "reverse" : "forward")
                  << " chunk_words=" << TOKEN_PLAN_WORDS[c]
                  << " chunk_bytes="
                  << TOKEN_PLAN_WORDS[c] * sizeof(std::uint32_t)
                  << " messages=" << messages[c]
                  << " doorbells_unbatched=" << messages[c]
                  << " doorbells_batch8=" << (messages[c] + 7ULL) / 8ULL
                  << " avg_payload_bytes=" << avg_payload
                  << " metadata_GiB="
                  << metadata_bytes / double(1ULL << 30)
                  << " metadata_over_payload="
                  << (logical_bytes ? metadata_bytes / logical_bytes : 0.0)
                  << '\n';
    }

    cudaFree(d_error);
    cudaFree(d_cross_cycles);
    cudaFree(d_messages);
    cudaFree(d_logical);
    cudaFree(d_owner_lut);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 13;
    const int S = argc > 3 ? std::atoi(argv[3]) : 13;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 4096u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 8;
    if (W < 6 || W > RP_MAX_W || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > 8) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "token plan device count");
    if (visible < 1) return 3;

    run_token_queue_plan(W, Kwin, S, false, ngpu, blocks);
    run_token_queue_plan(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK gridfp_p2p_token_queue_plan=1\n";
    return 0;
}
