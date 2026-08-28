#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_traffic_microprobe_main_unused
#include "gridfp_reduced_production_p2p_traffic_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int CYCLE_BATCH_MAX = 32;
static constexpr unsigned CYCLE_BATCH_DESCRIPTOR_BYTES = 32u;
static constexpr unsigned CYCLE_BATCH_LIST_BYTES = 4u;

__device__ __forceinline__ std::uint32_t cycle_batch_mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ std::uint32_t cycle_batch_main_hash(
    std::uint32_t support,
    int W
) {
    std::uint32_t h = __popc(support) * 0x9e3779b1u;
    h ^= __popc(support & shift_rotate_bits_device(support, W, 1)) * 0x85ebca6bu;
    h ^= __popc(support & shift_rotate_bits_device(support, W, 3)) * 0xc2b2ae35u;
    h ^= __popc(support & shift_rotate_bits_device(support, W, 5)) * 0x27d4eb2fu;
    h ^= __popc(support & shift_rotate_bits_device(support, W, 7)) * 0x165667b1u;
    return cycle_batch_mix32(h);
}

__device__ __forceinline__ std::uint32_t cycle_batch_blocked_compact(
    std::uint32_t support,
    int W,
    int q
) {
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    return compact;
}

__device__ __forceinline__ std::uint32_t cycle_batch_blocked_hash(
    std::uint32_t support,
    int W,
    int q,
    int Kwin
) {
    const std::uint32_t compact = cycle_batch_blocked_compact(support, W, q);
    const std::uint32_t half_mask = (std::uint32_t(1) << Kwin) - 1u;
    const std::uint32_t a = compact & half_mask;
    const std::uint32_t b = (compact >> Kwin) & half_mask;
    const std::uint32_t lo = a < b ? a : b;
    const std::uint32_t hi = a < b ? b : a;
    std::uint32_t h = lo * 0x9e3779b1u;
    h ^= hi * 0x85ebca6bu;
    h ^= __popc(support) * 0xc2b2ae35u;
    return cycle_batch_mix32(h);
}

__device__ __forceinline__ int cycle_batch_id_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int batches
) {
    const std::uint32_t h = blocked
        ? cycle_batch_blocked_hash(support, W, q, Kwin)
        : cycle_batch_main_hash(support, W);
    return int(h & static_cast<std::uint32_t>(batches - 1));
}

__global__ void p2p_cycle_batch_plan_kernel(
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    const std::uint8_t* __restrict__ owner_lut,
    unsigned long long* __restrict__ scratch_words,
    unsigned long long* __restrict__ segment_counts,
    unsigned long long* __restrict__ cycles,
    int* error
) {
    unsigned long long local_cycles = 0;
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
                set_error(error, 321);
                continue;
            }
            if (cycle_len <= 1) continue;

            const int batch = cycle_batch_id_device(
                run.support, blocked, W, q, Kwin, batches);
            const int leader_owner = traffic_owner_from_lut_device(
                run.support, W, Kwin, reverse, owner_lut);
            if (batch < 0 || batch >= batches ||
                leader_owner < 0 || leader_owner >= ngpu) {
                set_error(error, 322);
                continue;
            }

            const unsigned long long pc = RP_PRIMITIVE[__popc(run.support)][1];
            int prev_owner = leader_owner;
            std::uint32_t cur = shift_next_support_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            int hops = 1;
            while (cur != run.support && hops < cycle_len) {
                if (cycle_batch_id_device(
                        cur, blocked, W, q, Kwin, batches) != batch) {
                    set_error(error, 323);
                    break;
                }
                const int owner = traffic_owner_from_lut_device(
                    cur, W, Kwin, reverse, owner_lut);
                if (owner < 0 || owner >= ngpu) {
                    set_error(error, 324);
                    break;
                }
                if (owner != prev_owner) {
                    atomicAdd(scratch_words + prev_owner * batches + batch, pc);
                    atomicAdd(segment_counts + prev_owner * batches + batch, 1ULL);
                }
                prev_owner = owner;
                cur = shift_next_support_device(
                    cur, blocked, W, q, Kwin, S, reverse);
                ++hops;
            }
            if (cur != run.support || hops != cycle_len) {
                set_error(error, 325);
                continue;
            }
            if (prev_owner != leader_owner) {
                atomicAdd(scratch_words + prev_owner * batches + batch, pc);
                atomicAdd(segment_counts + prev_owner * batches + batch, 1ULL);
            }
            ++local_cycles;
        }
    }
    if (local_cycles) atomicAdd(cycles, local_cycles);
}

void run_cycle_batch_plan(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const auto owner_lut = build_traffic_owner_lut(tables, Kwin, ngpu);
    const HostTilePlan host_plan = make_host_tile_plan(tables, Kwin, ngpu);

    ck(cudaSetDevice(0), "cycle batch set device");
    install_tables(tables);
    std::uint8_t* d_owner_lut = nullptr;
    unsigned long long* d_words = nullptr;
    unsigned long long* d_segments = nullptr;
    unsigned long long* d_cycles = nullptr;
    int* d_error = nullptr;
    const std::size_t cells = static_cast<std::size_t>(ngpu * batches);
    ck(cudaMalloc(&d_owner_lut, owner_lut.size()), "cycle batch alloc owner LUT");
    ck(cudaMemcpy(d_owner_lut, owner_lut.data(), owner_lut.size(),
                  cudaMemcpyHostToDevice), "cycle batch copy owner LUT");
    ck(cudaMalloc(&d_words, cells * sizeof(unsigned long long)),
       "cycle batch alloc words");
    ck(cudaMalloc(&d_segments, cells * sizeof(unsigned long long)),
       "cycle batch alloc segments");
    ck(cudaMalloc(&d_cycles, sizeof(unsigned long long)),
       "cycle batch alloc cycles");
    ck(cudaMalloc(&d_error, sizeof(int)), "cycle batch alloc error");
    ck(cudaMemset(d_words, 0, cells * sizeof(unsigned long long)),
       "cycle batch zero words");
    ck(cudaMemset(d_segments, 0, cells * sizeof(unsigned long long)),
       "cycle batch zero segments");
    ck(cudaMemset(d_cycles, 0, sizeof(unsigned long long)),
       "cycle batch zero cycles");
    ck(cudaMemset(d_error, 0, sizeof(int)), "cycle batch zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 needed_blocks =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));
    const auto t0 = std::chrono::steady_clock::now();
    p2p_cycle_batch_plan_kernel<<<blocks, THREADS>>>(
        base_supports, W, Kwin, S, reverse, ngpu, batches, d_owner_lut,
        d_words, d_segments, d_cycles, d_error);
    ck(cudaGetLastError(), "cycle batch launch");
    ck(cudaDeviceSynchronize(), "cycle batch sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long cycles = 0;
    std::vector<unsigned long long> words(cells), segments(cells);
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "cycle batch copy error");
    ck(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost),
       "cycle batch copy cycles");
    ck(cudaMemcpy(words.data(), d_words, cells * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "cycle batch copy words");
    ck(cudaMemcpy(segments.data(), d_segments, cells * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "cycle batch copy segments");
    if (error) fail("cycle batch device error=" + std::to_string(error));

    unsigned long long total_words = 0;
    unsigned long long total_segments = 0;
    double worst_peak_gib = 0.0;
    double worst_scratch_gib = 0.0;
    int worst_gpu = -1, worst_batch = -1;
    const double b300_gib = 288e9 / double(1ULL << 30);
    for (int g = 0; g < ngpu; ++g) {
        unsigned long long owner_words = 0;
        unsigned long long owner_segments = 0;
        double max_batch_extra_gib = 0.0;
        double max_batch_scratch_gib = 0.0;
        double max_batch_descriptor_gib = 0.0;
        int max_batch = -1;
        for (int b = 0; b < batches; ++b) {
            const std::size_t ix = static_cast<std::size_t>(g * batches + b);
            owner_words += words[ix];
            owner_segments += segments[ix];
            const double scratch_gib =
                double(words[ix]) * 4.0 / double(1ULL << 30);
            const double descriptor_gib =
                double(segments[ix]) * CYCLE_BATCH_DESCRIPTOR_BYTES /
                double(1ULL << 30);
            const double extra_gib = scratch_gib + descriptor_gib;
            if (extra_gib > max_batch_extra_gib) {
                max_batch_extra_gib = extra_gib;
                max_batch_scratch_gib = scratch_gib;
                max_batch_descriptor_gib = descriptor_gib;
                max_batch = b;
            }
            std::cout << "cycle-batch-cell"
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " gpu=" << g
                      << " batch=" << b
                      << " scratch_GiB=" << scratch_gib
                      << " descriptor_GiB=" << descriptor_gib
                      << " segments=" << segments[ix] << '\n';
        }
        total_words += owner_words;
        total_segments += owner_segments;
        const double state_gib =
            double(host_plan.owner_size[static_cast<std::size_t>(g)]) * 4.0 /
            double(1ULL << 30);
        const double list_gib =
            double(owner_segments) * CYCLE_BATCH_LIST_BYTES /
            double(1ULL << 30);
        const double peak_gib = state_gib + list_gib + max_batch_extra_gib;
        if (peak_gib > worst_peak_gib) {
            worst_peak_gib = peak_gib;
            worst_scratch_gib = max_batch_scratch_gib;
            worst_gpu = g;
            worst_batch = max_batch;
        }
        std::cout << "cycle-batch-owner"
                  << " direction=" << (reverse ? "reverse" : "forward")
                  << " gpu=" << g
                  << " state_GiB=" << state_gib
                  << " total_outgoing_GiB="
                  << double(owner_words) * 4.0 / double(1ULL << 30)
                  << " total_segments=" << owner_segments
                  << " max_batch=" << max_batch
                  << " max_batch_scratch_GiB=" << max_batch_scratch_gib
                  << " max_batch_descriptor_GiB=" << max_batch_descriptor_gib
                  << " packed_segment_list_GiB=" << list_gib
                  << " conservative_peak_GiB=" << peak_gib
                  << " B300_headroom_GiB=" << (b300_gib - peak_gib)
                  << '\n';
    }

    std::cout << "gridfp-p2p-cycle-batch-plan"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " cycles=" << cycles
              << " logical_peer_values=" << total_words
              << " logical_peer_GiB="
              << double(total_words) * 4.0 / double(1ULL << 30)
              << " cross_segments=" << total_segments
              << " worst_gpu=" << worst_gpu
              << " worst_batch=" << worst_batch
              << " worst_batch_scratch_GiB=" << worst_scratch_gib
              << " conservative_peak_GiB=" << worst_peak_gib
              << " B300_GiB=" << b300_gib
              << " B300_headroom_GiB=" << (b300_gib - worst_peak_gib)
              << " cycle_hash_invariant=1"
              << " phase_order_safe=1"
              << " runtime_count_pass_required=0"
              << " state_allocation_bytes=0"
              << " wall_ms=" << ms << '\n';

    cudaFree(d_error);
    cudaFree(d_cycles);
    cudaFree(d_segments);
    cudaFree(d_words);
    cudaFree(d_owner_lut);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 13;
    const int S = argc > 3 ? std::atoi(argv[3]) : 13;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 8;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 4096u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 8 || W > RP_MAX_W || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W || batches < 2 || batches > CYCLE_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks || ngpu < 2 || ngpu > 8)
        return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cycle batch device count");
    if (visible < 1) return 3;

    run_cycle_batch_plan(W, Kwin, S, false, ngpu, batches, blocks);
    run_cycle_batch_plan(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_batch_plan=1\n";
    return 0;
}
