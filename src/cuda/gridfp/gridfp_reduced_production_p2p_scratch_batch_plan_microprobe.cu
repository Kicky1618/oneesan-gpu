#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_traffic_microprobe_main_unused
#include "gridfp_reduced_production_p2p_traffic_microprobe.cu"
#pragma pop_macro("main")

#include <array>
#include <limits>
#include <vector>

namespace {

static constexpr int SCRATCH_MAX_GPU = 8;
static constexpr unsigned SCRATCH_DESCRIPTOR_BYTES = 32u;

__global__ void p2p_scratch_batch_plan_kernel(
    Rank64 base_supports,
    Rank64 base_batch,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const std::uint8_t* __restrict__ owner_lut,
    unsigned long long* __restrict__ scratch_words,
    unsigned long long* __restrict__ descriptors,
    unsigned long long* __restrict__ total_cross_words,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);

    unsigned long long local_total = 0;
    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        const Rank64 batch = base_rank / base_batch;
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            if (cycle_len < 0) {
                set_error(error, 271);
                continue;
            }
            if (cycle_len <= 1) continue;

            const int occupied = __popc(run.support);
            const unsigned long long pc = RP_PRIMITIVE[occupied][1];
            int first_owner = traffic_owner_from_lut_device(
                run.support, W, Kwin, reverse, owner_lut);
            if (first_owner < 0 || first_owner >= ngpu) {
                set_error(error, 272);
                continue;
            }

            int prev_owner = first_owner;
            std::uint32_t support = shift_next_support_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            int hops = 1;
            while (support != run.support && hops < cycle_len) {
                const int owner = traffic_owner_from_lut_device(
                    support, W, Kwin, reverse, owner_lut);
                if (owner < 0 || owner >= ngpu) {
                    set_error(error, 273);
                    break;
                }
                if (owner != prev_owner) {
                    const Rank64 cell = batch * SCRATCH_MAX_GPU + prev_owner;
                    atomicAdd(scratch_words + cell, pc);
                    atomicAdd(descriptors + cell, 1ULL);
                    local_total += pc;
                }
                prev_owner = owner;
                support = shift_next_support_device(
                    support, blocked, W, q, Kwin, S, reverse);
                ++hops;
            }
            if (support != run.support || hops != cycle_len) {
                set_error(error, 274);
                continue;
            }
            if (prev_owner != first_owner) {
                const Rank64 cell = batch * SCRATCH_MAX_GPU + prev_owner;
                atomicAdd(scratch_words + cell, pc);
                atomicAdd(descriptors + cell, 1ULL);
                local_total += pc;
            }
        }
    }
    if (local_total) atomicAdd(total_cross_words, local_total);
}

void run_scratch_batch_plan(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    Rank64 base_batch,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const auto owner_lut = build_traffic_owner_lut(tables, Kwin, ngpu);
    ck(cudaSetDevice(0), "scratch plan set device");
    install_tables(tables);

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 nbatches = (base_supports + base_batch - 1) / base_batch;
    if (!base_batch || nbatches == 0) fail("scratch plan batch geometry");
    const Rank64 cells = nbatches * SCRATCH_MAX_GPU;

    std::uint8_t* d_owner_lut = nullptr;
    unsigned long long* d_words = nullptr;
    unsigned long long* d_desc = nullptr;
    unsigned long long* d_total = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_owner_lut, owner_lut.size()), "scratch plan alloc LUT");
    ck(cudaMemcpy(d_owner_lut, owner_lut.data(), owner_lut.size(),
                  cudaMemcpyHostToDevice), "scratch plan copy LUT");
    ck(cudaMalloc(&d_words, cells * sizeof(unsigned long long)),
       "scratch plan alloc words");
    ck(cudaMalloc(&d_desc, cells * sizeof(unsigned long long)),
       "scratch plan alloc descriptors");
    ck(cudaMalloc(&d_total, sizeof(unsigned long long)),
       "scratch plan alloc total");
    ck(cudaMalloc(&d_error, sizeof(int)), "scratch plan alloc error");
    ck(cudaMemset(d_words, 0, cells * sizeof(unsigned long long)),
       "scratch plan zero words");
    ck(cudaMemset(d_desc, 0, cells * sizeof(unsigned long long)),
       "scratch plan zero descriptors");
    ck(cudaMemset(d_total, 0, sizeof(unsigned long long)),
       "scratch plan zero total");
    ck(cudaMemset(d_error, 0, sizeof(int)), "scratch plan zero error");

    const Rank64 needed_blocks =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));

    const auto t0 = std::chrono::steady_clock::now();
    p2p_scratch_batch_plan_kernel<<<blocks, THREADS>>>(
        base_supports, base_batch, W, Kwin, S, reverse, ngpu,
        d_owner_lut, d_words, d_desc, d_total, d_error);
    ck(cudaGetLastError(), "scratch plan launch");
    ck(cudaDeviceSynchronize(), "scratch plan sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long total = 0;
    std::vector<unsigned long long> words(static_cast<std::size_t>(cells));
    std::vector<unsigned long long> desc(static_cast<std::size_t>(cells));
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "scratch plan copy error");
    ck(cudaMemcpy(&total, d_total, sizeof(total), cudaMemcpyDeviceToHost),
       "scratch plan copy total");
    ck(cudaMemcpy(words.data(), d_words, cells * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "scratch plan copy words");
    ck(cudaMemcpy(desc.data(), d_desc, cells * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "scratch plan copy descriptors");
    if (error) fail("scratch batch plan device error=" + std::to_string(error));

    unsigned long long sum_words = 0;
    unsigned long long max_words = 0;
    unsigned long long max_desc = 0;
    unsigned long long max_batch_total_words = 0;
    Rank64 max_batch = 0;
    int max_gpu = -1;
    for (Rank64 b = 0; b < nbatches; ++b) {
        unsigned long long batch_total = 0;
        for (int g = 0; g < ngpu; ++g) {
            const Rank64 cell = b * SCRATCH_MAX_GPU + g;
            const auto w = words[static_cast<std::size_t>(cell)];
            const auto d = desc[static_cast<std::size_t>(cell)];
            sum_words += w;
            batch_total += w;
            if (w > max_words) {
                max_words = w;
                max_batch = b;
                max_gpu = g;
            }
            max_desc = std::max(max_desc, d);
        }
        max_batch_total_words = std::max(max_batch_total_words, batch_total);
    }
    if (sum_words != total) fail("scratch batch plan total mismatch");

    const double max_payload_bytes = double(max_words) * sizeof(std::uint32_t);
    const double max_desc_bytes = double(max_desc) * SCRATCH_DESCRIPTOR_BYTES;
    const double max_gpu_bytes = max_payload_bytes + max_desc_bytes;
    const char* direction = reverse ? "reverse" : "forward";
    std::cout << "gridfp-p2p-scratch-batch-plan"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << direction
              << " ngpu=" << ngpu
              << " base_supports=" << base_supports
              << " base_batch=" << base_batch
              << " batches=" << nbatches
              << " logical_peer_GiB="
              << double(total) * sizeof(std::uint32_t) / double(1ULL << 30)
              << " max_gpu_scratch_payload_MiB="
              << max_payload_bytes / double(1ULL << 20)
              << " max_gpu_descriptors=" << max_desc
              << " max_gpu_descriptor_MiB="
              << max_desc_bytes / double(1ULL << 20)
              << " max_gpu_total_scratch_MiB="
              << max_gpu_bytes / double(1ULL << 20)
              << " max_at_batch=" << max_batch
              << " max_at_gpu=" << max_gpu
              << " max_batch_all_gpu_payload_MiB="
              << double(max_batch_total_words) * sizeof(std::uint32_t) /
                     double(1ULL << 20)
              << " blocks=" << blocks
              << " wall_ms=" << ms
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0"
              << " two_phase_kernel_barrier=1 exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_total);
    cudaFree(d_desc);
    cudaFree(d_words);
    cudaFree(d_owner_lut);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 13;
    const int S = argc > 3 ? std::atoi(argv[3]) : 13;
    const Rank64 base_batch = argc > 4
        ? static_cast<Rank64>(std::strtoull(argv[4], nullptr, 10)) : Rank64(65536);
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 4096u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 6 || W > RP_MAX_W || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || base_batch == 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "scratch plan device count");
    if (visible < 1) return 3;

    run_scratch_batch_plan(W, Kwin, S, false, ngpu, base_batch, blocks);
    run_scratch_batch_plan(W, Kwin, S, true, ngpu, base_batch, blocks);
    std::cout << "ALL_OK gridfp_p2p_scratch_batch_plan=1\n";
    return 0;
}
