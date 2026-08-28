#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_traffic_microprobe_main_unused
#include "gridfp_reduced_production_p2p_traffic_microprobe.cu"
#pragma pop_macro("main")

#include <array>

namespace {

static constexpr int TRAFFIC_MATRIX_GPU = 8;
static constexpr int TRAFFIC_MATRIX_SIZE = TRAFFIC_MATRIX_GPU * TRAFFIC_MATRIX_GPU;

__global__ void p2p_cycle_traffic_matrix_kernel(
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const std::uint8_t* __restrict__ owner_lut,
    unsigned long long* __restrict__ global_cross_matrix,
    unsigned long long* __restrict__ global_direct_matrix,
    unsigned long long* __restrict__ cycles,
    unsigned long long* __restrict__ rotated_values,
    int* error
) {
    __shared__ unsigned long long cross_matrix[TRAFFIC_MATRIX_SIZE];
    __shared__ unsigned long long direct_matrix[TRAFFIC_MATRIX_SIZE];

    for (int i = threadIdx.x; i < TRAFFIC_MATRIX_SIZE; i += blockDim.x) {
        cross_matrix[i] = 0;
        direct_matrix[i] = 0;
    }
    __syncthreads();

    unsigned long long local_cycles = 0;
    unsigned long long local_rotated = 0;
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
                set_error(error, 211);
                continue;
            }
            if (cycle_len <= 1) continue;

            const int occupied = __popc(run.support);
            const unsigned long long pc = RP_PRIMITIVE[occupied][1];
            const int leader_owner = traffic_owner_from_lut_device(
                run.support, W, Kwin, reverse, owner_lut);
            if (leader_owner < 0 || leader_owner >= ngpu) {
                set_error(error, 212);
                continue;
            }

            int prev_owner = leader_owner;
            std::uint32_t cur_support = shift_next_support_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            int hops = 1;
            while (cur_support != run.support && hops < cycle_len) {
                const int owner = traffic_owner_from_lut_device(
                    cur_support, W, Kwin, reverse, owner_lut);
                if (owner < 0 || owner >= ngpu) {
                    set_error(error, 213);
                    break;
                }
                if (owner != prev_owner) {
                    atomicAdd(&cross_matrix[prev_owner * TRAFFIC_MATRIX_GPU + owner], pc);
                }
                if (owner != leader_owner) {
                    atomicAdd(
                        &direct_matrix[leader_owner * TRAFFIC_MATRIX_GPU + owner],
                        2ULL * pc);
                }
                prev_owner = owner;
                cur_support = shift_next_support_device(
                    cur_support, blocked, W, q, Kwin, S, reverse);
                ++hops;
            }
            if (hops != cycle_len || cur_support != run.support) {
                set_error(error, 214);
                continue;
            }
            if (prev_owner != leader_owner) {
                atomicAdd(
                    &cross_matrix[prev_owner * TRAFFIC_MATRIX_GPU + leader_owner], pc);
            }

            ++local_cycles;
            local_rotated += pc * static_cast<unsigned long long>(cycle_len);
        }
    }

    if (local_cycles) atomicAdd(cycles, local_cycles);
    if (local_rotated) atomicAdd(rotated_values, local_rotated);
    __syncthreads();

    for (int i = threadIdx.x; i < TRAFFIC_MATRIX_SIZE; i += blockDim.x) {
        const unsigned long long x = cross_matrix[i];
        const unsigned long long d = direct_matrix[i];
        if (x) atomicAdd(&global_cross_matrix[i], x);
        if (d) atomicAdd(&global_direct_matrix[i], d);
    }
}

void print_p2p_matrix_summary(
    const std::array<unsigned long long, TRAFFIC_MATRIX_SIZE>& cross,
    const std::array<unsigned long long, TRAFFIC_MATRIX_SIZE>& direct,
    int ngpu,
    const char* direction
) {
    unsigned long long cross_total = 0;
    unsigned long long direct_total = 0;
    unsigned long long max_pair = 0;
    int max_src = -1, max_dst = -1;

    for (int src = 0; src < ngpu; ++src) {
        for (int dst = 0; dst < ngpu; ++dst) {
            const auto x = cross[src * TRAFFIC_MATRIX_GPU + dst];
            const auto d = direct[src * TRAFFIC_MATRIX_GPU + dst];
            cross_total += x;
            direct_total += d;
            if (x > max_pair) {
                max_pair = x;
                max_src = src;
                max_dst = dst;
            }
        }
    }

    const double direct_over_logical = cross_total
        ? double(direct_total) / double(cross_total) : 0.0;
    std::cout << "p2p-traffic-matrix-summary"
              << " direction=" << direction
              << " logical_peer_GiB="
              << double(cross_total) * 4.0 / double(1ULL << 30)
              << " direct_remote_GiB="
              << double(direct_total) * 4.0 / double(1ULL << 30)
              << " direct_over_logical=" << direct_over_logical
              << " hottest_pair=" << max_src << "->" << max_dst
              << " hottest_pair_GiB="
              << double(max_pair) * 4.0 / double(1ULL << 30)
              << '\n';

    for (int g = 0; g < ngpu; ++g) {
        unsigned long long logical_out = 0, logical_in = 0;
        unsigned long long direct_issue = 0, direct_target = 0;
        for (int h = 0; h < ngpu; ++h) {
            logical_out += cross[g * TRAFFIC_MATRIX_GPU + h];
            logical_in += cross[h * TRAFFIC_MATRIX_GPU + g];
            direct_issue += direct[g * TRAFFIC_MATRIX_GPU + h];
            direct_target += direct[h * TRAFFIC_MATRIX_GPU + g];
        }
        std::cout << "p2p-traffic-gpu"
                  << " direction=" << direction
                  << " gpu=" << g
                  << " logical_out_GiB="
                  << double(logical_out) * 4.0 / double(1ULL << 30)
                  << " logical_in_GiB="
                  << double(logical_in) * 4.0 / double(1ULL << 30)
                  << " direct_issue_GiB="
                  << double(direct_issue) * 4.0 / double(1ULL << 30)
                  << " direct_target_GiB="
                  << double(direct_target) * 4.0 / double(1ULL << 30)
                  << '\n';
    }
}

void run_p2p_traffic_matrix_probe(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const std::vector<std::uint8_t> owner_lut =
        build_traffic_owner_lut(tables, Kwin, ngpu);

    ck(cudaSetDevice(0), "traffic matrix set device");
    install_tables(tables);

    std::uint8_t* d_owner_lut = nullptr;
    unsigned long long* d_cross = nullptr;
    unsigned long long* d_direct = nullptr;
    unsigned long long* d_cycles = nullptr;
    unsigned long long* d_rotated = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_owner_lut, owner_lut.size()), "traffic matrix alloc LUT");
    ck(cudaMemcpy(d_owner_lut, owner_lut.data(), owner_lut.size(), cudaMemcpyHostToDevice),
       "traffic matrix copy LUT");
    ck(cudaMalloc(&d_cross, TRAFFIC_MATRIX_SIZE * sizeof(unsigned long long)),
       "traffic matrix alloc cross");
    ck(cudaMalloc(&d_direct, TRAFFIC_MATRIX_SIZE * sizeof(unsigned long long)),
       "traffic matrix alloc direct");
    ck(cudaMalloc(&d_cycles, sizeof(unsigned long long)), "traffic matrix alloc cycles");
    ck(cudaMalloc(&d_rotated, sizeof(unsigned long long)), "traffic matrix alloc rotated");
    ck(cudaMalloc(&d_error, sizeof(int)), "traffic matrix alloc error");
    ck(cudaMemset(d_cross, 0, TRAFFIC_MATRIX_SIZE * sizeof(unsigned long long)),
       "traffic matrix zero cross");
    ck(cudaMemset(d_direct, 0, TRAFFIC_MATRIX_SIZE * sizeof(unsigned long long)),
       "traffic matrix zero direct");
    ck(cudaMemset(d_cycles, 0, sizeof(unsigned long long)), "traffic matrix zero cycles");
    ck(cudaMemset(d_rotated, 0, sizeof(unsigned long long)), "traffic matrix zero rotated");
    ck(cudaMemset(d_error, 0, sizeof(int)), "traffic matrix zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 needed_blocks =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));

    const auto t0 = std::chrono::steady_clock::now();
    p2p_cycle_traffic_matrix_kernel<<<blocks, THREADS>>>(
        base_supports, W, Kwin, S, reverse, ngpu, d_owner_lut,
        d_cross, d_direct, d_cycles, d_rotated, d_error);
    ck(cudaGetLastError(), "traffic matrix launch");
    ck(cudaDeviceSynchronize(), "traffic matrix sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long cycles = 0, rotated = 0;
    std::array<unsigned long long, TRAFFIC_MATRIX_SIZE> cross{};
    std::array<unsigned long long, TRAFFIC_MATRIX_SIZE> direct{};
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "traffic matrix copy error");
    ck(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost),
       "traffic matrix copy cycles");
    ck(cudaMemcpy(&rotated, d_rotated, sizeof(rotated), cudaMemcpyDeviceToHost),
       "traffic matrix copy rotated");
    ck(cudaMemcpy(cross.data(), d_cross, sizeof(cross), cudaMemcpyDeviceToHost),
       "traffic matrix copy cross");
    ck(cudaMemcpy(direct.data(), d_direct, sizeof(direct), cudaMemcpyDeviceToHost),
       "traffic matrix copy direct");
    if (error) fail("p2p traffic matrix device error=" + std::to_string(error));

    const char* direction = reverse ? "reverse" : "forward";
    std::cout << "gridfp-reduced-production-p2p-traffic-matrix"
              << " W=" << W << " Kwin=" << Kwin << " shift=" << S
              << " direction=" << direction
              << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " cycles=" << cycles
              << " rotated_values=" << rotated
              << " owner_lut_bytes=" << owner_lut.size()
              << " blocks=" << blocks
              << " wall_ms=" << ms
              << " block_shared_matrix_bytes="
              << 2 * TRAFFIC_MATRIX_SIZE * sizeof(unsigned long long)
              << " state_allocation_bytes=0 exact_matrix=OK\n";
    print_p2p_matrix_summary(cross, direct, ngpu, direction);

    cudaFree(d_error);
    cudaFree(d_rotated);
    cudaFree(d_cycles);
    cudaFree(d_direct);
    cudaFree(d_cross);
    cudaFree(d_owner_lut);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const int S = argc > 3 ? std::atoi(argv[3]) : Kwin;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 4096u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 8;
    if (W < 6 || W > RP_MAX_W || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > TRAFFIC_MATRIX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "traffic matrix device count");
    if (visible < 1) return 3;

    run_p2p_traffic_matrix_probe(W, Kwin, S, false, ngpu, blocks);
    run_p2p_traffic_matrix_probe(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_p2p_traffic_matrix=1\n";
    return 0;
}
