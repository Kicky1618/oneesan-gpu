#pragma push_macro("main")
#undef main
#define main two_cell_component_device_probe_main_unused
#include "../../cpp/probes/two_cell_component_device_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_component_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using oneesan::twocell::ComponentBlocks;
using oneesan::twocell::PackedKey;
using oneesan::twocell::PackedWord;
using oneesan::twocell::Rank;
using oneesan::twocell::RankTables;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = 32 * WARPS_PER_BLOCK;
constexpr int MAX_STATES = 18;

__constant__ RankTables TC_RANK_TABLES;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(120);
    }
}

__device__ __forceinline__ void set_error(int* error, int code) {
    atomicCAS(error, 0, code);
}

__device__ __forceinline__ Rank shared_rank(
    PackedKey key,
    int W,
    int window,
    int outer_ones,
    Rank base
) {
    return key.type == 0
        ? oneesan::twocell::rank_A_with_block(
              key.support, key.left, W, window, outer_ones, base, TC_RANK_TABLES)
        : oneesan::twocell::rank_C_with_block(
              key.support, key.left, W, window, outer_ones, base, TC_RANK_TABLES);
}

__global__ void two_cell_component_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    Rank components,
    Rank state_count,
    int W,
    int i,
    std::uint32_t mod,
    unsigned long long* processed,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ PackedKey sh_dst[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];
    __shared__ int sh_in_ones[WARPS_PER_BLOCK];
    __shared__ int sh_out_ones[WARPS_PER_BLOCK];
    __shared__ Rank sh_in_base[WARPS_PER_BLOCK];
    __shared__ Rank sh_out_base[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;

    for (Rank component_rank = first; component_rank < components; component_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;

            const PackedKey label_key = oneesan::twocell::component_label_unrank(
                W, component_rank, TC_RANK_TABLES);
            const PackedWord label{
                label_key.support,
                label_key.left,
                static_cast<std::uint8_t>(W - 2)
            };
            const auto blocks = oneesan::twocell::component_blocks(
                label.support, W, i, TC_RANK_TABLES);
            sh_in_ones[warp] = blocks.input_ones;
            sh_out_ones[warp] = blocks.output_ones;
            sh_in_base[warp] = blocks.input_base;
            sh_out_base[warp] = blocks.output_base;

            const auto src = oneesan::twocell::direct_component_sources(label, W, i);
            if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                set_error(error, 121);
            } else {
                sh_ns[warp] = src.size;
                for (int s = 0; s < src.size; ++s) sh_src[warp][s] = src.value[s];

                for (int s = 0; s < src.size; ++s) {
                    const auto edges = oneesan::twocell::K_step(src.value[s], W, i);
                    if (edges.overflow) {
                        set_error(error, 122);
                        break;
                    }
                    for (int e = 0; e < edges.size; ++e) {
                        bool seen = false;
                        for (int d = 0; d < sh_nd[warp]; ++d)
                            seen |= oneesan::twocell::equal(sh_dst[warp][d], edges.value[e]);
                        if (seen) continue;
                        if (sh_nd[warp] >= MAX_STATES) {
                            set_error(error, 123);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = edges.value[e];
                    }
                }
                if (sh_nd[warp] != sh_ns[warp]) set_error(error, 124);
                atomicAdd(processed, 1ULL);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        if (lane < ns) {
            const Rank r = shared_rank(
                sh_src[warp][lane], W, i - 1,
                sh_in_ones[warp], sh_in_base[warp]);
            if (r >= state_count) {
                sh_value[warp][lane] = 0;
                set_error(error, 125);
            } else {
                sh_value[warp][lane] = input[r];
            }
        }
        __syncwarp();

        if (lane < nd) {
            const PackedKey mine = sh_dst[warp][lane];
            unsigned long long acc = 0;
            for (int s = 0; s < ns; ++s) {
                const auto edges = oneesan::twocell::K_step(sh_src[warp][s], W, i);
                for (int e = 0; e < edges.size; ++e)
                    if (oneesan::twocell::equal(edges.value[e], mine))
                        acc += sh_value[warp][s];
            }
            const Rank r = shared_rank(
                mine, W, i,
                sh_out_ones[warp], sh_out_base[warp]);
            if (r >= state_count) {
                set_error(error, 126);
            } else {
                const unsigned long long empty = ~0ULL;
                const unsigned long long previous = atomicCAS(
                    owner + r, empty, static_cast<unsigned long long>(component_rank));
                if (previous != empty && previous != static_cast<unsigned long long>(component_rank))
                    set_error(error, 127);
                output[r] = static_cast<std::uint32_t>(acc % mod);
            }
        }
        __syncwarp();
    }
}

std::uint32_t add_mod(std::uint32_t a, std::uint32_t b, std::uint32_t mod) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(a) + b) % mod);
}

bool has_arg(int argc, char** argv, const char* needle) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == needle) return true;
    return false;
}

void install_tables(const RankTables& tables) {
    ck(cudaMemcpyToSymbol(TC_RANK_TABLES, &tables, sizeof(tables)), "copy rank tables");
}

void run_position(
    int W,
    int i,
    const RankTables& tables,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const Rank states = tables.suffix[W - 5][0];
    const Rank components = oneesan::twocell::component_label_count(W, tables);
    if (i < 1 || i > W - 5) {
        std::cerr << "invalid interior position i=" << i << '\n';
        std::exit(121);
    }

    const auto qsrc = q_basis(W, i, [&]() {
        std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
        for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
        return words;
    }());
    if (qsrc.size() != states) {
        std::cerr << "state dimension mismatch\n";
        std::exit(122);
    }

    // Rebuild the small-width word table once for the exact CPU oracle.
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(i) * 17ULL) % (mod - 1ULL)));

    for (const Key& s : q_basis(W, i, words)) {
        const auto ps = device_key(s);
        const Rank sr = oneesan::twocell::rank_state(ps, W, i - 1, tables);
        const std::uint32_t value = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_basis(s, W, i)) {
            if (c != 1) {
                std::cerr << "nonunit two-cell coefficient\n";
                std::exit(123);
            }
            const Rank dr = oneesan::twocell::rank_state(device_key(d), W, i, tables);
            reference[static_cast<std::size_t>(dr)] =
                add_mod(reference[static_cast<std::size_t>(dr)], value, mod);
        }
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "alloc output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "alloc processed");
    ck(cudaMalloc(&d_error, sizeof(int)), "alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "zero processed");
    ck(cudaMemset(d_error, 0, sizeof(int)), "zero error");

    const Rank one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
        1, std::min<Rank>(requested_blocks, one_pass_blocks)));

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "event begin");
    ck(cudaEventCreate(&end), "event end");
    ck(cudaEventRecord(begin), "record begin");
    two_cell_component_kernel<<<blocks, THREADS>>>(
        d_input, d_output, d_owner, components, states, W, i, mod,
        d_processed, d_error);
    ck(cudaGetLastError(), "launch component kernel");
    ck(cudaEventRecord(end), "record end");
    ck(cudaEventSynchronize(end), "sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "copy processed");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "copy error");

    if (error || processed != components) {
        std::cerr << "FAIL two-cell component CUDA error=" << error
                  << " processed=" << processed << " expected=" << components << '\n';
        std::exit(124);
    }
    for (Rank r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL two-cell component arithmetic W=" << W
                      << " i=" << i << " rank=" << r << '\n';
            std::exit(125);
        }
    }

    const double logical_gib = double(states) * 8.0 / double(1ULL << 30);
    const double components_per_warp =
        double(components) / double(Rank(blocks) * WARPS_PER_BLOCK);
    std::cout << "two-cell-component-sliding-microprobe"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " blocks=" << blocks
              << " components_per_launched_warp=" << components_per_warp
              << " kernel_ms=" << ms
              << " logical_count_GiB=" << logical_gib
              << " source_load_once=1 destination_store_once=1"
              << " component_graph_search=0 permutation_table_bytes=0"
              << " arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || blocks == 0 || mod < 3) return 2;

    const RankTables tables = oneesan::twocell::make_rank_tables();
    const Rank states = tables.suffix[W - 5][0];
    const Rank components = oneesan::twocell::component_label_count(W, tables);

    if (plan_only) {
        std::cout << "two-cell-component-sliding-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " max_component_sources=17"
                  << " rank_table_KiB=" << double(sizeof(tables)) / 1024.0
                  << " logical_count_GiB_per_step="
                  << double(states) * 8.0 / double(1ULL << 30)
                  << " component_label_table_bytes=0 permutation_table_bytes=0"
                  << "\n";
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "set device");
    install_tables(tables);

    for (int i = 1; i <= W - 5; ++i) run_position(W, i, tables, mod, blocks);
    std::cout << "ALL_OK two_cell_component_sliding_cuda=1 W=" << W << '\n';
    return 0;
}
