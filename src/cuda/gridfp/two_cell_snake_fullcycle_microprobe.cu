#pragma push_macro("main")
#undef main
#define main two_cell_stationary_snake_cycle_probe_main_unused_for_cuda
#include "../../cpp/probes/two_cell_stationary_snake_cycle_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_snake_schedule.hpp"
#include "two_cell_snake_stage_api.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

namespace {

void ck_snake(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": "
                  << cudaGetErrorString(e) << '\n';
        std::exit(620);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const int max_cluster = argc > 2 ? std::atoi(argv[2]) : 8;
    const std::uint64_t shared_kib = argc > 3
        ? std::strtoull(argv[3], nullptr, 10) : 228ULL;
    if (W < 6 || W > 10 || (W & 1) ||
        (max_cluster != 1 && max_cluster != 2 &&
         max_cluster != 4 && max_cluster != 8) || !shared_kib)
        return 2;

    constexpr std::uint32_t mod = static_cast<std::uint32_t>(kMod);
    const std::uint64_t shared_bytes = shared_kib * 1024ULL;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);

    const auto start_basis = q_basis(W, 0, words);
    std::map<Key, Value> exact;
    std::vector<std::uint32_t> input(static_cast<std::size_t>(st.total[W]), 0);
    for (Rank q = 0; q < start_basis.size(); ++q) {
        const Value x = 1 + ((q * 2654435761ULL + 97ULL) % (kMod - 1));
        const Key& k = start_basis[static_cast<std::size_t>(q)];
        exact[k] = x;
        const Rank r = snake_rank(k, W, 0, rt, st);
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(x);
    }

    // Independent exact CPU reference: ordinary one-step transfers and turns.
    for (int i = 0; i <= W - 4; ++i)
        exact = exact_forward(exact, W, i);
    exact = exact_turn(exact, W, true);
    for (int pair = W - 2; pair >= 2; --pair)
        exact = exact_reverse(exact, W, pair);
    exact = exact_turn(exact, W, false);

    int visible = 0;
    ck_snake(cudaGetDeviceCount(&visible), "snake device count");
    if (visible < 1) return 3;
    ck_snake(cudaSetDevice(0), "snake set device");
    cudaDeviceProp prop{};
    ck_snake(cudaGetDeviceProperties(&prop, 0), "snake props");
    if (prop.major < 9) {
        std::cerr << "full snake cluster microprobe requires compute capability >= 9.0\n";
        return 4;
    }

    std::uint32_t* d_values = nullptr;
    ck_snake(cudaMalloc(&d_values, input.size() * sizeof(std::uint32_t)),
             "snake alloc values");
    ck_snake(cudaMemcpy(d_values, input.data(),
                        input.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice),
             "snake copy input");

    const auto schedule = oneesan::twocell::make_snake_schedule(W);
    if (!schedule.valid) return 5;

    int completed_pairs = 0;
    for (int q = 0; q < schedule.size; ++q) {
        const auto item = schedule.pair[q];
        int rc = 0;
        using Kind = oneesan::twocell::SnakePairKind;
        switch (item.kind) {
            case Kind::ForwardFusion2:
                rc = oneesan_two_cell_forward2_stage(
                    d_values, W, item.start, max_cluster, shared_bytes, mod);
                break;
            case Kind::RightBoundary:
                rc = oneesan_two_cell_right_boundary_stage(
                    d_values, W, max_cluster, shared_bytes, mod);
                break;
            case Kind::ReverseFusion2:
                rc = oneesan_two_cell_reverse2_stage(
                    d_values, W, item.start, max_cluster, shared_bytes, mod);
                break;
            case Kind::LeftBoundary:
                rc = oneesan_two_cell_left_boundary_stage(
                    d_values, W, max_cluster, shared_bytes, mod);
                break;
        }
        if (rc != 0) {
            std::cerr << "FAIL snake stage"
                      << " pair=" << q
                      << " kind=" << oneesan::twocell::snake_pair_kind_name(item.kind)
                      << " start=" << item.start
                      << " rc=" << rc << '\n';
            cudaFree(d_values);
            return 6;
        }
        ++completed_pairs;
        std::cout << "snake-stage OK"
                  << " pair=" << q
                  << " kind=" << oneesan::twocell::snake_pair_kind_name(item.kind)
                  << " start=" << item.start << '\n';
    }
    ck_snake(cudaDeviceSynchronize(), "snake final sync");

    std::vector<std::uint32_t> output(input.size());
    ck_snake(cudaMemcpy(output.data(), d_values,
                        output.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost),
             "snake copy output");
    cudaFree(d_values);

    const auto end_basis = q_basis(W, 0, words);
    Rank checked = 0;
    for (const Key& k : end_basis) {
        const Rank r = snake_rank(k, W, 0, rt, st);
        const std::uint32_t expected = static_cast<std::uint32_t>(
            exact.count(k) ? exact[k] : 0);
        const std::uint32_t got = output[static_cast<std::size_t>(r)];
        if (got != expected) {
            std::cerr << "FAIL full snake arithmetic"
                      << " W=" << W
                      << " rank=" << r
                      << " expected=" << expected
                      << " got=" << got << '\n';
            return 7;
        }
        ++checked;
    }

    std::cout << "ALL_OK two_cell_full_snake_cuda=1"
              << " W=" << W
              << " states=" << st.total[W]
              << " pairs=" << completed_pairs
              << " operators=" << 2 * completed_pairs
              << " checked=" << checked
              << " global_vectors=1"
              << " layout_conversions=0"
              << " max_cluster=" << max_cluster
              << '\n';
    return 0;
}
