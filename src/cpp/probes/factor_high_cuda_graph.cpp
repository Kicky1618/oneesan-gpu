#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int L = argc > 2 ? std::atoi(argv[2]) : 14;
    const int H = argc > 3 ? std::atoi(argv[3]) : W - L - 1;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 3 || (W & 1) || L < 1 || L >= 31 || H < 1
        || L + H + 1 != W || ngpu < 1 || ngpu > 8) {
        std::cerr << "usage: factor_high_cuda_graph [W L H ngpu]\n";
        return 2;
    }

    const std::uint64_t classes = std::uint64_t(L + 1);
    const std::uint64_t caps = std::uint64_t(W / 2);
    const std::uint64_t slots_per_gpu = classes * caps;
    const std::uint64_t jobs_per_residue =
        (std::uint64_t(1) << L) * std::uint64_t(W);
    // Structural upper bound for the current chain: lazy BLOCKED gather,
    // H possible orbit launches, H possible closure launches, MAIN+BLOCKED
    // scatter. Zero-task orbit/closure phases are skipped at runtime, so these
    // native counts and the derived reduction are upper-bound/maximum values,
    // not a claim that every HIGH job executes all 29 kernels.
    const std::uint64_t native_max_per_job = 1u + 2u * std::uint64_t(H) + 2u;
    const std::uint64_t native_max_launches =
        native_max_per_job * jobs_per_residue;
    const std::uint64_t graph_launches = jobs_per_residue;
    const std::uint64_t max_captures = slots_per_gpu * std::uint64_t(ngpu);
    const double max_reduction =
        100.0 * (1.0 - double(graph_launches) / double(native_max_launches));

    std::cout << "W=" << W
              << " L=" << L
              << " H=" << H
              << " ngpu=" << ngpu
              << " classes=" << classes
              << " caps=" << caps
              << " graph_slots_per_gpu=" << slots_per_gpu
              << " max_graph_captures=" << max_captures
              << " jobs_per_residue=" << jobs_per_residue
              << " native_max_launches_per_job=" << native_max_per_job
              << " native_max_launches_per_residue=" << native_max_launches
              << " graph_launches_per_residue=" << graph_launches
              << " max_steady_launch_call_reduction_pct=" << max_reduction
              << '\n';

    if (W == 28 && L == 14 && H == 13 && ngpu == 8) {
        if (slots_per_gpu != 210ULL || max_captures != 1680ULL
            || jobs_per_residue != 458752ULL || native_max_per_job != 29ULL
            || native_max_launches != 13303808ULL
            || graph_launches != 458752ULL) {
            std::cerr << "n=27 HIGH CUDA Graph model regression\n";
            return 3;
        }
    }
    return 0;
}
