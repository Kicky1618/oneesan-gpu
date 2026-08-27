#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int L = argc > 2 ? std::atoi(argv[2]) : 14;
    const int H = argc > 3 ? std::atoi(argv[3]) : W - L - 1;
    if (W < 3 || L < 1 || H < 1 || L + H + 1 != W) {
        std::cerr << "usage: factor_high_perthread_stream [W L H]\n";
        return 2;
    }

    // Current v0.70 chain has lazy BLOCKED gather: one MAIN gather, H orbit,
    // H closure, then MAIN+BLOCKED scatter. Sync interception still observes
    // gather + 2H inner phases + scatter = 2H+2 phase boundaries.
    const std::uint64_t kernel_launches = 1u + 2u * std::uint64_t(H) + 2u;
    const std::uint64_t sync_phases = 2u * std::uint64_t(H) + 2u;
    const std::uint64_t high_jobs_per_residue =
        (std::uint64_t(1) << L) * std::uint64_t(W);

    std::cout << "W=" << W
              << " L=" << L
              << " H=" << H
              << " kernel_launches_per_high_job=" << kernel_launches
              << " sync_phases_per_high_job=" << sync_phases
              << " high_jobs_per_residue=" << high_jobs_per_residue
              << " kernel_launches_per_residue="
              << kernel_launches * high_jobs_per_residue
              << '\n';

    if (W == 28 && L == 14 && H == 13) {
        if (kernel_launches != 29 || sync_phases != 28
            || high_jobs_per_residue != 458752ULL) {
            std::cerr << "n=27 HIGH per-thread stream model regression\n";
            return 3;
        }
    }
    return 0;
}
