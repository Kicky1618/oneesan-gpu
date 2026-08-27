#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#define LOW_LUT_K 4

struct MaskShardHighStaticLptSchedule {
    std::vector<std::vector<std::size_t>> jobs_by_gpu;
    std::vector<std::uint64_t> work_by_gpu;
    std::uint64_t total_work = 0;
    std::uint64_t min_work = 0;
    std::uint64_t max_work = 0;
};

static int maskshard_high_cap_lpt_popcount(std::uint32_t mask) {
    int n = 0;
    while (mask) {
        mask &= mask - 1;
        ++n;
    }
    return n;
}

#include "../../cuda/b300/maskshard_high_lpt_locality_policy.hpp"

struct Job {
    std::uint32_t low_mask = 0;
};

static std::vector<std::uint64_t> sorted_loads(
    const MaskShardHighStaticLptSchedule& p
) {
    auto x = p.work_by_gpu;
    std::sort(x.begin(), x.end());
    return x;
}

int main() {
    const std::vector<Job> jobs = {{0u}, {1u}, {2u}, {0u}};
    const std::vector<std::uint64_t> weight = {10, 10, 10, 10};
    const std::array<std::array<std::uint64_t, 2>, 4> local = {{
        {{100, 1}},
        {{1, 100}},
        {{1, 100}},
        {{100, 1}},
    }};

    const auto p = maskshard_build_high_lpt_from_weights_affinity_locality(
        weight, jobs, 2,
        [&](std::size_t q, int d) { return local[q][std::size_t(d)]; });
    if (sorted_loads(p) != std::vector<std::uint64_t>({20, 20})) {
        std::cerr << "locality policy changed LPT load multiset\n";
        return 1;
    }

    std::uint64_t local_sum = 0;
    bool seen[2][LOW_LUT_K + 1]{};
    std::uint64_t classes = 0;
    for (int d = 0; d < 2; ++d) {
        for (std::size_t q : p.jobs_by_gpu[std::size_t(d)]) {
            local_sum += local[q][std::size_t(d)];
            const int pc = maskshard_high_cap_lpt_popcount(jobs[q].low_mask);
            if (!seen[d][pc]) {
                seen[d][pc] = true;
                ++classes;
            }
        }
    }
    if (classes != 2 || local_sum != 400) {
        std::cerr << "locality policy failed class/local preference classes="
                  << classes << " local=" << local_sum << '\n';
        return 2;
    }

    std::cout << "factor-high-cap-lpt-locality-policy OK classes=" << classes
              << " local=" << local_sum << " loads=20,20\n";
    return 0;
}
