#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../cuda/b300/maskshard_high_static_lpt_schedule.hpp"

struct Job { std::uint64_t work = 0; };

static int popcount32(std::uint32_t x) {
    int n = 0;
    while (x) { x &= x - 1; ++n; }
    return n;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    if (W < 1 || low < 1 || low >= 31 || ngpu < 1) return 1;

    const std::uint64_t masks = std::uint64_t(1) << low;
    std::vector<Job> jobs;
    jobs.reserve(std::size_t(masks));
    for (std::uint32_t mask = 0; mask < masks; ++mask) {
        const int pc = popcount32(mask);
        const std::uint64_t c = std::uint64_t(low + 1 - pc);
        jobs.push_back({1 + c * c * c * c});
    }
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        return a.work > b.work;
    });

    const auto plan = maskshard_build_high_static_lpt_schedule(jobs, ngpu);
    const std::uint64_t dynamic_fetches =
        (masks + std::uint64_t(ngpu)) * std::uint64_t(W);
    const std::uint64_t successful_tickets = masks * std::uint64_t(W);
    const std::uint64_t failed_tail_tickets =
        std::uint64_t(ngpu) * std::uint64_t(W);
    const std::uint64_t max_job = jobs.empty() ? 0 : jobs.front().work;
    if (plan.max_work - plan.min_work > max_job) {
        std::cerr << "LPT load spread exceeds max-job bound\n";
        return 2;
    }
    if (W == 28 && low == 14 && ngpu == 8) {
        if (masks != 16384ULL
            || successful_tickets != 458752ULL
            || failed_tail_tickets != 224ULL
            || dynamic_fetches != 458976ULL) {
            std::cerr << "n=27 static-LPT scheduler regression mismatch\n";
            return 3;
        }
    }

    std::cout << "high-static-lpt W=" << W
              << " low=" << low
              << " ngpu=" << ngpu << '\n'
              << "jobs=" << masks
              << " schedule_build_assignments=" << masks << '\n'
              << "dynamic_successful_tickets=" << successful_tickets
              << " dynamic_failed_tail_tickets=" << failed_tail_tickets
              << " dynamic_atomic_fetch_adds=" << dynamic_fetches << '\n'
              << "static_atomic_fetch_adds=0\n"
              << "static_min_model_work=" << plan.min_work
              << " static_max_model_work=" << plan.max_work
              << " static_spread=" << plan.max_work - plan.min_work << '\n';
    return 0;
}
