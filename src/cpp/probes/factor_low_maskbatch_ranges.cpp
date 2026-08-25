#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../cuda/b300/maskshard_low_maskbatch_plan.hpp"
#include "../../cuda/b300/maskshard_low_maskbatch_range.hpp"

int main(int argc, char** argv) {
    const int masks = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int devices = argc > 2 ? std::atoi(argv[2]) : 8;
    const int max_replicas = argc > 3 ? std::atoi(argv[3]) : 1024;
    const std::uint64_t target = argc > 4
        ? std::strtoull(argv[4], nullptr, 10) : 256ULL;
    if (masks < 1 || masks > 65535 || devices < 1 || devices > 8
        || max_replicas < 1 || max_replicas > 65535 || target == 0)
        return 1;

    std::vector<std::uint8_t> owner(std::size_t(masks));
    std::vector<std::uint64_t> tasks(std::size_t(masks));
    for (int m = 0; m < masks; ++m) {
        owner[std::size_t(m)] = std::uint8_t((m * 5 + 3) % devices);
        tasks[std::size_t(m)] = (m % 19 == 0) ? 0ULL
            : ((std::uint64_t(m + 11) * 104729ULL) % 400000ULL) + 1ULL;
    }

    std::uint64_t expanded_total = 0;
    std::uint64_t range_total = 0;
    bool saw_wide = false;
    for (int d = 0; d < devices; ++d) {
        const auto expanded = maskshard_build_low_batch_plan(
            owner, d, tasks, target, max_replicas);
        const auto ranges = maskshard_build_low_batch_range_plan(
            owner, d, tasks, target, max_replicas);

        std::vector<std::uint32_t> ends;
        ends.reserve(ranges.size());
        std::uint64_t end = 0;
        for (const auto& r : ranges) {
            end += r.replicas;
            if (end > 0xffffffffULL) return 2;
            ends.push_back(std::uint32_t(end));
            saw_wide = saw_wide || r.replicas > 255u;
        }
        if (end != expanded.size()) {
            std::cerr << "range CTA total mismatch dev=" << d
                      << " range=" << end
                      << " expanded=" << expanded.size() << '\n';
            return 3;
        }

        const std::uint32_t ctas = ends.empty() ? 0u : ends.back();
        for (std::uint32_t cta = 0; cta < ctas; ++cta) {
            const auto it = std::upper_bound(ends.begin(), ends.end(), cta);
            if (it == ends.end()) return 4;
            const std::size_t g = std::size_t(it - ends.begin());
            const std::uint32_t begin = g ? ends[g - 1] : 0u;
            const auto& r = ranges[g];
            const auto& e = expanded[cta];
            if (e.mask != r.mask
                || e.replica != std::uint16_t(cta - begin)
                || e.replicas != r.replicas) {
                std::cerr << "range decode mismatch dev=" << d
                          << " cta=" << cta << " group=" << g << '\n';
                return 5;
            }
        }

        expanded_total += expanded.size();
        range_total += ranges.size();
    }

    if (max_replicas > 255 && !saw_wide) {
        std::cerr << "wide compact range was not exercised\n";
        return 6;
    }
    if (range_total > expanded_total) return 7;

    std::cout << "low-maskbatch-ranges masks=" << masks
              << " devices=" << devices
              << " target=" << target
              << " max_replicas=" << max_replicas
              << " expanded_descriptors=" << expanded_total
              << " range_descriptors=" << range_total
              << " compression="
              << (range_total ? double(expanded_total) / double(range_total) : 0.0)
              << " wide_range_exercised=" << (saw_wide ? 1 : 0)
              << " exact_decode=1\n";
    return 0;
}
