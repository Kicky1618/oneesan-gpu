#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_host_persistent_list_probe_main_unused
#include "gridfp_reduced_production_host_persistent_list_probe.cpp"
#pragma pop_macro("main")

#include <unordered_map>

namespace {

bool verify_cycle_owner_coverage(
    int W,
    int K,
    int ngpu,
    bool reverse
) {
    const int S = K;
    if (W != 2 * K + 2) return false;
    const Tables tables(W);
    const int q = (reverse ? 1 : W - 1) + (reverse ? S : -S);

    const auto key = [](Mask support, bool blocked) {
        return (Rank(blocked) << 32) | support;
    };
    std::unordered_map<Rank, int> coverage;
    coverage.reserve(
        std::size_t(5) * (std::size_t(1) << (W - 3)) * 2);

    Rank total_slabs = 0;
    Rank cross_entries = 0;
    Rank original_segments = 0;
    Rank local_entries = 0;
    Rank peer_words = 0;
    int max_segments = 0;

    for (Rank compact = 0; compact < (Rank(1) << (W - 2)); ++compact) {
        for (const Seed seed : run_seeds(compact, W, q, reverse)) {
            ++total_slabs;
            const bool blocked = seed.blocked;
            const int cycle_len = cycle_leader_length(
                seed.support, blocked, W, q, K, S, reverse);
            if (cycle_len < 0) return false;
            if (cycle_len <= 1) continue;

            std::vector<Mask> route(static_cast<std::size_t>(cycle_len));
            std::vector<int> route_owner(static_cast<std::size_t>(cycle_len));
            Mask cur = seed.support;
            for (int h = 0; h < cycle_len; ++h) {
                route[static_cast<std::size_t>(h)] = cur;
                route_owner[static_cast<std::size_t>(h)] =
                    support_owner(cur, W, K, reverse, ngpu, tables);
                cur = shift_next_support(cur, blocked, W, q, K, S, reverse);
            }
            if (cur != seed.support) return false;

            bool all_local = true;
            for (int h = 1; h < cycle_len; ++h)
                all_local = all_local &&
                    route_owner[static_cast<std::size_t>(h)] == route_owner[0];
            if (all_local) {
                ++local_entries;
                for (Mask s : route) ++coverage[key(s, blocked)];
                continue;
            }

            int segments_per_owner[8]{};
            int h = 0;
            while (h < cycle_len) {
                const int owner = route_owner[static_cast<std::size_t>(h)];
                int len = 1;
                while (h + len < cycle_len &&
                       route_owner[static_cast<std::size_t>(h + len)] == owner)
                    ++len;
                ++segments_per_owner[owner];
                h += len;
            }
            if (route_owner.front() == route_owner.back())
                --segments_per_owner[route_owner.front()];

            const Rank primitive_count =
                tables.primitive[__builtin_popcount(seed.support)][1];
            if (!primitive_count) return false;
            for (int owner = 0; owner < ngpu; ++owner) {
                const int segments = segments_per_owner[owner];
                if (!segments) continue;
                ++cross_entries;
                original_segments += segments;
                peer_words += Rank(segments) * primitive_count;
                max_segments = std::max(max_segments, segments);
                for (int i = 0; i < cycle_len; ++i) {
                    if (route_owner[static_cast<std::size_t>(i)] == owner)
                        ++coverage[key(route[static_cast<std::size_t>(i)], blocked)];
                }
            }
        }
    }

    Rank nonfixed = 0;
    Rank fixed = 0;
    Rank bad = 0;
    for (Rank compact = 0; compact < (Rank(1) << (W - 2)); ++compact) {
        for (const Seed seed : run_seeds(compact, W, q, reverse)) {
            const Mask next = shift_next_support(
                seed.support, seed.blocked, W, q, K, S, reverse);
            const int seen = coverage[key(seed.support, seed.blocked)];
            if (next == seed.support) {
                ++fixed;
                if (seen != 0) ++bad;
            } else {
                ++nonfixed;
                if (seen != 1) ++bad;
            }
        }
    }

    if (total_slabs != Rank(5) * (Rank(1) << (W - 3))) return false;
    std::cout << "cycle-owner-coverage"
              << " W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " support_slabs=" << total_slabs
              << " nonfixed_slabs=" << nonfixed
              << " fixed_slabs=" << fixed
              << " compressed_cross_entries=" << cross_entries
              << " original_cross_segments=" << original_segments
              << " local_entries=" << local_entries
              << " total_list_entries=" << (cross_entries + local_entries)
              << " logical_peer_values=" << peer_words
              << " max_segments_per_owner_cycle=" << max_segments
              << " coverage_bad=" << bad
              << " exact=" << (bad ? "FAIL" : "OK")
              << '\n';
    return bad == 0;
}

} // namespace

int main() {
    for (const int W : {8, 10, 12, 14, 16, 18}) {
        const int K = (W - 2) / 2;
        const int ngpu = std::min(8, 1 << std::min(3, K));
        for (const bool reverse : {false, true}) {
            if (!verify_cycle_owner_coverage(W, K, ngpu, reverse)) return 1;
        }
    }
    std::cout << "ALL_OK production_cycle_owner_coverage=1\n";
    return 0;
}
