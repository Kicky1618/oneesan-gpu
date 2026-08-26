#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

static std::vector<U64> high_counts(int len) {
    std::vector<U64> cur(std::size_t(len + 3), 0), next(cur.size(), 0);
    cur[1] = 1;
    for (int step = 0; step < len; ++step) {
        std::fill(next.begin(), next.end(), 0);
        for (int h = 0; h <= len + 1; ++h) {
            const U64 c = cur[std::size_t(h)];
            if (!c) continue;
            next[std::size_t(h)] += c;
            if (h > 0) next[std::size_t(h - 1)] += c;
            next[std::size_t(h + 1)] += c;
        }
        cur.swap(next);
    }
    cur.resize(std::size_t(len + 2));
    return cur;
}

// With a fully fixed LOW occupancy mask, unoccupied positions are identity
// (forced N) steps. Therefore the per-height counts depend only on the number
// k of occupied positions, not on their locations. This is make_spec() with
// k +/- steps and LOW_LUT_K-k identity steps removed.
static std::vector<U64> low_counts_for_popcount(int low, int k) {
    std::vector<U64> cur(std::size_t(low + 3), 0), next(cur.size(), 0);
    cur[0] = 1;
    for (int step = 0; step < k; ++step) {
        std::fill(next.begin(), next.end(), 0);
        for (int h = 0; h <= low + 1; ++h) {
            U64 x = 0;
            if (h > 0) x += cur[std::size_t(h - 1)];
            x += cur[std::size_t(h + 1)];
            next[std::size_t(h)] = x;
        }
        cur.swap(next);
    }
    cur.resize(std::size_t(low + 2));
    return cur;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || low >= 31 || high < 1 || ngpu < 1) return 1;

    const auto hc = high_counts(high);
    std::vector<std::vector<U64>> layouts;
    std::vector<U64> work;
    layouts.reserve(std::size_t(low + 1));
    work.reserve(std::size_t(low + 1));

    bool strict_work = true;
    U64 prev_work = 0;
    for (int k = 0; k <= low; ++k) {
        auto lc = low_counts_for_popcount(low, k);
        U64 main_n = 0, block_n = 0;
        for (int he = 0; he <= high + 1; ++he) {
            const U64 rows = hc[std::size_t(he)];
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs >= 0 && hs <= low + 1)
                    main_n += rows * lc[std::size_t(hs)];
            }
            if (he <= low + 1)
                block_n += rows * lc[std::size_t(he)];
        }
        const U64 w = main_n + block_n;
        if (k && w <= prev_work) strict_work = false;
        prev_work = w;
        layouts.push_back(std::move(lc));
        work.push_back(w);
    }

    U64 distinct_layouts = 0;
    for (std::size_t i = 0; i < layouts.size(); ++i) {
        bool first = true;
        for (std::size_t j = 0; j < i; ++j)
            if (layouts[i] == layouts[j]) { first = false; break; }
        if (first) ++distinct_layouts;
    }

    const U64 masks = U64(1) << low;
    const U64 jobs = masks * U64(W);
    const U64 old_copy_calls = jobs * 2;
    const U64 blocks_per_mask = U64(3 * (high + 2) + (high + 2));
    constexpr U64 fblock_bytes = 24;
    const U64 bytes_per_layout = blocks_per_mask * fblock_bytes;
    const U64 old_payload = jobs * bytes_per_layout;

    U64 layout_worker_visits_per_row = 0;
    for (int k = 0; k <= low; ++k)
        layout_worker_visits_per_row += std::min<U64>(U64(ngpu), choose_u64(low, k));
    const U64 upper_copy_calls = strict_work
        ? U64(W) * layout_worker_visits_per_row * 2
        : old_copy_calls;
    const U64 upper_payload = strict_work
        ? U64(W) * layout_worker_visits_per_row * bytes_per_layout
        : old_payload;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (!strict_work
            || distinct_layouts != 15ULL
            || jobs != 458752ULL
            || layout_worker_visits_per_row != 106ULL
            || old_copy_calls != 917504ULL
            || upper_copy_calls != 5936ULL
            || bytes_per_layout != 1440ULL
            || old_payload != 660602880ULL
            || upper_payload != 4273920ULL) {
            std::cerr << "n=27 HIGH FBlock layout-dedup regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-fblock-layout-dedup W=" << W
              << " low=" << low << " high=" << high
              << " ngpu=" << ngpu << '\n'
              << "distinct_layouts=" << distinct_layouts
              << " strict_work_by_popcount=" << (strict_work ? 1 : 0) << '\n'
              << "layout_worker_visits_upper_per_row="
              << layout_worker_visits_per_row << '\n'
              << "old_fblock_copy_calls_per_residue=" << old_copy_calls
              << " dedup_copy_calls_upper_per_residue=" << upper_copy_calls
              << " call_reduction_lower_pct="
              << 100.0 * (1.0 - double(upper_copy_calls) / double(old_copy_calls))
              << '\n'
              << "old_fblock_payload_bytes_per_residue=" << old_payload
              << " dedup_payload_upper_bytes_per_residue=" << upper_payload
              << " payload_reduction_lower_pct="
              << 100.0 * (1.0 - double(upper_payload) / double(old_payload))
              << '\n';
    return 0;
}
