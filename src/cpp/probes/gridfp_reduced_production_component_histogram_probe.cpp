#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_kernel_probe_main_unused
#include "gridfp_reduced_production_component_kernel_probe.cpp"
#pragma pop_macro("main")

#include <iomanip>
#include <map>

namespace {

struct HistogramStats {
    Rank states = 0;
    Rank components = 0;
    std::map<Rank, Rank> pairs;
};

HistogramStats component_histogram(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    const auto src = layout(main, block, p);
    const auto dst = layout(main, block, next);
    const Rank n = src.size();
    if (dst.size() != n) fail("histogram square layout");

    std::map<Key, Rank> sr, dr;
    for (Rank i = 0; i < n; ++i) {
        sr.emplace(src[static_cast<std::size_t>(i)], i);
        dr.emplace(dst[static_cast<std::size_t>(i)], i);
    }
    std::vector<std::vector<Rank>> s2d(static_cast<std::size_t>(n));
    std::vector<std::vector<Rank>> d2s(static_cast<std::size_t>(n));
    for (Rank s = 0; s < n; ++s) {
        for (const auto& [d,c] : reduced_step_basis(src[static_cast<std::size_t>(s)], W, p, reverse)) {
            if (c != 1 && c != -1) fail("histogram coefficient");
            const Rank x = dr.at(d);
            s2d[static_cast<std::size_t>(s)].push_back(x);
            d2s[static_cast<std::size_t>(x)].push_back(s);
        }
    }

    std::vector<std::uint8_t> seen_s(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> seen_d(static_cast<std::size_t>(n));
    HistogramStats out;
    out.states = n;
    for (Rank root = 0; root < n; ++root) {
        if (seen_s[static_cast<std::size_t>(root)]) continue;
        std::deque<std::pair<bool,Rank>> q;
        seen_s[static_cast<std::size_t>(root)] = 1;
        q.emplace_back(false, root);
        Rank ns = 0, nd = 0;
        while (!q.empty()) {
            const auto [is_d, x] = q.front();
            q.pop_front();
            if (!is_d) {
                ++ns;
                for (Rank y : s2d[static_cast<std::size_t>(x)]) {
                    if (!seen_d[static_cast<std::size_t>(y)]) {
                        seen_d[static_cast<std::size_t>(y)] = 1;
                        q.emplace_back(true, y);
                    }
                }
            } else {
                ++nd;
                for (Rank y : d2s[static_cast<std::size_t>(x)]) {
                    if (!seen_s[static_cast<std::size_t>(y)]) {
                        seen_s[static_cast<std::size_t>(y)] = 1;
                        q.emplace_back(false, y);
                    }
                }
            }
        }
        if (ns != nd) fail("histogram unbalanced component");
        ++out.components;
        ++out.pairs[ns];
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        const HistogramStats f = component_histogram(words[W], words[W - 1], W, W - 1, false);
        const HistogramStats r = component_histogram(words[W], words[W - 1], W, 1, true);
        if (f.states != r.states || f.components != r.components || f.pairs != r.pairs)
            fail("histogram direction mismatch W=" + std::to_string(W));

        Rank le3 = 0, le8 = 0, max_pairs = 0, weighted = 0;
        std::cout << "W=" << W
                  << " states=" << f.states
                  << " components=" << f.components
                  << " histogram=";
        bool first = true;
        for (const auto& [pairs,count] : f.pairs) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << pairs << ':' << count;
            if (pairs <= 3) le3 += count;
            if (pairs <= 8) le8 += count;
            max_pairs = std::max(max_pairs, pairs);
            weighted += pairs * count;
        }
        if (weighted != f.states) fail("histogram weighted state count");
        std::cout << " avg_pairs=" << std::setprecision(9)
                  << double(f.states) / double(f.components)
                  << " frac_le3=" << double(le3) / double(f.components)
                  << " frac_le8=" << double(le8) / double(f.components)
                  << " max_pairs=" << max_pairs
                  << " forward_reverse_same=1\n";
    }

    constexpr Rank D28 = 473397057701ULL;
    constexpr Rank C28 = 118389089432ULL;
    std::cout << "W=28_theory states=" << D28
              << " components=" << C28
              << " avg_pairs=" << std::setprecision(12) << double(D28) / double(C28)
              << " subgroup_candidate=8"
              << " components_per_warp_candidate=4\n";
    std::cout << "ALL_OK production_component_histogram=1\n";
    return 0;
}
