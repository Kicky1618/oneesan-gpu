#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_factorized_probe_main_unused
#include "gridfp_reduced_production_factorized_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>
#include <set>

namespace {

std::vector<Key> alignment_component_seeds(
    const std::vector<MateID>& labels, int W, int p, bool reverse
) {
    std::vector<Key> out;
    for (MateID v : labels) {
        const int other = reverse ? p : p - 2;
        if (mget(v, p - 1) == N && mget(v, other) == N) continue;
        if (mget(v, p - 1) != N) out.push_back(Key{true, v});
        else out.push_back(Key{false, reverse
            ? blocked_exclude_reverse(v, W, p)
            : blocked_exclude(v, p)});
    }
    return out;
}

struct AlignmentStats {
    Rank components = 0;
    Rank states = 0;
    Rank aligned_ranks = 0;
    Rank max_pairs = 0;
};

AlignmentStats verify_alignment_position(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const ProductionFactorTables& tables,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    ProductionFactorCodec src_codec(tables, p - 1);
    ProductionFactorCodec dst_codec(tables, next - 1);
    const Rank n = src_codec.size();
    if (dst_codec.size() != n) fail("alignment non-square factor layout");

    const Rank bad = std::numeric_limits<Rank>::max();
    std::vector<Rank> source_owner(static_cast<std::size_t>(n), bad);
    std::vector<Rank> dest_owner(static_cast<std::size_t>(n), bad);
    const auto seeds = alignment_component_seeds(block, W, p, reverse);

    AlignmentStats st;
    st.components = seeds.size();
    st.states = n;

    Rank cid = 0;
    for (Key seed : seeds) {
        std::set<Key> sources;
        std::set<Key> destinations;
        std::deque<Key> queue;
        sources.insert(seed);
        queue.push_back(seed);
        while (!queue.empty()) {
            const Key s = queue.front();
            queue.pop_front();
            for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse)) {
                if (c != 1 && c != -1) fail("alignment edge coefficient");
                if (!destinations.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, p, reverse)) {
                    if (a != 1 && a != -1) fail("alignment inverse coefficient");
                    if (sources.insert(pre).second) queue.push_back(pre);
                }
            }
        }
        if (sources.size() != destinations.size()) fail("alignment component imbalance");
        st.max_pairs = std::max<Rank>(st.max_pairs, sources.size());

        for (Key s : sources) {
            const Rank r = src_codec.rank(s);
            if (r >= n || source_owner[static_cast<std::size_t>(r)] != bad)
                fail("alignment source rank overlap");
            source_owner[static_cast<std::size_t>(r)] = cid;
        }
        for (Key d : destinations) {
            const Rank r = dst_codec.rank(d);
            if (r >= n || dest_owner[static_cast<std::size_t>(r)] != bad)
                fail("alignment destination rank overlap");
            dest_owner[static_cast<std::size_t>(r)] = cid;
        }
        ++cid;
    }
    if (cid != st.components) fail("alignment component count");

    for (Rank r = 0; r < n; ++r) {
        const Rank a = source_owner[static_cast<std::size_t>(r)];
        const Rank b = dest_owner[static_cast<std::size_t>(r)];
        if (a == bad || b == bad) fail("alignment uncovered factor rank");
        if (a != b)
            fail("factorized rank crosses component W=" + std::to_string(W) +
                 " p=" + std::to_string(p) +
                 " reverse=" + std::to_string(reverse) +
                 " rank=" + std::to_string(r));
        ++st.aligned_ranks;
    }
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;
    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        ProductionFactorTables tables(W);
        AlignmentStats ref{};
        bool first = true;
        for (int p = W - 1; p >= 3; --p) {
            const auto s = verify_alignment_position(words[W], words[W - 1], tables, W, p, false);
            if (first) { ref = s; first = false; }
            else if (s.components != ref.components || s.states != ref.states ||
                     s.aligned_ranks != ref.aligned_ranks)
                fail("alignment forward position totals");
            ref.max_pairs = std::max(ref.max_pairs, s.max_pairs);
        }
        for (int p = 1; p <= W - 3; ++p) {
            const auto s = verify_alignment_position(words[W], words[W - 1], tables, W, p, true);
            if (s.components != ref.components || s.states != ref.states ||
                s.aligned_ranks != ref.aligned_ranks)
                fail("alignment reverse position totals");
            ref.max_pairs = std::max(ref.max_pairs, s.max_pairs);
        }
        if (ref.aligned_ranks != ref.states) fail("alignment incomplete");
        std::cout << "W=" << W
                  << " states=" << ref.states
                  << " components=" << ref.components
                  << " max_pairs=" << ref.max_pairs
                  << " factor_rank_component_crossings=0"
                  << " source_rank_set_equals_destination_rank_set_per_component=1"
                  << " inplace_component_writes_safe=1"
                  << " forward=OK reverse=OK\n";
    }

    constexpr Rank states28 = 473397057701ULL;
    constexpr Rank components28 = 118389089432ULL;
    const double gib = double(states28) * 4.0 / double(1ULL << 30);
    std::cout << "W=28_theory states=" << states28
              << " components=" << components28
              << " one_u32_buffer_GiB=" << gib
              << " per_8gpu_GiB=" << gib / 8.0
              << " factorized_component_crossings_candidate=0"
              << '\n';
    std::cout << "ALL_OK production_factorized_component_alignment=1\n";
    return 0;
}
