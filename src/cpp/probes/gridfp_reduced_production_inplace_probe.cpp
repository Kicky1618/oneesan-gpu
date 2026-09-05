#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

using Slot = std::pair<int, Rank>;

struct InplaceStats {
    Rank components = 0;
    Rank states = 0;
    Rank max_pairs = 0;
    Rank identical_slot_sets = 0;
};

InplaceStats verify_inplace_position(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const OwnerPlan& plan
) {
    const int next = reverse ? q + 1 : q - 1;
    ProductionFactorTables tables(W);
    const auto seeds = component_seeds_outer(block, W, q, reverse);

    std::set<Slot> all_source_slots;
    std::set<Slot> all_destination_slots;
    InplaceStats st;
    st.components = seeds.size();

    for (Key seed : seeds) {
        std::set<Key> sources;
        std::set<Key> destinations;
        std::deque<Key> queue;
        sources.insert(seed);
        queue.push_back(seed);
        while (!queue.empty()) {
            const Key s = queue.front();
            queue.pop_front();
            for (const auto& [d, c] : reduced_step_basis(s, W, q, reverse)) {
                if (c != 1 && c != -1) fail("inplace edge coefficient");
                if (!destinations.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, q, reverse)) {
                    if (a != 1 && a != -1) fail("inplace inverse coefficient");
                    if (sources.insert(pre).second) queue.push_back(pre);
                }
            }
        }
        if (sources.size() != destinations.size()) fail("inplace component imbalance");
        st.max_pairs = std::max<Rank>(st.max_pairs, sources.size());
        st.states += sources.size();

        std::set<Slot> src_slots;
        std::set<Slot> dst_slots;
        for (Key s : sources) {
            const GroupedRank r = grouped_rank(
                s, tables, W, q, reverse, tile_start, K, ngpu, plan);
            src_slots.emplace(r.owner, r.local);
            if (!all_source_slots.emplace(r.owner, r.local).second)
                fail("inplace source slot overlap");
        }
        for (Key d : destinations) {
            const GroupedRank r = grouped_rank(
                d, tables, W, next, reverse, tile_start, K, ngpu, plan);
            dst_slots.emplace(r.owner, r.local);
            if (!all_destination_slots.emplace(r.owner, r.local).second)
                fail("inplace destination slot overlap");
        }
        if (src_slots != dst_slots)
            fail(std::string(reverse ? "reverse" : "forward") +
                 " component slot set changed W=" + std::to_string(W) +
                 " q=" + std::to_string(q) +
                 " K=" + std::to_string(K));
        ++st.identical_slot_sets;
    }

    if (all_source_slots != all_destination_slots) fail("inplace global slot permutation");
    if (st.states != tables.size()) fail("inplace state coverage");
    return st;
}

void verify_tile_inplace(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int tile_start,
    int K,
    bool reverse,
    int ngpu
) {
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    int q = tile_start;
    Rank max_pairs = 0;
    Rank components = 0;
    for (int step = 0; step < K; ++step) {
        const InplaceStats st = verify_inplace_position(
            main, block, W, q, reverse, tile_start, K, ngpu, plan);
        max_pairs = std::max(max_pairs, st.max_pairs);
        components = st.components;
        if (st.identical_slot_sets != st.components) fail("inplace component count");
        q += reverse ? 1 : -1;
    }
    std::cout << "W=" << W
              << " tile_start=" << tile_start
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " components=" << components
              << " max_pairs=" << max_pairs
              << " component_source_destination_slots_identical=1"
              << " double_buffer_required=0"
              << " OK\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 11 || ngpu < 2 || ngpu > 16) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 7; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            for (int start = K + 2; start <= W - 1; ++start)
                verify_tile_inplace(words[W], words[W - 1], W, start, K, false, ngpu);
            for (int start = 1; start <= W - K - 2; ++start)
                verify_tile_inplace(words[W], words[W - 1], W, start, K, true, ngpu);
        }
    }

    std::cout << "W=28_theory per_gpu_u32_GiB~220.44"
              << " in_place_component_update_candidate=1"
              << " second_220GiB_buffer_eliminated=1\n";
    std::cout << "ALL_OK production_grouped_inplace_slots=1\n";
    return 0;
}
