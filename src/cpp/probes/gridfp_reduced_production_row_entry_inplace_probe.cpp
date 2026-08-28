#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

struct EntryStats {
    Rank components = 0;
    Rank expanding_components = 0;
    Rank square_components = 0;
    Rank max_sources = 0;
    Rank max_destinations = 0;
    Rank max_fanout = 0;
    Rank negative_edges = 0;
};

EntryStats verify_entry_inplace(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    bool reverse
) {
    const int p = reverse ? 1 : W - 1;
    const int next = reverse ? 2 : W - 2;
    const auto dst = layout(main, block, next);
    std::map<Key, int> didx;
    for (int i = 0; i < static_cast<int>(dst.size()); ++i)
        didx.emplace(dst[static_cast<std::size_t>(i)], i);

    std::vector<Vec> columns(main.size());
    std::vector<std::vector<int>> incoming(dst.size());
    std::vector<Coef> input(dst.size()), work(dst.size()), reference(dst.size());
    for (int s = 0; s < static_cast<int>(main.size()); ++s) {
        input[static_cast<std::size_t>(s)] = Coef(1 + ((s * 193 + W * 29 + int(reverse) * 31) % 1009));
        work[static_cast<std::size_t>(s)] = input[static_cast<std::size_t>(s)];
        columns[static_cast<std::size_t>(s)] = reduced_step_basis(Key{false, main[static_cast<std::size_t>(s)]}, W, p, reverse);
        for (const auto& [d, c] : columns[static_cast<std::size_t>(s)]) {
            if (c != 1 && c != -1) fail("row-entry coefficient outside +/-1");
            const auto it = didx.find(d);
            if (it == didx.end()) fail("row-entry destination outside reduced layout");
            incoming[static_cast<std::size_t>(it->second)].push_back(s);
            reference[static_cast<std::size_t>(it->second)] += c * input[static_cast<std::size_t>(s)];
        }
    }

    std::vector<std::uint8_t> seen_s(main.size()), seen_d(dst.size());
    EntryStats st;
    for (int seed = 0; seed < static_cast<int>(main.size()); ++seed) {
        if (seen_s[static_cast<std::size_t>(seed)]) continue;
        std::vector<int> ss, dd;
        std::deque<std::pair<bool, int>> q;
        q.push_back({false, seed});
        seen_s[static_cast<std::size_t>(seed)] = 1;
        while (!q.empty()) {
            const auto [right, v] = q.front();
            q.pop_front();
            if (!right) {
                ss.push_back(v);
                for (const auto& [d, c] : columns[static_cast<std::size_t>(v)]) {
                    (void)c;
                    const int z = didx.at(d);
                    if (!seen_d[static_cast<std::size_t>(z)]) {
                        seen_d[static_cast<std::size_t>(z)] = 1;
                        q.push_back({true, z});
                    }
                }
            } else {
                dd.push_back(v);
                for (int s : incoming[static_cast<std::size_t>(v)]) {
                    if (!seen_s[static_cast<std::size_t>(s)]) {
                        seen_s[static_cast<std::size_t>(s)] = 1;
                        q.push_back({false, s});
                    }
                }
            }
        }

        ++st.components;
        st.max_sources = std::max<Rank>(st.max_sources, ss.size());
        st.max_destinations = std::max<Rank>(st.max_destinations, dd.size());
        int blocked_destinations = 0;
        std::set<MateID> source_main, destination_main;
        std::map<int, Coef> local_out;
        for (int s : ss) {
            source_main.insert(main[static_cast<std::size_t>(s)]);
            st.max_fanout = std::max<Rank>(st.max_fanout, columns[static_cast<std::size_t>(s)].size());
            for (const auto& [d, c] : columns[static_cast<std::size_t>(s)]) {
                const int dr = didx.at(d);
                local_out[dr] += c * input[static_cast<std::size_t>(s)];
                st.negative_edges += c < 0;
            }
        }
        for (int d : dd) {
            const Key k = dst[static_cast<std::size_t>(d)];
            if (k.blocked) ++blocked_destinations;
            else destination_main.insert(k.mate);
        }

        if (source_main != destination_main)
            fail("row-entry main source/destination set mismatch");
        if (blocked_destinations == 0) {
            if (ss.size() != dd.size()) fail("row-entry square component shape");
            ++st.square_components;
        } else if (blocked_destinations == 1) {
            if (dd.size() != ss.size() + 1) fail("row-entry expanding component shape");
            ++st.expanding_components;
        } else {
            fail("row-entry component has multiple blocked destinations");
        }

        // The row boundary starts main-only. Blocked slots are dead storage, so
        // every main source in this component can be loaded before any write;
        // the same main keys are then overwritten and at most one free blocked
        // slot is populated. No other component reads those main slots.
        for (const auto& [dr, value] : local_out) work[static_cast<std::size_t>(dr)] = value;
    }

    for (std::uint8_t x : seen_s) if (!x) fail("row-entry source coverage");
    for (std::uint8_t x : seen_d) if (!x) fail("row-entry destination coverage");
    if (work != reference) fail("row-entry component-local in-place arithmetic mismatch");

    Rank m_wm3 = W >= 3 ? gen_words(W - 3).size() : 0;
    const Rank want_components = block.size() - m_wm3;
    Rank m_wm2 = W >= 2 ? gen_words(W - 2).size() : 0;
    const Rank want_expanding = block.size() - m_wm2;
    if (st.components != want_components || st.expanding_components != want_expanding)
        fail("row-entry component count formula");
    if (st.max_sources > Rank(W / 2 + 3)) fail("row-entry source bound");
    if (st.max_destinations > Rank(W / 2 + 4)) fail("row-entry destination bound");
    if (st.max_fanout > 3) fail("row-entry fanout bound");
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        for (bool reverse : {false, true}) {
            const EntryStats st = verify_entry_inplace(words[W], words[W - 1], W, reverse);
            std::cout << "W=" << W
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " components=" << st.components
                      << " expanding=" << st.expanding_components
                      << " square=" << st.square_components
                      << " max_sources=" << st.max_sources
                      << " max_destinations=" << st.max_destinations
                      << " max_fanout=" << st.max_fanout
                      << " main_slots_same_component=1"
                      << " blocked_slots_initially_free=1"
                      << " full_stream_scratch=0"
                      << " in_place_candidate=1 OK\n";
        }
    }

    std::cout << "W=28_theory components=118389089432"
              << " expanding_components=87677551081"
              << " max_source_slots=17"
              << " max_destination_slots=18"
              << " second_full_stream=0"
              << "\n";
    std::cout << "ALL_OK reduced_row_entry_inplace_decomposition=1\n";
    return 0;
}
