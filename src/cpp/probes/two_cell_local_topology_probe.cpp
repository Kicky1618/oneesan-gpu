#pragma push_macro("main")
#undef main
#define main two_cell_local_permutation_probe_main_unused
#include "two_cell_local_permutation_probe.cpp"
#pragma pop_macro("main")

namespace {

Word primitive_word(const Word& w) {
    Word out;
    out.reserve(w.size());
    for (char c : w) if (c != N) out.push_back(c);
    return out;
}

int occupied_prefix(const Word& w, int end_inclusive) {
    int n = 0;
    for (int i = 0; i <= end_inclusive; ++i) n += w[i] != N;
    return n;
}

void verify_local_topology_step(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words,
    Rank& topology_preserving,
    Rank& cup_insert,
    Rank& cup_remove
) {
    const ReducedLayout src = make_layout(W, i, words);
    const ReducedLayout dst = make_layout(W, i + 1, words);
    const StepGraph g = build_step_graph(src, dst, W, i);
    const auto [match_l, match_r] = forced_matching(g);
    (void)match_r;

    for (Rank s = 0; s < src.size(); ++s) {
        const Key& a = src.key[s];
        const Key& b = dst.key[match_l[s]];
        if (!(b == local_permutation_forward(a, i)))
            fail("topology probe local permutation mismatch");

        const Word pa = primitive_word(a.w);
        const Word pb = primitive_word(b.w);
        if (pa == pb) {
            ++topology_preserving;
            continue;
        }

        if (a.type != 'A' || b.type != 'A')
            fail("topology-changing cross-type permutation");

        const int j = i + 1;
        const int split = occupied_prefix(a.w, i);
        if (a.w[i] != N && a.w[j] == N && a.w[j + 1] == N) {
            const Word expected = pa.substr(0, split) + L + R + pa.substr(split);
            if (pb != expected) fail("primitive cup insertion formula mismatch");
            ++cup_insert;
        } else if (a.w[i] != N && a.w[j] == L && a.w[j + 1] == R) {
            if (split + 1 >= static_cast<int>(pa.size()) ||
                pa[split] != L || pa[split + 1] != R)
                fail("primitive cup removal source mismatch");
            const Word expected = pa.substr(0, split) + pa.substr(split + 2);
            if (pb != expected) fail("primitive cup removal formula mismatch");
            ++cup_remove;
        } else {
            fail("unclassified primitive topology change");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 4 || maxW > 14) return 2;
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank preserve = 0, insert = 0, remove = 0;
        for (int i = 0; i <= W - 4; ++i) {
            Rank p = 0, a = 0, r = 0;
            verify_local_topology_step(W, i, words, p, a, r);
            if (i == 0) {
                preserve = p;
                insert = a;
                remove = r;
            } else if (p != preserve || a != insert || r != remove) {
                fail("topology action counts depend on position");
            }
        }
        const Rank m1 = static_cast<Rank>(words[W - 1].size());
        const Rank m2 = static_cast<Rank>(words[W - 2].size());
        const Rank m3 = static_cast<Rank>(words[W - 3].size());
        const Rank m4 = W >= 5 ? static_cast<Rank>(words[W - 4].size()) : 0;
        const Rank dim = m1 + m2 - m3;
        const Rank expected_change_each = m3 - m4;
        if (insert != expected_change_each || remove != expected_change_each ||
            preserve + insert + remove != dim)
            fail("topology action count formula mismatch");
        std::cout << "W=" << W
                  << " dim=" << dim
                  << " primitive_preserve=" << preserve
                  << " primitive_insert_LR=" << insert
                  << " primitive_remove_LR=" << remove
                  << " preserve_fraction=" << (double(preserve) / double(dim))
                  << " local_topology_action=1\n";
    }

    constexpr int W = 28;
    const std::uint64_t m1 = count_words_u64(W - 1);
    const std::uint64_t m2 = count_words_u64(W - 2);
    const std::uint64_t m3 = count_words_u64(W - 3);
    const std::uint64_t m4 = count_words_u64(W - 4);
    const std::uint64_t dim = m1 + m2 - m3;
    const std::uint64_t each = m3 - m4;
    const std::uint64_t preserve = dim - 2 * each;
    std::cout << "W28_topology_projection"
              << " reduced=" << dim
              << " primitive_preserve=" << preserve
              << " primitive_insert_LR=" << each
              << " primitive_remove_LR=" << each
              << " preserve_fraction=" << (double(preserve) / double(dim))
              << '\n';
    return 0;
}
