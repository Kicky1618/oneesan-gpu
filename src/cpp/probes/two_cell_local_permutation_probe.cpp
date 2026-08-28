#pragma push_macro("main")
#undef main
#define main two_cell_inplace_shear_probe_main_unused
#include "two_cell_inplace_shear_probe.cpp"
#pragma pop_macro("main")

namespace {

Key local_permutation_forward(const Key& src, int i) {
    const int j = i + 1;
    if (src.type == 'C') {
        const Word& w = src.w;
        return Key{'A', w.substr(0, j) + N + w.substr(j)};
    }

    assert(src.type == 'A');
    const Word& w = src.w;
    if (w[j] == N && w[j + 1] != N)
        return Key{'C', w.substr(0, j) + w.substr(j + 1)};

    if (w[i] != N && w[j] == N && w[j + 1] == N)
        return Key{'A', w.substr(0, j) + L + R + w.substr(j + 2)};

    if (w[i] != N && w[j] == L && w[j + 1] == R)
        return Key{'A', w.substr(0, i) + N + N + w[i] + w.substr(j + 2)};

    return src;
}

Key local_permutation_inverse(const Key& dst, int i) {
    const int j = i + 1;
    if (dst.type == 'C') {
        const Word& w = dst.w;
        return Key{'A', w.substr(0, j) + N + w.substr(j)};
    }

    assert(dst.type == 'A');
    const Word& w = dst.w;
    if (w[i] != N && w[j] == N)
        return Key{'C', w.substr(0, j) + w.substr(j + 1)};

    if (w[i] != N && w[j] == L && w[j + 1] == R)
        return Key{'A', w.substr(0, j) + N + N + w.substr(j + 2)};

    if (w[i] == N && w[j] == N && w[j + 1] != N)
        return Key{'A', w.substr(0, i) + w[j + 1] + L + R + w.substr(j + 2)};

    return dst;
}

struct LocalPermutationStats {
    Rank aa_identity = 0;
    Rank aa_make_cup = 0;
    Rank aa_consume_cup = 0;
    Rank ac_delete_zero = 0;
    Rank ca_insert_zero = 0;
};

LocalPermutationStats verify_local_permutation_step(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words
) {
    const ReducedLayout src = make_layout(W, i, words);
    const ReducedLayout dst = make_layout(W, i + 1, words);
    const StepGraph g = build_step_graph(src, dst, W, i);
    const auto [match_l, match_r] = forced_matching(g);
    LocalPermutationStats stats;

    for (Rank s = 0; s < src.size(); ++s) {
        const Key predicted = local_permutation_forward(src.key[s], i);
        const auto it = dst.rank.find(predicted);
        if (it == dst.rank.end())
            fail("local permutation destination missing W=" + std::to_string(W) +
                 " i=" + std::to_string(i));
        if (it->second != match_l[s])
            fail("local permutation differs from forced matching W=" +
                 std::to_string(W) + " i=" + std::to_string(i) +
                 " s=" + std::to_string(s));

        bool edge = false;
        for (Rank d : g.out[s]) edge |= d == it->second;
        if (!edge) fail("local permutation is not an operator edge");

        if (!(local_permutation_inverse(predicted, i) == src.key[s]))
            fail("local permutation inverse mismatch");

        if (src.key[s].type == 'C') {
            if (predicted.type != 'A') fail("expected C->A permutation");
            ++stats.ca_insert_zero;
        } else if (predicted.type == 'C') {
            ++stats.ac_delete_zero;
        } else if (predicted == src.key[s]) {
            ++stats.aa_identity;
        } else {
            const Word& a = src.key[s].w;
            const Word& b = predicted.w;
            const int j = i + 1;
            if (a[i] != N && a[j] == N && a[j + 1] == N &&
                b[i] == a[i] && b[j] == L && b[j + 1] == R) {
                ++stats.aa_make_cup;
            } else if (a[i] != N && a[j] == L && a[j + 1] == R &&
                       b[i] == N && b[j] == N && b[j + 1] == a[i]) {
                ++stats.aa_consume_cup;
            } else {
                fail("unclassified A->A permutation rule");
            }
        }
    }

    for (Rank d = 0; d < dst.size(); ++d) {
        const Key predicted = local_permutation_inverse(dst.key[d], i);
        const auto it = src.rank.find(predicted);
        if (it == src.rank.end() || it->second != match_r[d])
            fail("local inverse differs from forced matching");
        if (!(local_permutation_forward(predicted, i) == dst.key[d]))
            fail("local inverse forward mismatch");
    }

    const Rank m1 = static_cast<Rank>(words[W - 1].size());
    const Rank m2 = static_cast<Rank>(words[W - 2].size());
    const Rank m3 = static_cast<Rank>(words[W - 3].size());
    const Rank m4 = W >= 5 ? static_cast<Rank>(words[W - 4].size()) : 0;
    const Rank c = m2 - m3;
    const Rank cup = m3 - m4;
    const Rank identity = m1 - m2 - m3 + 2 * m4;

    if (stats.ac_delete_zero != c || stats.ca_insert_zero != c ||
        stats.aa_make_cup != cup || stats.aa_consume_cup != cup ||
        stats.aa_identity != identity)
        fail("local permutation count formula mismatch");

    return stats;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 4 || maxW > 14) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        LocalPermutationStats reference{};
        bool first = true;
        for (int i = 0; i <= W - 4; ++i) {
            const auto s = verify_local_permutation_step(W, i, words);
            if (first) {
                reference = s;
                first = false;
            } else if (s.aa_identity != reference.aa_identity ||
                       s.aa_make_cup != reference.aa_make_cup ||
                       s.aa_consume_cup != reference.aa_consume_cup ||
                       s.ac_delete_zero != reference.ac_delete_zero ||
                       s.ca_insert_zero != reference.ca_insert_zero) {
                fail("local permutation counts depend on position");
            }
        }
        const Rank dim = static_cast<Rank>(words[W - 1].size()) +
                         static_cast<Rank>(words[W - 2].size()) -
                         static_cast<Rank>(words[W - 3].size());
        std::cout << "W=" << W
                  << " dim=" << dim
                  << " aa_identity=" << reference.aa_identity
                  << " aa_XNN_to_XLR=" << reference.aa_make_cup
                  << " aa_XLR_to_NNX=" << reference.aa_consume_cup
                  << " ac_delete_N=" << reference.ac_delete_zero
                  << " ca_insert_N=" << reference.ca_insert_zero
                  << " local_P=1 inverse_local=1 forced_matching_equal=1\n";
    }
    std::cout << "ALL_OK local_forest_permutation=1\n";
    return 0;
}
