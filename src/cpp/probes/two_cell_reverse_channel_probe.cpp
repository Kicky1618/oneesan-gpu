#pragma push_macro("main")
#undef main
#define main two_cell_channel_probe_main_unused
#include "two_cell_channel_probe.cpp"
#pragma pop_macro("main")

namespace {

Word reflect_word(const Word& w) {
    const LinkState s = decode(w);
    const int n = static_cast<int>(w.size());
    LinkState t;
    t.mate.assign(static_cast<std::size_t>(n), -2);
    t.root = n - 1 - s.root;
    t.mate[t.root] = -1;
    for (int p = 0; p < n; ++p) {
        if (s.mate[p] < 0) continue;
        t.mate[n - 1 - p] = n - 1 - s.mate[p];
    }
    return encode(t);
}

Key reflect_key(const Key& k) {
    return Key{k.type, reflect_word(k.w)};
}

Key project_key_reverse(const Key& k, int pair, int W) {
    if (k.type == 'A') return k;
    assert(k.type == 'C');
    const int fixed = pair - 1;
    if (pair >= 1 && fixed < static_cast<int>(k.w.size()) && k.w[fixed] == N) {
        const Word a = k.w.substr(0, fixed) + L + R + k.w.substr(fixed + 1);
        assert(static_cast<int>(a.size()) == W - 1 && valid_word(a));
        return Key{'A', a};
    }
    return k;
}

std::vector<Key> reverse_q_basis(
    int W, int pair, const std::vector<std::vector<Word>>& words
) {
    const int fixed = pair - 1;
    std::vector<Key> out;
    for (const Word& w : words[W - 1]) out.push_back(Key{'A', w});
    for (const Word& w : words[W - 2])
        if (w[fixed] != N) out.push_back(Key{'C', w});
    return out;
}

CVec K_reverse_basis(const Key& src, int W, int pair) {
    assert(pair >= 2 && pair <= W - 2);
    CVec out;
    for (const Word& w : E_raw_basis(src, pair)) {
        const CVec r = R_raw_basis(w, pair - 1);
        for (const auto& kv : r)
            add(out, project_key_reverse(kv.first, pair - 1, W), kv.second);
    }
    return out;
}

CVec reflect_vec(const CVec& v) {
    CVec out;
    for (const auto& kv : v) add(out, reflect_key(kv.first), kv.second);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        for (const Word& w : words[W]) {
            if (reflect_word(reflect_word(w)) != w)
                fail("reflection involution W=" + std::to_string(W));
            for (int i = 0; i < W - 1; ++i) {
                const int j = W - 2 - i;
                CVec lhs, rhs;
                for (const Word& z : apply_T_basis(w, i))
                    add(lhs, Key{'A', reflect_word(z)});
                for (const Word& z : apply_T_basis(reflect_word(w), j))
                    add(rhs, Key{'A', z});
                if (lhs != rhs)
                    fail("T reflection W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
            }
        }

        Rank checked = 0;
        for (int i = 0; i <= W - 4; ++i) {
            const int j = W - 2 - i;
            std::vector<Key> fsrc;
            for (const Word& w : words[W - 1]) fsrc.push_back(Key{'A', w});
            for (const Word& w : words[W - 2])
                if (w[i] != N) fsrc.push_back(Key{'C', w});

            const auto rsrc = reverse_q_basis(W, j, words);
            const auto rdst = reverse_q_basis(W, j - 1, words);

            std::set<Key> mirrored;
            for (const Key& k : rsrc) mirrored.insert(reflect_key(k));
            if (mirrored != std::set<Key>(fsrc.begin(), fsrc.end()))
                fail("layout reflection W=" + std::to_string(W) +
                     " i=" + std::to_string(i));

            // Forward Q_i fixes C position i. Reflection maps that support bit
            // to W-3-i = j-1, hence reverse Q_j fixes the left lookahead bit.
            for (const Word& w : words[W - 2]) {
                const Key raw{'C', w};
                const Key lhs = reflect_key(project_key(raw, i, W));
                const Key rhs = project_key_reverse(reflect_key(raw), j, W);
                if (!(lhs == rhs))
                    fail("quotient reflection W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
            }

            std::set<Key> rdst_set(rdst.begin(), rdst.end());
            for (const Key& src : rsrc) {
                const Key fwd_src = reflect_key(src);
                const CVec conjugated = reflect_vec(K_basis(fwd_src, W, i));
                const CVec direct = K_reverse_basis(src, W, j);
                if (direct != conjugated)
                    fail("reverse K conjugacy W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                for (const auto& kv : direct)
                    if (!rdst_set.count(kv.first))
                        fail("reverse destination layout");
                ++checked;
            }
        }

        std::cout << "W=" << W
                  << " reverse_sources_checked=" << checked
                  << " quotient_fixed=pair-1"
                  << " J_involution=OK"
                  << " Jinv_K_J=OK\n";
    }
    std::cout << "ALL_OK reverse_two_cell=1\n";
    return 0;
}
