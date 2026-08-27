#pragma push_macro("main")
#undef main
#define main two_cell_channel_probe_main_unused
#include "two_cell_channel_probe.cpp"
#pragma pop_macro("main")

#include <optional>

namespace {

std::vector<int> path_heights(const Word& w) {
    std::vector<int> h(w.size() + 1);
    h[0] = 1;
    for (std::size_t i = 0; i < w.size(); ++i) {
        h[i + 1] = h[i];
        if (w[i] == L) ++h[i + 1];
        else if (w[i] == R) --h[i + 1];
    }
    return h;
}

Word split_arc_candidate(const Word& z, int j, int p, int q) {
    Word w = z;
    if (q < j) {
        w[q] = L;
        w[j] = R;
        w[j + 1] = R;
    } else if (p > j + 1) {
        w[p] = R;
        w[j] = L;
        w[j + 1] = L;
    } else if (p < j && q > j + 1) {
        w[j] = R;
        w[j + 1] = L;
    } else {
        return {};
    }
    return valid_word(w) ? w : Word{};
}

Word split_root_candidate(const Word& z, int j, int root) {
    Word w = z;
    if (root < j) {
        w[root] = L;
        w[j] = R;
        w[j + 1] = R;
    } else if (root > j + 1) {
        w[j] = R;
        w[j + 1] = L;
    } else {
        return {};
    }
    return valid_word(w) ? w : Word{};
}

std::set<Word> inverse_R_raw(const Key& raw, int j, int W) {
    std::set<Word> out;
    if (raw.type == 'C') {
        const Word w = raw.w.substr(0, j) + Word(2, N) + raw.w.substr(j);
        if (static_cast<int>(w.size()) != W || !valid_word(w)) std::abort();
        out.insert(w);
        return out;
    }

    assert(raw.type == 'A');
    assert(static_cast<int>(raw.w.size()) == W - 1);
    const char local = raw.w[j];
    if (local != N) {
        const Word a = raw.w.substr(0, j) + local + N + raw.w.substr(j + 1);
        const Word b = raw.w.substr(0, j) + N + local + raw.w.substr(j + 1);
        if (valid_word(a)) out.insert(a);
        if (valid_word(b)) out.insert(b);
        return out;
    }

    // Expand the collapsed vacancy back to the two vacant sites seen after a
    // cap. Every nontrivial inverse is obtained by cutting one strand on the
    // boundary of the face containing the gap j,j+1.
    const Word z = raw.w.substr(0, j) + Word(2, N) + raw.w.substr(j + 1);
    if (static_cast<int>(z.size()) != W || !valid_word(z)) std::abort();
    out.insert(z); // vacancy -> vacancy

    const LinkState state = decode(z);
    const std::vector<int> h = path_heights(z);
    const int level = h[j];

    // Maximal level-face interval containing the two vacancies. Vertices in
    // [left,right] stay at height >= level. Excursions that start at exactly
    // level are the sibling strands visible from this face.
    int left = j;
    while (left > 0 && h[left - 1] >= level) --left;
    int right = j + 2;
    while (right < W && h[right + 1] >= level) ++right;

    for (int p = 0; p < W; ++p) {
        const int q = state.mate[p];
        if (q <= p) continue;
        if (p >= left && q < right && h[p] == level) {
            const Word w = split_arc_candidate(z, j, p, q);
            if (!w.empty()) out.insert(w);
        }
    }

    // If the face is enclosed, its innermost enclosing arc is another legal
    // strand to cut. Cutting any outer enclosing arc would encode the same
    // local RL word, so only the face boundary is needed.
    if (left > 0) {
        const int p = left - 1;
        const int q = state.mate[p];
        if (z[p] == L && q == right && p < j && q > j + 1) {
            const Word w = split_arc_candidate(z, j, p, q);
            if (!w.empty()) out.insert(w);
        }
    }

    // The distinguished root strand bounds the level-1 prefix face and is an
    // ordinary exposed strand on the outer level-0 face.
    if (level == 0 || (left == 0 && right < W && state.root == right)) {
        const Word w = split_root_candidate(z, j, state.root);
        if (!w.empty()) out.insert(w);
    }

    return out;
}

std::vector<Key> inverse_project(const Key& dst, int j, int W) {
    std::vector<Key> out{dst};
    if (dst.type == 'A' && j <= W - 3 && j + 1 < static_cast<int>(dst.w.size()) &&
        dst.w[j] == L && dst.w[j + 1] == R) {
        const Key c{'C', dst.w.substr(0, j) + N + dst.w.substr(j + 2)};
        if (!(project_key(c, j, W) == dst)) std::abort();
        out.push_back(c);
    }
    return out;
}

std::optional<Key> inverse_E(const Word& z, int i) {
    const bool a = z[i] != N, b = z[i + 1] != N;
    if (!a && !b)
        return Key{'A', z.substr(0, i) + N + z.substr(i + 2)};
    if (a != b) {
        const char s = a ? z[i] : z[i + 1];
        return Key{'A', z.substr(0, i) + s + z.substr(i + 2)};
    }
    if (z[i] == L && z[i + 1] == R)
        return Key{'C', z.substr(0, i) + z.substr(i + 2)};
    return std::nullopt;
}

bool in_source_layout(const Key& k, int W, int i) {
    if (k.type == 'A')
        return static_cast<int>(k.w.size()) == W - 1 && valid_word(k.w);
    if (k.type == 'C')
        return static_cast<int>(k.w.size()) == W - 2 && valid_word(k.w) &&
               (i > W - 3 || k.w[i] != N);
    return false;
}

std::set<Key> inverse_K(const Key& dst, int W, int i) {
    const int j = i + 1;
    std::set<Key> out;
    for (const Key& raw : inverse_project(dst, j, W)) {
        for (const Word& z : inverse_R_raw(raw, j, W)) {
            const auto src = inverse_E(z, i);
            if (src && in_source_layout(*src, W, i)) out.insert(*src);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::stoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) {
        std::cerr << "maxW must be 4..15\n";
        return 2;
    }

    std::vector<std::vector<Word>> words(maxW + 1);
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const Rank m1 = words[W - 1].size();
        const Rank m2 = words[W - 2].size();
        const Rank m3 = words[W - 3].size();
        const Rank dim = m1 + m2 - m3;
        const Rank nnz = 2 * m1 + m2 - 2 * m3;
        Rank max_indeg = 0;
        Rank max_c_indeg = 0;

        for (int i = 0; i <= W - 4; ++i) {
            const ReducedLayout src = make_layout(W, i, words);
            const ReducedLayout dst = make_layout(W, i + 1, words);
            if (src.size() != dim || dst.size() != dim) std::abort();

            std::vector<std::vector<Rank>> incoming(static_cast<std::size_t>(dst.size()));
            for (Rank s = 0; s < src.size(); ++s) {
                for (const auto& [k, c] : K_basis(src.key[s], W, i)) {
                    if (c != 1) std::abort();
                    const auto it = dst.rank.find(k);
                    if (it == dst.rank.end()) std::abort();
                    incoming[it->second].push_back(s);
                }
            }

            for (Rank d = 0; d < dst.size(); ++d) {
                std::vector<Rank> got;
                for (const Key& k : inverse_K(dst.key[d], W, i)) {
                    const auto it = src.rank.find(k);
                    if (it == src.rank.end())
                        fail("inverse source outside layout W=" + std::to_string(W) +
                             " i=" + std::to_string(i));
                    got.push_back(it->second);
                }
                std::sort(got.begin(), got.end());
                if (got != incoming[d])
                    fail("table-free inverse mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i) + " d=" + std::to_string(d));
                max_indeg = std::max<Rank>(max_indeg, got.size());
                if (dst.key[d].type == 'C') {
                    max_c_indeg = std::max<Rank>(max_c_indeg, got.size());
                    if (got.size() != 1)
                        fail("C destination must have one preimage W=" + std::to_string(W));
                }
            }
        }

        const Rank csr_bytes = (dim + 1 + nnz) * sizeof(Rank);
        std::cout << "W=" << W
                  << " reduced=" << dim
                  << " nnz=" << nnz
                  << " max_indeg=" << max_indeg
                  << " max_c_indeg=" << max_c_indeg
                  << " csr_kib=" << double(csr_bytes) / 1024.0
                  << " inverse_table_bytes=0"
                  << " inverse_scan=O(W)"
                  << " OK\n";
    }

    std::cout << "ALL_OK maxW=" << maxW << " table_free_inverse=1\n";
    return 0;
}
