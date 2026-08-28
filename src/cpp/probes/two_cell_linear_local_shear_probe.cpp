#pragma push_macro("main")
#undef main
#define main two_cell_linear_face_formula_probe_main_unused
#include "two_cell_linear_face_formula_probe.cpp"
#pragma pop_macro("main")

#include <deque>

namespace {

PackedKey linear_local_P(PackedKey src, int i) {
    const int j = i + 1;
    if (src.type == 'C') {
        src.type = 'A';
        src.w = packed_insert(src.w, j, N);
        return src;
    }
    if (src.type != 'A') fail("linear local P type");

    const char xi = packed_symbol(src.w, i);
    const char xj = packed_symbol(src.w, j);
    const char xk = packed_symbol(src.w, j + 1);
    if (xj == N && xk != N) {
        src.type = 'C';
        src.w = packed_remove(src.w, j);
    } else if (xi != N && xj == N && xk == N) {
        src.w = packed_set(src.w, j, L);
        src.w = packed_set(src.w, j + 1, R);
    } else if (xi != N && xj == L && xk == R) {
        src.w = packed_set(src.w, i, N);
        src.w = packed_set(src.w, j, N);
        src.w = packed_set(src.w, j + 1, xi);
    }
    return src;
}

PackedKey linear_local_P_inverse(PackedKey dst, int i) {
    const int j = i + 1;
    if (dst.type == 'C') {
        dst.type = 'A';
        dst.w = packed_insert(dst.w, j, N);
        return dst;
    }
    if (dst.type != 'A') fail("linear local Pinv type");

    const char xi = packed_symbol(dst.w, i);
    const char xj = packed_symbol(dst.w, j);
    const char xk = packed_symbol(dst.w, j + 1);
    if (xi != N && xj == N) {
        dst.type = 'C';
        dst.w = packed_remove(dst.w, j);
    } else if (xi != N && xj == L && xk == R) {
        dst.w = packed_set(dst.w, j, N);
        dst.w = packed_set(dst.w, j + 1, N);
    } else if (xi == N && xj == N && xk != N) {
        dst.w = packed_set(dst.w, i, xk);
        dst.w = packed_set(dst.w, j, L);
        dst.w = packed_set(dst.w, j + 1, R);
    }
    return dst;
}

template<int CAP>
int linear_local_index(const SmallUnique<PackedKey, CAP>& a, const PackedKey& k) {
    for (int q = 0; q < a.size; ++q)
        if (a.value[q] == k) return q;
    return -1;
}

struct LinearShearStats {
    Rank components = 0;
    Rank states = 0;
    Rank correction_edges = 0;
    Rank max_pairs = 0;
    Rank max_fanin = 0;
    Rank max_fanout = 0;
};

void verify_linear_component_shear(
    const PackedWord& label,
    int W,
    int i,
    LinearShearStats& st
) {
    const auto source = packed_linear_direct_component_sources(label, W, i);
    const int n = source.size;
    if (n <= 0 || n > 32) fail("linear shear component capacity");

    SmallUnique<PackedKey, 32> matched_destinations;
    std::array<std::array<int, 32>, 32> incoming{};
    std::array<int, 32> in_count{};
    std::array<int, 32> out_count{};
    std::array<int, 32> precedence_indegree{};
    Rank extras = 0;

    for (int s = 0; s < n; ++s) {
        const PackedKey matched = linear_local_P(source.value[s], i);
        if (!(linear_local_P_inverse(matched, i) == source.value[s]))
            fail("linear shear P inverse");
        if (!matched_destinations.insert(matched))
            fail("linear shear P collision");

        const auto image = packed_K(source.value[s], W, i);
        bool saw_matching = false;
        for (int q = 0; q < image.size; ++q) {
            const PackedKey d = image.value[q];
            if (d == matched) {
                saw_matching = true;
                continue;
            }
            const PackedKey owner_key = linear_local_P_inverse(d, i);
            const int owner = linear_local_index(source, owner_key);
            if (owner < 0) fail("linear shear edge escapes component");
            if (in_count[owner] >= 32) fail("linear shear fanin capacity");
            incoming[owner][in_count[owner]++] = s;
            ++out_count[s];
            ++precedence_indegree[s]; // owner must consume s before s is changed.
            ++extras;
        }
        if (!saw_matching) fail("linear shear local P not operator edge");
    }
    if (matched_destinations.size != n) fail("linear shear P not bijective");
    if (extras + 1 != Rank(n)) fail("linear shear correction is not tree-sized");

    std::deque<int> ready;
    for (int v = 0; v < n; ++v)
        if (!precedence_indegree[v]) ready.push_back(v);
    std::array<int, 32> order{};
    int order_n = 0;
    while (!ready.empty()) {
        const int target = ready.front();
        ready.pop_front();
        order[order_n++] = target;
        for (int q = 0; q < in_count[target]; ++q) {
            const int src = incoming[target][q];
            if (precedence_indegree[src] <= 0)
                fail("linear shear precedence underflow");
            if (--precedence_indegree[src] == 0) ready.push_back(src);
        }
    }
    if (order_n != n) fail("linear shear correction cycle");

    std::array<std::uint64_t, 32> input{};
    std::array<std::uint64_t, 32> direct{};
    std::array<std::uint64_t, 32> work{};
    for (int s = 0; s < n; ++s) {
        input[s] = 1 + ((std::uint64_t(s) * 0x9e3779b97f4a7c15ULL) ^
                        (std::uint64_t(W) << 40) ^ (std::uint64_t(i) << 32));
        work[s] = input[s];
        const auto image = packed_K(source.value[s], W, i);
        for (int q = 0; q < image.size; ++q) {
            const int owner = linear_local_index(
                source, linear_local_P_inverse(image.value[q], i));
            if (owner < 0) fail("linear shear direct owner");
            direct[owner] += input[s];
        }
    }

    for (int q = 0; q < order_n; ++q) {
        const int target = order[q];
        for (int z = 0; z < in_count[target]; ++z)
            work[target] += work[incoming[target][z]];
    }
    for (int v = 0; v < n; ++v)
        if (work[v] != direct[v]) fail("linear shear in-place mismatch");

    ++st.components;
    st.states += n;
    st.correction_edges += extras;
    st.max_pairs = std::max<Rank>(st.max_pairs, n);
    for (int v = 0; v < n; ++v) {
        st.max_fanin = std::max<Rank>(st.max_fanin, in_count[v]);
        st.max_fanout = std::max<Rank>(st.max_fanout, out_count[v]);
    }
}

Rank count_one_defect_shear(int W) {
    std::vector<Rank> cur(static_cast<std::size_t>(W + 2));
    std::vector<Rank> next(static_cast<std::size_t>(W + 2));
    cur[1] = 1;
    for (int p = 0; p < W; ++p) {
        std::fill(next.begin(), next.end(), 0);
        for (int h = 0; h <= W; ++h) if (cur[h]) {
            next[h] += cur[h];
            next[h + 1] += cur[h];
            if (h) next[h - 1] += cur[h];
        }
        cur.swap(next);
    }
    return cur[0];
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        LinearShearStats reference{};
        bool first = true;
        for (int i = 0; i <= W - 4; ++i) {
            LinearShearStats st;
            for (const Word& u : words[W - 2])
                verify_linear_component_shear(pack_word(u), W, i, st);

            const Rank m2 = words[W - 2].size();
            const Rank m3 = words[W - 3].size();
            const Rank reduced = words[W - 1].size() + m2 - m3;
            if (st.components != m2 || st.states != reduced ||
                st.correction_edges != reduced - m2)
                fail("linear shear aggregate formula");
            if (first) {
                reference = st;
                first = false;
            } else if (st.components != reference.components ||
                       st.states != reference.states ||
                       st.correction_edges != reference.correction_edges ||
                       st.max_pairs != reference.max_pairs ||
                       st.max_fanin != reference.max_fanin ||
                       st.max_fanout != reference.max_fanout) {
                fail("linear shear stats depend on position");
            }
        }

        std::cout << "W=" << W
                  << " components=" << reference.components
                  << " states=" << reference.states
                  << " correction_adds=" << reference.correction_edges
                  << " max_pairs=" << reference.max_pairs
                  << " max_fanin=" << reference.max_fanin
                  << " max_fanout=" << reference.max_fanout
                  << " component_BFS=0 mate_array=0 partner_calls=0"
                  << " matching_leaf_peeling=0 matching_table=0"
                  << " local_P=1 linear_face=1 inplace_shear=1 OK\n";
    }

    constexpr int W = 28;
    const Rank m27 = count_one_defect_shear(27);
    const Rank m26 = count_one_defect_shear(26);
    const Rank m25 = count_one_defect_shear(25);
    const Rank reduced = m27 + m26 - m25;
    std::cout << "W=28_theory"
              << " components=" << m26
              << " states=" << reduced
              << " correction_adds=" << (reduced - m26)
              << " adds_per_state=" << (double(reduced - m26) / double(reduced))
              << " max_pairs=17 warp_local=1"
              << '\n';
    std::cout << "ALL_OK linear_local_forest_shear=1\n";
    return 0;
}
