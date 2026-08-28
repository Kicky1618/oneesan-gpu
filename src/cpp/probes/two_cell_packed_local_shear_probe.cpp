#pragma push_macro("main")
#undef main
#define main two_cell_packed_component_probe_main_unused
#include "two_cell_packed_component_probe.cpp"
#pragma pop_macro("main")

#include <deque>

namespace {

PackedKey packed_local_P(PackedKey src, int i) {
    const int j = i + 1;
    if (src.type == 'C') {
        src.type = 'A';
        src.w = packed_insert(src.w, j, N);
        return src;
    }
    if (src.type != 'A') fail("packed local P type");

    const char xi = packed_symbol(src.w, i);
    const char xj = packed_symbol(src.w, j);
    const char xk = packed_symbol(src.w, j + 1);
    if (xj == N && xk != N) {
        src.type = 'C';
        src.w = packed_remove(src.w, j);
        return src;
    }
    if (xi != N && xj == N && xk == N) {
        src.w = packed_set(src.w, j, L);
        src.w = packed_set(src.w, j + 1, R);
        return src;
    }
    if (xi != N && xj == L && xk == R) {
        src.w = packed_set(src.w, i, N);
        src.w = packed_set(src.w, j, N);
        src.w = packed_set(src.w, j + 1, xi);
        return src;
    }
    return src;
}

PackedKey packed_local_P_inverse(PackedKey dst, int i) {
    const int j = i + 1;
    if (dst.type == 'C') {
        dst.type = 'A';
        dst.w = packed_insert(dst.w, j, N);
        return dst;
    }
    if (dst.type != 'A') fail("packed local Pinv type");

    const char xi = packed_symbol(dst.w, i);
    const char xj = packed_symbol(dst.w, j);
    const char xk = packed_symbol(dst.w, j + 1);
    if (xi != N && xj == N) {
        dst.type = 'C';
        dst.w = packed_remove(dst.w, j);
        return dst;
    }
    if (xi != N && xj == L && xk == R) {
        dst.w = packed_set(dst.w, j, N);
        dst.w = packed_set(dst.w, j + 1, N);
        return dst;
    }
    if (xi == N && xj == N && xk != N) {
        dst.w = packed_set(dst.w, i, xk);
        dst.w = packed_set(dst.w, j, L);
        dst.w = packed_set(dst.w, j + 1, R);
        return dst;
    }
    return dst;
}

template<int CAP>
int local_index(const SmallUnique<PackedKey, CAP>& v, const PackedKey& x) {
    for (int i = 0; i < v.size; ++i)
        if (v.value[i] == x) return i;
    return -1;
}

template<int CAP>
bool contains_key(const SmallUnique<PackedKey, CAP>& v, const PackedKey& x) {
    return local_index(v, x) >= 0;
}

struct PackedLocalStats {
    Rank components = 0;
    Rank pairs = 0;
    Rank correction_edges = 0;
    Rank max_pairs = 0;
    Rank max_correction_in = 0;
    Rank max_correction_out = 0;
};

void verify_component_local_shear(
    const Word& label,
    int W,
    int i,
    PackedLocalStats& stats
) {
    const Key seed_key = project_key(Key{'C', label}, i, W);
    const auto source = packed_component_sources(pack_key(seed_key), W, i);
    if (source.size <= 0 || source.size > 32) fail("packed local source capacity");

    SmallUnique<PackedKey, 32> matched_dst;
    std::array<std::array<int, 2>, 32> incoming{};
    std::array<std::array<int, 2>, 32> outgoing{};
    std::array<int, 32> indeg{};
    std::array<int, 32> in_count{};
    std::array<int, 32> out_count{};
    Rank extras = 0;

    for (int s = 0; s < source.size; ++s) {
        const PackedKey p = packed_local_P(source.value[s], i);
        if (!(packed_local_P_inverse(p, i) == source.value[s]))
            fail("packed local P inverse mismatch");
        const auto image = packed_K(source.value[s], W, i);
        if (!contains_key(image, p)) fail("packed local P is not K edge");
        if (!matched_dst.insert(p)) fail("packed local P destination collision");

        for (int q = 0; q < image.size; ++q) {
            const PackedKey d = image.value[q];
            if (d == p) continue;
            const PackedKey owner = packed_local_P_inverse(d, i);
            const int t = local_index(source, owner);
            if (t < 0) fail("packed correction owner outside component");
            if (out_count[s] >= 2 || in_count[t] >= 2)
                fail("packed correction degree capacity");
            outgoing[s][out_count[s]++] = t;
            incoming[t][in_count[t]++] = s;
            ++indeg[s]; // target t must execute before source s.
            ++extras;
        }
    }
    if (matched_dst.size != source.size) fail("packed local P not perfect matching");
    if (extras + 1 != Rank(source.size)) fail("packed correction graph not tree-sized");

    std::deque<int> ready;
    for (int v = 0; v < source.size; ++v)
        if (!indeg[v]) ready.push_back(v);
    std::array<int, 32> order{};
    int order_n = 0;
    while (!ready.empty()) {
        const int target = ready.front();
        ready.pop_front();
        order[order_n++] = target;
        for (int q = 0; q < in_count[target]; ++q) {
            const int src = incoming[target][q];
            if (indeg[src] <= 0) fail("packed local precedence underflow");
            if (--indeg[src] == 0) ready.push_back(src);
        }
    }
    if (order_n != source.size) fail("packed local correction cycle");

    std::array<std::uint64_t, 32> input{};
    std::array<std::uint64_t, 32> work{};
    std::array<std::uint64_t, 32> reference{};
    for (int s = 0; s < source.size; ++s) {
        input[s] = 1 + ((std::uint64_t(s) * 0x9e3779b97f4a7c15ULL) ^
                        (std::uint64_t(W) << 40) ^ (std::uint64_t(i) << 32));
        work[s] = input[s];
        const auto image = packed_K(source.value[s], W, i);
        for (int q = 0; q < image.size; ++q) {
            const int owner = local_index(source, packed_local_P_inverse(image.value[q], i));
            if (owner < 0) fail("packed reference owner outside component");
            reference[owner] += input[s];
        }
    }

    for (int q = 0; q < order_n; ++q) {
        const int target = order[q];
        for (int z = 0; z < in_count[target]; ++z)
            work[target] += work[incoming[target][z]];
    }
    for (int v = 0; v < source.size; ++v)
        if (work[v] != reference[v]) fail("packed local in-place shear mismatch");

    ++stats.components;
    stats.pairs += source.size;
    stats.correction_edges += extras;
    stats.max_pairs = std::max<Rank>(stats.max_pairs, source.size);
    for (int v = 0; v < source.size; ++v) {
        stats.max_correction_in = std::max<Rank>(stats.max_correction_in, in_count[v]);
        stats.max_correction_out = std::max<Rank>(stats.max_correction_out, out_count[v]);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        PackedLocalStats reference{};
        bool first = true;
        for (int i = 0; i <= W - 4; ++i) {
            PackedLocalStats st;
            for (const Word& label : words[W - 2])
                verify_component_local_shear(label, W, i, st);
            const Rank reduced = words[W - 1].size() + words[W - 2].size() - words[W - 3].size();
            if (st.components != words[W - 2].size() || st.pairs != reduced ||
                st.correction_edges != reduced - st.components)
                fail("packed local global count mismatch");
            if (first) {
                reference = st;
                first = false;
            } else if (st.components != reference.components || st.pairs != reference.pairs ||
                       st.correction_edges != reference.correction_edges ||
                       st.max_pairs != reference.max_pairs ||
                       st.max_correction_in != reference.max_correction_in ||
                       st.max_correction_out != reference.max_correction_out) {
                fail("packed local stats depend on position");
            }
        }
        std::cout << "W=" << W
                  << " components=" << reference.components
                  << " reduced=" << reference.pairs
                  << " correction_edges=" << reference.correction_edges
                  << " max_pairs=" << reference.max_pairs
                  << " max_correction_in=" << reference.max_correction_in
                  << " max_correction_out=" << reference.max_correction_out
                  << " leaf_peeling=0 matching_table=0 local_P=1 inplace_shear=1 OK\n";
    }
    std::cout << "ALL_OK packed_local_shear=1\n";
    return 0;
}
