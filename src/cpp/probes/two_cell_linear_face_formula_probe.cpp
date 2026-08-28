#pragma push_macro("main")
#undef main
#define main two_cell_direct_face_preimage_probe_main_unused
#include "two_cell_direct_face_preimage_probe.cpp"
#pragma pop_macro("main")

namespace {

PackedWord packed_insert_pair(PackedWord w, int pos, char a, char b) {
    w = packed_insert(w, pos, a);
    return packed_insert(w, pos + 1, b);
}

SmallUnique<PackedKey, 32> packed_linear_face_preimages(
    const PackedWord& v,
    int i
) {
    const int n = v.len;
    if (i < 0 || i >= n) fail("linear face mark");

    std::array<int, 33> h{};
    h[0] = 1;
    int root = -1;
    for (int pos = 0; pos < n; ++pos) {
        h[pos + 1] = h[pos];
        const char c = packed_symbol(v, pos);
        if (c == L) ++h[pos + 1];
        else if (c == R) {
            --h[pos + 1];
            if (h[pos + 1] == 0 && root < 0) root = pos;
        }
    }
    if (h[n] != 0 || root < 0) fail("linear face invalid one-defect word");

    const int level = h[i];
    int left = i;
    while (left > 0 && h[left - 1] >= level) --left;
    int right = i;
    while (right < n && h[right + 1] >= level) ++right;

    SmallUnique<PackedKey, 32> out;
    out.insert(PackedKey{'A', packed_insert_pair(v, i, N, N)});

    // Top-level excursions of this face are disjoint.  Once an L at face
    // level is found, its matching R is simply the first position whose next
    // height returns to `level`.  Jump to q+1, so the complete face is scanned
    // once: no stack, mate array, or repeated partner search.
    int p = left;
    while (p < right) {
        if (h[p] != level || packed_symbol(v, p) != L) {
            ++p;
            continue;
        }
        int q = p + 1;
        while (q < right && h[q + 1] != level) ++q;
        if (q >= right || packed_symbol(v, q) != R)
            fail("linear face excursion endpoint");

        PackedWord w = v;
        if (q < i) {
            // Strand lies to the left of the marked interval: cut its right
            // endpoint and expose RR at the two inserted sites.
            w = packed_set(w, q, L);
            out.insert(PackedKey{'A', packed_insert_pair(w, i, R, R)});
        } else if (p >= i) {
            // Symmetric right-hand strand.
            w = packed_set(w, p, R);
            out.insert(PackedKey{'A', packed_insert_pair(w, i, L, L)});
        } else {
            // The strand itself surrounds the marked interval.
            out.insert(PackedKey{'A', packed_insert_pair(v, i, R, L)});
        }
        p = q + 1;
    }

    // If the face is nested, the immediately enclosing arc is another visible
    // boundary strand.  Height maximality makes these endpoint tests
    // equivalent to mate[left-1] == right; no partner lookup is needed.
    if (left > 0 && right < n &&
        packed_symbol(v, left - 1) == L && packed_symbol(v, right) == R) {
        out.insert(PackedKey{'A', packed_insert_pair(v, i, R, L)});
    }

    // The distinguished root is the remaining boundary strand on the outer
    // face (or on the prefix face whose right boundary is the root).
    if (level == 0 || (left == 0 && right < n && root == right)) {
        if (root < i) {
            PackedWord w = packed_set(v, root, L);
            out.insert(PackedKey{'A', packed_insert_pair(w, i, R, R)});
        } else {
            out.insert(PackedKey{'A', packed_insert_pair(v, i, R, L)});
        }
    }
    return out;
}

SmallUnique<PackedKey, 32> packed_linear_direct_component_sources(
    const PackedWord& label,
    int W,
    int i
) {
    SmallUnique<PackedKey, 32> out;
    const PackedKey c{'C', label};
    if (packed_symbol(label, i) == N) {
        out.insert(packed_project(c, i, W));
        return out;
    }

    out.insert(c);
    out.insert(PackedKey{'A', packed_insert(label, i, N)});
    out.insert(PackedKey{'A', packed_insert(label, i + 1, N)});

    PackedWord collapsed;
    if (!packed_deep_collapse(label, i, collapsed)) return out;
    const auto face = packed_linear_face_preimages(collapsed, i);
    for (int q = 0; q < face.size; ++q) out.insert(face.value[q]);
    return out;
}

Rank one_defect_count_linear(int W) {
    std::vector<Rank> cur(static_cast<std::size_t>(W + 2));
    std::vector<Rank> next(static_cast<std::size_t>(W + 2));
    cur[1] = 1;
    for (int pos = 0; pos < W; ++pos) {
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
        Rank checked = 0;
        Rank deep = 0;
        Rank max_face_preimages = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const PackedWord label = pack_word(u);
                const auto got = packed_linear_direct_component_sources(label, W, i);
                const auto oracle = packed_direct_component_sources_face_only(label, W, i);

                std::set<Key> a, b;
                for (int q = 0; q < got.size; ++q) a.insert(unpack_key(got.value[q]));
                for (int q = 0; q < oracle.size; ++q) b.insert(unpack_key(oracle.value[q]));
                if (a != b)
                    fail("linear direct component mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                PackedWord collapsed;
                if (packed_deep_collapse(label, i, collapsed)) {
                    ++deep;
                    const auto direct = packed_linear_face_preimages(collapsed, i);
                    const auto specialized = packed_central_face_preimages(collapsed, W, i);
                    std::set<Key> x, y;
                    for (int q = 0; q < direct.size; ++q) x.insert(unpack_key(direct.value[q]));
                    for (int q = 0; q < specialized.size; ++q) y.insert(unpack_key(specialized.value[q]));
                    if (x != y) fail("linear marked-face preimage mismatch");
                    const int strands = marked_face_strands(unpack_word(collapsed), i);
                    if (direct.size != 1 + strands)
                        fail("linear marked-face count mismatch");
                    max_face_preimages = std::max<Rank>(max_face_preimages, direct.size);
                }
                ++checked;
            }
        }
        const Rank positions = W - 3;
        const Rank m3 = words[W - 3].size();
        if (deep != positions * m3) fail("linear deep count mismatch");
        std::cout << "W=" << W
                  << " checked=" << checked
                  << " deep=" << deep
                  << " max_face_preimages=" << max_face_preimages
                  << " graph_expansion_rounds=0"
                  << " K_calls=0 inverse_K_calls=0 inverse_R_calls=0"
                  << " mate_array=0 partner_calls=0 stack=0"
                  << " marked_face_passes_per_deep=1"
                  << " exact=1 OK\n";
    }

    constexpr int W = 28;
    const Rank m26 = one_defect_count_linear(26);
    const Rank m25 = one_defect_count_linear(25);
    std::cout << "W=28_theory"
              << " components=" << m26
              << " deep=" << m25
              << " linear_face_passes=" << m25
              << " face_scan_fraction=" << (double(m25) / double(m26))
              << " max_face_preimages=14"
              << " max_component_pairs=17"
              << '\n';
    std::cout << "ALL_OK linear_direct_component_formula=1\n";
    return 0;
}
