#pragma push_macro("main")
#undef main
#define main two_cell_direct_component_probe_main_unused
#include "two_cell_direct_component_probe.cpp"
#pragma pop_macro("main")

namespace {

SmallUnique<PackedKey, 32> packed_central_face_preimages(
    const PackedWord& collapsed,
    int W,
    int i
) {
    // B0 = A(v[0..i) NN v[i..]).  At the destination quotient mark j=i+1,
    // B0[j] is always N.  Therefore inverse_project has exactly one branch:
    // B0 itself.  Skip generic inverse_K and directly perform its only
    // inverse-R face scan followed by inverse-E.
    PackedWord central = packed_insert(collapsed, i, N);
    central = packed_insert(central, i, N);
    const PackedKey raw{'A', central};
    const auto full = packed_inverse_R_raw(raw, i + 1, W);

    SmallUnique<PackedKey, 32> out;
    for (int q = 0; q < full.size; ++q) {
        PackedKey src;
        if (packed_inverse_E(full.value[q], i, src) &&
            packed_in_source_layout(src, W, i))
            out.insert(src);
    }
    return out;
}

SmallUnique<PackedKey, 32> packed_direct_component_sources_face_only(
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
    const auto pre = packed_central_face_preimages(collapsed, W, i);
    for (int q = 0; q < pre.size; ++q) out.insert(pre.value[q]);
    return out;
}

Rank count_one_defect_face(int W) {
    std::vector<Rank> cur(static_cast<std::size_t>(W + 2));
    std::vector<Rank> nxt(static_cast<std::size_t>(W + 2));
    cur[1] = 1;
    for (int p = 0; p < W; ++p) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= W; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            nxt[h + 1] += cur[h];
            if (h) nxt[h - 1] += cur[h];
        }
        cur.swap(nxt);
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
        Rank face_scans = 0;
        Rank max_face_preimages = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const PackedWord label = pack_word(u);
                const auto got = packed_direct_component_sources_face_only(label, W, i);
                const auto oracle = packed_direct_component_sources(label, W, i);

                std::set<Key> a, b;
                for (int q = 0; q < got.size; ++q) a.insert(unpack_key(got.value[q]));
                for (int q = 0; q < oracle.size; ++q) b.insert(unpack_key(oracle.value[q]));
                if (a != b)
                    fail("face-only component mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                PackedWord collapsed;
                if (packed_deep_collapse(label, i, collapsed)) {
                    ++deep;
                    ++face_scans;
                    const auto specialized = packed_central_face_preimages(collapsed, W, i);
                    PackedWord central = packed_insert(collapsed, i, N);
                    central = packed_insert(central, i, N);
                    const auto generic = packed_inverse_K(PackedKey{'A', central}, W, i);

                    std::set<Key> x, y;
                    for (int q = 0; q < specialized.size; ++q)
                        x.insert(unpack_key(specialized.value[q]));
                    for (int q = 0; q < generic.size; ++q)
                        y.insert(unpack_key(generic.value[q]));
                    if (x != y)
                        fail("specialized central preimage mismatch");
                    const int face = marked_face_strands(unpack_word(collapsed), i);
                    if (specialized.size != 1 + face)
                        fail("specialized face preimage count");
                    max_face_preimages = std::max<Rank>(max_face_preimages, specialized.size);
                }
                ++checked;
            }
        }

        const Rank positions = W - 3;
        const Rank m3 = words[W - 3].size();
        if (deep != positions * m3 || face_scans != deep)
            fail("face-only deep count");
        std::cout << "W=" << W
                  << " checked=" << checked
                  << " deep=" << deep
                  << " face_scans=" << face_scans
                  << " max_face_preimages=" << max_face_preimages
                  << " reconstruction_K_calls=0"
                  << " reconstruction_inverse_K_calls=0"
                  << " inverse_project_calls=0"
                  << " inverse_R_face_scans_per_deep=1"
                  << " exact=1 OK\n";
    }

    constexpr int W = 28;
    const Rank m26 = count_one_defect_face(26);
    const Rank m25 = count_one_defect_face(25);
    std::cout << "W=28_theory"
              << " components=" << m26
              << " deep=" << m25
              << " face_scans=" << m25
              << " face_scan_fraction=" << (double(m25) / double(m26))
              << " max_face_preimages=14"
              << " expansion_rounds=0 generic_inverse_K=0"
              << '\n';
    std::cout << "ALL_OK specialized_face_component_reconstruction=1\n";
    return 0;
}
