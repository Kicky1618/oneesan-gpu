#pragma push_macro("main")
#undef main
#define main two_cell_component_face_probe_main_unused
#include "two_cell_component_face_probe.cpp"
#pragma pop_macro("main")

namespace {

bool packed_deep_collapse(const PackedWord& u, int fixed, PackedWord& v) {
    const char a = packed_symbol(u, fixed);
    const char b = packed_symbol(u, fixed + 1);
    if ((a == R || a == L) && b == N) {
        v = packed_remove(u, fixed + 1);
        return true;
    }
    if (a == L && b == R) {
        v = packed_remove(u, fixed + 1);
        v = packed_set(v, fixed, N);
        return true;
    }
    return false;
}

SmallUnique<PackedKey, 32> packed_direct_component_sources(
    const PackedWord& label,
    int W,
    int i
) {
    SmallUnique<PackedKey, 32> out;
    const PackedKey c{'C', label};

    // Eliminated C_i(Nq) components are singletons represented by A_i(LRq).
    if (packed_symbol(label, i) == N) {
        out.insert(packed_project(c, i, W));
        return out;
    }

    // Every retained component starts with the same three source coordinates:
    // C(u), and the two A words obtained by inserting one vacancy on either
    // side of the active symbol.
    out.insert(c);
    out.insert(PackedKey{'A', packed_insert(label, i, N)});
    out.insert(PackedKey{'A', packed_insert(label, i + 1, N)});

    PackedWord collapsed;
    if (!packed_deep_collapse(label, i, collapsed)) {
        // The remaining local patterns have exactly three source coordinates.
        return out;
    }

    // For a deep component collapse RN->R, LN->L or LR->N. The unique central
    // destination is obtained by inserting two vacancies immediately before
    // the collapsed symbol. Its complete inverse consists of the fourth base
    // source plus exactly one source for each strand bordering the marked face.
    PackedWord central = packed_insert(collapsed, i, N);
    central = packed_insert(central, i, N);
    const auto pre = packed_inverse_K(PackedKey{'A', central}, W, i);
    for (int q = 0; q < pre.size; ++q) out.insert(pre.value[q]);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank checked = 0;
        Rank singleton = 0;
        Rank triple = 0;
        Rank deep = 0;
        Rank max_pairs = 0;
        Rank max_central_preimages = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const PackedWord label = pack_word(u);
                const auto direct = packed_direct_component_sources(label, W, i);
                const Key seed = project_key(Key{'C', u}, i, W);
                const auto closure = packed_component_sources(pack_key(seed), W, i);

                std::set<Key> a, b;
                for (int q = 0; q < direct.size; ++q) a.insert(unpack_key(direct.value[q]));
                for (int q = 0; q < closure.size; ++q) b.insert(unpack_key(closure.value[q]));
                if (a != b)
                    fail("direct component mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                if (direct.size == 1) {
                    ++singleton;
                } else if (direct.size == 3) {
                    ++triple;
                } else {
                    ++deep;
                    PackedWord collapsed;
                    if (!packed_deep_collapse(label, i, collapsed))
                        fail("direct deep classification");
                    PackedWord central = packed_insert(collapsed, i, N);
                    central = packed_insert(central, i, N);
                    const auto pre = packed_inverse_K(PackedKey{'A', central}, W, i);
                    max_central_preimages = std::max<Rank>(max_central_preimages, pre.size);

                    const int face = marked_face_strands(unpack_word(collapsed), i);
                    if (direct.size != 4 + face || pre.size != 1 + face)
                        fail("direct central face formula W=" + std::to_string(W));
                }

                max_pairs = std::max<Rank>(max_pairs, direct.size);
                ++checked;
            }
        }

        const Rank positions = W - 3;
        const Rank m2 = words[W - 2].size();
        const Rank m3 = words[W - 3].size();
        if (singleton != positions * m3 ||
            deep != positions * m3 ||
            triple != positions * (m2 - 2 * m3))
            fail("direct component class counts W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " checked_components=" << checked
                  << " singleton=" << singleton
                  << " triple=" << triple
                  << " deep=" << deep
                  << " max_pairs=" << max_pairs
                  << " max_central_preimages=" << max_central_preimages
                  << " reconstruction_forward_K_calls=0"
                  << " reconstruction_inverse_K_calls_deep=1"
                  << " local_insert_ops=O(1)"
                  << " face_scan=O(W)"
                  << " OK\n";
    }

    std::cout << "W=28_theory component_pairs_max=17"
              << " central_preimages_max=14"
              << " source_slots_per_warp=17"
              << "\n";
    std::cout << "ALL_OK one_scan_component_reconstruction=1\n";
    return 0;
}
