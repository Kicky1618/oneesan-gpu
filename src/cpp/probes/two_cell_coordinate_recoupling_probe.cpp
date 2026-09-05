#pragma push_macro("main")
#undef main
#define main two_cell_component_primitive_signature_probe_main_unused
#include "two_cell_component_primitive_signature_probe.cpp"
#pragma pop_macro("main")

namespace {

Key recouple_coordinate(Key src, int i) {
    if (src.type == 'A') return src;
    if (src.type != 'C') fail("recouple key type");
    if (src.w[i + 1] == N && src.w[i] != N)
        std::swap(src.w[i], src.w[i + 1]);
    return src;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank components = 0;
        Rank identical_components = 0;
        Rank one_c_swap_components = 0;
        Rank coordinates = 0;
        Rank identity_coordinates = 0;
        Rank swapped_coordinates = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto packed_sources = packed_direct_component_sources(pack_word(u), W, i);
                std::set<Key> src;
                std::set<Key> dst;
                std::set<Key> predicted;
                for (int q = 0; q < packed_sources.size; ++q) {
                    const Key s = unpack_key(packed_sources.value[q]);
                    src.insert(s);
                    const Key d = recouple_coordinate(s, i);
                    predicted.insert(d);
                    ++coordinates;
                    if (d == s) ++identity_coordinates;
                    else ++swapped_coordinates;
                    if (primitive_signature(d) != primitive_signature(s))
                        fail("recouple changed primitive signature W=" + std::to_string(W));
                    for (const auto& [z, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("recouple nonunit edge");
                        dst.insert(z);
                    }
                }
                if (predicted != dst)
                    fail("coordinate recoupling mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                if (src.size() != dst.size() || predicted.size() != src.size())
                    fail("coordinate recoupling bijection");

                const auto old_only = [&]() {
                    std::set<Key> z;
                    std::set_difference(src.begin(), src.end(), dst.begin(), dst.end(),
                                        std::inserter(z, z.end()));
                    return z;
                }();
                const auto new_only = [&]() {
                    std::set<Key> z;
                    std::set_difference(dst.begin(), dst.end(), src.begin(), src.end(),
                                        std::inserter(z, z.end()));
                    return z;
                }();
                if (old_only.empty()) {
                    if (!new_only.empty()) fail("recouple asymmetric difference");
                    ++identical_components;
                } else {
                    if (old_only.size() != 1 || new_only.size() != 1 ||
                        old_only.begin()->type != 'C' || new_only.begin()->type != 'C')
                        fail("recouple not one-C replacement");
                    if (!(u[i] != N && u[i + 1] == N))
                        fail("recouple swap classification");
                    Key want{'C', u};
                    std::swap(want.w[i], want.w[i + 1]);
                    if (*old_only.begin() != Key{'C', u} || *new_only.begin() != want)
                        fail("recouple exact C swap");
                    ++one_c_swap_components;
                }
                ++components;
            }
        }

        std::cout << "W=" << W
                  << " components=" << components
                  << " identical_components=" << identical_components
                  << " one_c_swap_components=" << one_c_swap_components
                  << " coordinates=" << coordinates
                  << " identity_coordinates=" << identity_coordinates
                  << " swapped_coordinates=" << swapped_coordinates
                  << " A_recoupling=identity"
                  << " C_recoupling=xN_to_Nx"
                  << " primitive_signature_preserved=1"
                  << " OK\n";
    }

    std::cout << "ALL_OK one_C_coordinate_recoupling=1\n";
    return 0;
}
