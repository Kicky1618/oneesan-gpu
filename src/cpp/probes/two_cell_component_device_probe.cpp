#pragma push_macro("main")
#undef main
#define main two_cell_direct_component_probe_main_unused
#include "two_cell_direct_component_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_component_device.cuh"

namespace {

oneesan::twocell::PackedWord device_word(const Word& w) {
    oneesan::twocell::PackedWord z{};
    z.len = static_cast<std::uint8_t>(w.size());
    for (int p = 0; p < static_cast<int>(w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (w[p] != N) z.support |= bit;
        if (w[p] == L) z.left |= bit;
    }
    return z;
}

oneesan::twocell::PackedKey device_key(const Key& k) {
    const auto w = device_word(k.w);
    return oneesan::twocell::PackedKey{w.support, w.left,
                                       static_cast<std::uint8_t>(k.type == 'C')};
}

Key oracle_key(oneesan::twocell::PackedKey k, int W) {
    const int len = k.type ? W - 2 : W - 1;
    Word w(static_cast<std::size_t>(len), N);
    for (int p = 0; p < len; ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (!(k.support & bit)) continue;
        w[p] = (k.left & bit) ? L : R;
    }
    return Key{k.type ? 'C' : 'A', w};
}

template <class List>
std::set<Key> device_set(const List& xs, int W) {
    std::set<Key> out;
    for (int q = 0; q < xs.size; ++q) out.insert(oracle_key(xs.value[q], W));
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank k_checks = 0;
        Rank inverse_checks = 0;
        Rank component_checks = 0;
        Rank max_sources = 0;
        Rank max_inverse = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Key& src : q_basis(W, i, words)) {
                const auto got = oneesan::twocell::K_step(device_key(src), W, i);
                if (got.overflow) fail("device K overflow");
                CVec expected = K_basis(src, W, i);
                CVec actual;
                for (const Key& k : device_set(got, W)) add(actual, k);
                if (actual != expected)
                    fail("device K mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                ++k_checks;
            }

            for (const Key& dst : q_basis(W, i + 1, words)) {
                const auto got = oneesan::twocell::inverse_K(device_key(dst), W, i);
                if (got.overflow) fail("device inverse overflow");
                const auto expected = inverse_K(dst, W, i);
                const auto actual = device_set(got, W);
                if (actual != expected)
                    fail("device inverse mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                max_inverse = std::max<Rank>(max_inverse, got.size);
                ++inverse_checks;
            }

            for (const Word& u : words[W - 2]) {
                const auto got = oneesan::twocell::direct_component_sources(
                    device_word(u), W, i);
                if (got.overflow) fail("device component overflow");
                const auto expected = packed_direct_component_sources(pack_word(u), W, i);
                std::set<Key> oracle;
                for (int q = 0; q < expected.size; ++q)
                    oracle.insert(unpack_key(expected.value[q]));
                if (device_set(got, W) != oracle)
                    fail("device direct component mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                max_sources = std::max<Rank>(max_sources, got.size);
                ++component_checks;
            }
        }

        std::cout << "W=" << W
                  << " K_checks=" << k_checks
                  << " inverse_checks=" << inverse_checks
                  << " component_checks=" << component_checks
                  << " max_inverse=" << max_inverse
                  << " max_component_sources=" << max_sources
                  << " graph_search=0"
                  << " host_device_algebra=OK"
                  << "\n";
    }

    std::cout << "W=28_theory max_component_sources=17"
              << " warp_slots=18"
              << " packed_state_payload_bytes=9"
              << "\n";
    std::cout << "ALL_OK two_cell_component_device_algebra=1\n";
    return 0;
}
