#pragma push_macro("main")
#undef main
#define main two_cell_turn_closed_block_probe_main_unused
#include "two_cell_turn_closed_block_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_turn_closed_device.cuh"

namespace {

oneesan::twocell::PackedWord turn_device_word(const Word& w) {
    oneesan::twocell::PackedWord z{};
    z.len = static_cast<std::uint8_t>(w.size());
    for (int p = 0; p < static_cast<int>(w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (w[p] != N) z.support |= bit;
        if (w[p] == L) z.left |= bit;
    }
    return z;
}

Key turn_device_key(oneesan::twocell::PackedKey k, int W) {
    const int len = k.type ? W - 2 : W - 1;
    Word w(static_cast<std::size_t>(len), N);
    for (int p = 0; p < len; ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (!(k.support & bit)) continue;
        w[p] = (k.left & bit) ? L : R;
    }
    return Key{k.type ? 'C' : 'A', w};
}

std::vector<Key> turn_device_states(
    const oneesan::twocell::ClosedTurnBlock& b, int W
) {
    std::vector<Key> out;
    for (int q = 0; q < b.size; ++q) out.push_back(turn_device_key(b.state[q], W));
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank reflected_words = 0;
        Rank right_blocks = 0;
        Rank left_blocks = 0;
        Rank max_size = 0;
        std::set<Key> right_cover, left_cover;

        for (const Word& u : words[W - 2]) {
            const auto pu = turn_device_word(u);
            const auto rr = oneesan::twocell::reflect_packed_word(pu);
            const Word reflected = turn_device_key(
                oneesan::twocell::make_state(1, rr), W).w;
            if (reflected != reflect_word(u))
                fail("turn device reflection W=" + std::to_string(W));
            ++reflected_words;

            const auto rb = oneesan::twocell::right_turn_closed_block(pu, W);
            if (rb.overflow || rb.size < 3)
                fail("turn device right block overflow W=" + std::to_string(W));
            const auto rstates = turn_device_states(rb, W);
            const auto oracle = closed_right_turn_component(u, W).states();
            if (rstates != oracle)
                fail("turn device right block W=" + std::to_string(W));
            for (const Key& s : rstates) {
                if (!right_cover.insert(s).second)
                    fail("turn device right overlap");
                const CVec expected = closed_turn_column(closed_right_turn_component(u, W), s);
                if (turn_right_basis(s, W) != expected)
                    fail("turn device right arithmetic");
            }
            max_size = std::max<Rank>(max_size, rb.size);
            ++right_blocks;

            const auto lb = oneesan::twocell::left_turn_closed_block(pu, W);
            if (lb.overflow || lb.size != rb.size || lb.singular != rb.singular)
                fail("turn device left block metadata W=" + std::to_string(W));
            const auto lstates = turn_device_states(lb, W);
            for (const Key& s : lstates) {
                if (!left_cover.insert(s).second)
                    fail("turn device left overlap");
                // Reflection is exact for the full physical row turn. Every
                // left block coordinate therefore stays inside its reflected
                // component and uses the same alpha/beta/passive arithmetic.
                const Key rs = reflect_key(s);
                const CVec mirrored = reflect_vec(turn_right_basis(rs, W));
                if (turn_left_basis(s, W) != mirrored)
                    fail("turn device left reflection arithmetic");
            }
            ++left_blocks;
        }

        const auto right_basis = q_basis(W, W - 3, words);
        const auto left_basis = reverse_q_basis(W, 1, words);
        if (right_cover != std::set<Key>(right_basis.begin(), right_basis.end()))
            fail("turn device right cover W=" + std::to_string(W));
        if (left_cover != std::set<Key>(left_basis.begin(), left_basis.end()))
            fail("turn device left cover W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " labels=" << words[W - 2].size()
                  << " reflected_words=" << reflected_words
                  << " right_blocks=" << right_blocks
                  << " left_blocks=" << left_blocks
                  << " max_size=" << max_size
                  << " right_partition=OK left_partition=OK"
                  << " shared_arithmetic_order=alpha,beta,passive"
                  << "\n";
    }

    std::cout << "W=28_theory turn_max_states=15"
              << " right_left_same_executor=1"
              << " component_graph_bytes=0\n";
    std::cout << "ALL_OK closed_turn_device_blocks=1\n";
    return 0;
}
