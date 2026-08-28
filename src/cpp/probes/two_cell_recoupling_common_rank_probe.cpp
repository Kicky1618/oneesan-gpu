#pragma push_macro("main")
#undef main
#define main two_cell_recoupling_rank_probe_main_unused
#include "two_cell_recoupling_rank_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_recoupling_rank.hpp"

namespace {

oneesan::twocell::PackedKey common_pack(const Key& key) {
    oneesan::twocell::PackedKey out{};
    out.type = key.type == 'A' ? 0 : 1;
    for (int p = 0; p < static_cast<int>(key.w.size()); ++p) {
        if (key.w[p] != N) out.support |= std::uint32_t(1) << p;
        if (key.w[p] == L) out.left |= std::uint32_t(1) << p;
    }
    return out;
}

std::uint32_t common_support(const Word& w) {
    std::uint32_t support = 0;
    for (int p = 0; p < static_cast<int>(w.size()); ++p)
        if (w[p] != N) support |= std::uint32_t(1) << p;
    return support;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 6 || maxW > 14) return 2;

    const auto tables = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        Rank state_checks = 0;
        Rank component_checks = 0;

        for (int window = 0; window <= W - 5; ++window) {
            SlidingRankCodec oracle(W, window);
            for (const Key& key : q_basis(W, window + 1, words)) {
                const Rank a = oracle.rank(key);
                const Rank b = oneesan::twocell::rank_state(
                    common_pack(key), W, window, tables);
                if (a != b)
                    fail("common recoupling rank mismatch W=" + std::to_string(W) +
                         " window=" + std::to_string(window));
                ++state_checks;
            }
        }

        for (int i = 1; i <= W - 5; ++i) {
            SlidingRankCodec in_oracle(W, i - 1);
            SlidingRankCodec out_oracle(W, i);
            for (const Word& u : words[W - 2]) {
                const std::uint32_t label_support = common_support(u);
                const auto blocks = oneesan::twocell::component_blocks(
                    label_support, W, i, tables);
                const Rank in_end = blocks.input_base + tables.state_block[blocks.input_ones];
                const Rank out_end = blocks.output_base + tables.state_block[blocks.output_ones];

                const auto src = packed_direct_component_sources(pack_word(u), W, i);
                for (int q = 0; q < src.size; ++q) {
                    const Key s = unpack_key(src.value[q]);
                    const auto p = common_pack(s);
                    const Rank full = oneesan::twocell::rank_state(p, W, i - 1, tables);
                    Rank shared = 0;
                    if (p.type == 0) {
                        shared = oneesan::twocell::rank_A_with_block(
                            p.support, p.left, W, i - 1,
                            blocks.input_ones, blocks.input_base, tables);
                    } else {
                        shared = oneesan::twocell::rank_C_with_block(
                            p.support, p.left, W, i - 1,
                            blocks.input_ones, blocks.input_base, tables);
                    }
                    if (full != shared || full != in_oracle.rank(s) ||
                        full < blocks.input_base || full >= in_end)
                        fail("common shared input block W=" + std::to_string(W));

                    for (const auto& [d, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("common block nonunit");
                        const auto pd = common_pack(d);
                        const Rank dfull = oneesan::twocell::rank_state(pd, W, i, tables);
                        Rank dshared = 0;
                        if (pd.type == 0) {
                            dshared = oneesan::twocell::rank_A_with_block(
                                pd.support, pd.left, W, i,
                                blocks.output_ones, blocks.output_base, tables);
                        } else {
                            dshared = oneesan::twocell::rank_C_with_block(
                                pd.support, pd.left, W, i,
                                blocks.output_ones, blocks.output_base, tables);
                        }
                        if (dfull != dshared || dfull != out_oracle.rank(d) ||
                            dfull < blocks.output_base || dfull >= out_end)
                            fail("common shared output block W=" + std::to_string(W));
                    }
                }
                ++component_checks;
            }
        }

        std::cout << "W=" << W
                  << " state_rank_checks=" << state_checks
                  << " component_block_checks=" << component_checks
                  << " common_header=OK"
                  << " component_shared_block_base=OK"
                  << " cuda_state=2xu32"
                  << "\n";
    }

    const auto& t = tables;
    std::size_t table_bytes = sizeof(t);
    std::cout << "W=28_theory rank_table_bytes=" << table_bytes
              << " rank_table_KiB=" << double(table_bytes) / 1024.0
              << " global_permutation_bytes=0"
              << "\n";
    std::cout << "ALL_OK common_cuda_recoupling_rank=1\n";
    return 0;
}
