#pragma push_macro("main")
#undef main
#define main two_cell_fusion_rank_probe_main_unused
#include "two_cell_fusion_rank_probe.cpp"
#pragma pop_macro("main")

namespace {

using FValue = std::uint64_t;
constexpr FValue kFusionMod = 1000000007ULL;

FValue fadd(FValue a, FValue b) { return (a + b) % kFusionMod; }

Rank global_stationary_rank(
    const Key& k,
    int W,
    int active,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    return oneesan::twocell::stationary_rank(
        stationary_pack(k), W, active, rt, st);
}

std::vector<FValue> ordinary_two_pass(
    const std::vector<FValue>& input,
    int W,
    int start,
    const std::vector<std::vector<Word>>& words,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    std::vector<FValue> cur = input;
    std::vector<FValue> next(input.size());
    for (int active = start; active < start + 2; ++active) {
        std::fill(next.begin(), next.end(), 0);
        for (const Key& s : q_basis(W, active, words)) {
            const Rank sr = global_stationary_rank(s, W, active, rt, st);
            const FValue x = cur[static_cast<std::size_t>(sr)];
            for (const auto& [d, c] : K_basis(s, W, active)) {
                if (c != 1) fail("fusion2 ordinary nonunit");
                const Rank dr = global_stationary_rank(d, W, active + 1, rt, st);
                next[static_cast<std::size_t>(dr)] =
                    fadd(next[static_cast<std::size_t>(dr)], x);
            }
        }
        cur.swap(next);
    }
    return cur;
}

struct BlockStats {
    Rank blocks = 0;
    Rank value_loads = 0;
    Rank value_stores = 0;
    Rank local_step_reads = 0;
    Rank local_step_writes = 0;
    Rank max_block = 0;
};

std::vector<FValue> fused_two_pass(
    const std::vector<FValue>& input,
    int W,
    int start,
    const std::vector<std::vector<Word>>& words,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st,
    BlockStats& stats
) {
    constexpr int steps = 2;
    const int outer_bits = W - steps - 3;
    std::vector<FValue> output(input.size());
    std::vector<std::uint8_t> globally_written(input.size());

    // Materialize each Q only once for the CPU proof. A CUDA implementation
    // instead generates the same states from outer support + local support code
    // + primitive rank inside one CTA.
    std::array<std::vector<Key>, 3> basis{
        q_basis(W, start, words),
        q_basis(W, start + 1, words),
        q_basis(W, start + 2, words)
    };

    std::array<std::map<std::uint32_t, std::vector<Key>>, 3> grouped;
    for (int phase = 0; phase <= steps; ++phase) {
        const int active = start + phase;
        for (const Key& k : basis[static_cast<std::size_t>(phase)]) {
            const auto p = stationary_pack(k);
            const std::uint32_t outer = oneesan::twocell::fusion_outer_mask_at(
                p, start, steps, active);
            grouped[static_cast<std::size_t>(phase)][outer].push_back(k);
        }
    }

    for (std::uint32_t outer = 0;
         outer < (std::uint32_t(1) << outer_bits); ++outer) {
        const int outer_ones = oneesan::twocell::popcount32(outer);
        const Rank block_size = oneesan::twocell::fusion_block_size(
            steps, outer_ones, rt);
        stats.max_block = std::max(stats.max_block, block_size);
        ++stats.blocks;

        std::vector<FValue> cur(static_cast<std::size_t>(block_size));
        std::vector<FValue> next(static_cast<std::size_t>(block_size));
        std::vector<std::uint8_t> loaded(static_cast<std::size_t>(block_size));

        const auto& source_states = grouped[0][outer];
        if (source_states.size() != block_size)
            fail("fusion2 source block dimension W=" + std::to_string(W));
        for (const Key& k : source_states) {
            const auto p = stationary_pack(k);
            const Rank lr = oneesan::twocell::fusion_local_rank_at(
                p, W, start, steps, start, outer_ones, rt);
            const Rank gr = global_stationary_rank(k, W, start, rt, st);
            if (lr >= block_size || loaded[static_cast<std::size_t>(lr)]++)
                fail("fusion2 source local rank collision");
            cur[static_cast<std::size_t>(lr)] = input[static_cast<std::size_t>(gr)];
            ++stats.value_loads;
        }
        for (std::uint8_t x : loaded)
            if (x != 1) fail("fusion2 source local rank gap");

        for (int phase = 0; phase < steps; ++phase) {
            const int active = start + phase;
            std::fill(next.begin(), next.end(), 0);
            std::vector<std::uint8_t> source_seen(static_cast<std::size_t>(block_size));
            const auto& phase_states = grouped[static_cast<std::size_t>(phase)][outer];
            if (phase_states.size() != block_size)
                fail("fusion2 phase dimension W=" + std::to_string(W));

            for (const Key& s : phase_states) {
                const auto ps = stationary_pack(s);
                const Rank sr = oneesan::twocell::fusion_local_rank_at(
                    ps, W, start, steps, active, outer_ones, rt);
                if (sr >= block_size || source_seen[static_cast<std::size_t>(sr)]++)
                    fail("fusion2 phase source rank collision");
                const FValue x = cur[static_cast<std::size_t>(sr)];
                ++stats.local_step_reads;

                for (const auto& [d, c] : K_basis(s, W, active)) {
                    if (c != 1) fail("fusion2 local nonunit");
                    const auto pd = stationary_pack(d);
                    const std::uint32_t dout = oneesan::twocell::fusion_outer_mask_at(
                        pd, start, steps, active + 1);
                    if (dout != outer)
                        fail("fusion2 edge escaped block");
                    const Rank dr = oneesan::twocell::fusion_local_rank_at(
                        pd, W, start, steps, active + 1, outer_ones, rt);
                    if (dr >= block_size) fail("fusion2 destination local range");
                    next[static_cast<std::size_t>(dr)] =
                        fadd(next[static_cast<std::size_t>(dr)], x);
                    ++stats.local_step_writes;
                }
            }
            for (std::uint8_t x : source_seen)
                if (x != 1) fail("fusion2 phase source rank gap");
            cur.swap(next);
        }

        std::vector<std::uint8_t> stored(static_cast<std::size_t>(block_size));
        const auto& destination_states = grouped[steps][outer];
        if (destination_states.size() != block_size)
            fail("fusion2 destination block dimension W=" + std::to_string(W));
        for (const Key& k : destination_states) {
            const auto p = stationary_pack(k);
            const Rank lr = oneesan::twocell::fusion_local_rank_at(
                p, W, start, steps, start + steps, outer_ones, rt);
            const Rank gr = global_stationary_rank(k, W, start + steps, rt, st);
            if (lr >= block_size || stored[static_cast<std::size_t>(lr)]++)
                fail("fusion2 destination local rank collision");
            if (globally_written[static_cast<std::size_t>(gr)]++)
                fail("fusion2 global destination overlap");
            output[static_cast<std::size_t>(gr)] = cur[static_cast<std::size_t>(lr)];
            ++stats.value_stores;
        }
        for (std::uint8_t x : stored)
            if (x != 1) fail("fusion2 destination local rank gap");
    }

    for (std::uint8_t x : globally_written)
        if (x != 1) fail("fusion2 global destination gap");
    return output;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 6 || maxW > 11) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        const Rank states = st.total[W];
        std::vector<FValue> input(static_cast<std::size_t>(states));
        for (Rank r = 0; r < states; ++r)
            input[static_cast<std::size_t>(r)] =
                1 + ((r * 11400714819323198485ULL + 12345ULL) % (kFusionMod - 1));

        for (int start = 0; start + 2 <= W - 3; ++start) {
            const auto ordinary = ordinary_two_pass(input, W, start, words, rt, st);
            BlockStats stats;
            const auto fused = fused_two_pass(input, W, start, words, rt, st, stats);
            if (fused != ordinary)
                fail("fusion2 executor mismatch W=" + std::to_string(W) +
                     " start=" + std::to_string(start));
            if (stats.value_loads != states || stats.value_stores != states)
                fail("fusion2 global traffic count");
            const Rank expected_blocks = Rank(1) << (W - 5);
            if (stats.blocks != expected_blocks)
                fail("fusion2 block count");

            std::cout << "W=" << W
                      << " start=" << start
                      << " states=" << states
                      << " blocks=" << stats.blocks
                      << " max_block=" << stats.max_block
                      << " global_loads=" << stats.value_loads
                      << " global_stores=" << stats.value_stores
                      << " baseline_two_pass_global_values=" << (4 * states)
                      << " fused_global_values=" << (2 * states)
                      << " HBM_value_traffic_ratio=0.5"
                      << " intermediate_global_values=0"
                      << " arithmetic=OK\n";
        }
    }

    const int W = 28;
    const Rank states = st.total[W];
    const Rank blocks = Rank(1) << (W - 5);
    std::cout << "W=28_theory states=" << states
              << " fusion2_blocks=" << blocks
              << " average_block_states=" << double(states) / double(blocks)
              << " baseline_two_pass_u32_GiB="
              << double(4 * states * sizeof(std::uint32_t)) / double(1ULL << 30)
              << " all_blocks_fused_u32_GiB="
              << double(2 * states * sizeof(std::uint32_t)) / double(1ULL << 30)
              << " ideal_HBM_reduction=0.5"
              << "\n";
    std::cout << "ALL_OK fusion2_local_executor=1\n";
    return 0;
}
