#pragma push_macro("main")
#undef main
#define main two_cell_recoupling_common_rank_probe_main_unused
#include "two_cell_recoupling_common_rank_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_recoupling_rank.hpp"

namespace {

std::uint32_t support_bits(const Word& w) {
    std::uint32_t z = 0;
    for (int p = 0; p < static_cast<int>(w.size()); ++p)
        if (w[p] != N) z |= std::uint32_t(1) << p;
    return z;
}

std::uint32_t left_bits(const Word& w) {
    std::uint32_t z = 0;
    for (int p = 0; p < static_cast<int>(w.size()); ++p)
        if (w[p] == L) z |= std::uint32_t(1) << p;
    return z;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 15;
    if (maxW < 1 || maxW > 15) return 2;

    const auto tables = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    Rank total_states = 0;
    Rank total_old_iterations = 0;
    Rank total_new_iterations = 0;
    int max_old = 0;
    int max_new = 0;

    for (int len = 1; len <= maxW; ++len) {
        Rank checked = 0;
        Rank old_iterations = 0;
        Rank new_iterations = 0;
        int local_max_old = 0;
        int local_max_new = 0;

        for (const Word& w : words[len]) {
            const std::uint32_t support = support_bits(w);
            const std::uint32_t left = left_bits(w);
            const Rank a = oneesan::twocell::primitive_rank_scan(
                support, left, len, tables);
            const Rank b = oneesan::twocell::primitive_rank(
                support, left, len, tables);
            if (a != b)
                fail("L-only primitive rank mismatch len=" + std::to_string(len));

            const int old_it = len;
            const int new_it = oneesan::twocell::popcount32(left);
            old_iterations += old_it;
            new_iterations += new_it;
            local_max_old = std::max(local_max_old, old_it);
            local_max_new = std::max(local_max_new, new_it);
            ++checked;
        }

        std::cout << "len=" << len
                  << " checked=" << checked
                  << " old_iterations=" << old_iterations
                  << " new_iterations=" << new_iterations
                  << " reduction="
                  << (new_iterations ? double(old_iterations) / double(new_iterations) : 0.0)
                  << " max_old=" << local_max_old
                  << " max_new=" << local_max_new
                  << " exact=OK\n";

        total_states += checked;
        total_old_iterations += old_iterations;
        total_new_iterations += new_iterations;
        max_old = std::max(max_old, local_max_old);
        max_new = std::max(max_new, local_max_new);
    }

    std::cout << "aggregate states=" << total_states
              << " old_iterations=" << total_old_iterations
              << " new_iterations=" << total_new_iterations
              << " reduction="
              << (total_new_iterations
                    ? double(total_old_iterations) / double(total_new_iterations)
                    : 0.0)
              << " max_old=" << max_old
              << " max_new=" << max_new
              << "\n";

    std::cout << "W=28_theory A_len=27 max_L=13"
              << " C_len=26 max_L=12"
              << " primitive_rank_slot_scan=0"
              << " primitive_rank_L_endpoint_scan=1"
              << "\n";
    std::cout << "ALL_OK primitive_rank_lonly=1\n";
    return 0;
}
