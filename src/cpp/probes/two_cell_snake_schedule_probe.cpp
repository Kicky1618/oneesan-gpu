#include "../../common/two_cell_snake_schedule.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void fail(const char* what, int W, int index = -1) {
    std::cerr << "FAIL " << what << " W=" << W;
    if (index >= 0) std::cerr << " index=" << index;
    std::cerr << '\n';
    std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 28;
    if (maxW < 6 || maxW > oneesan::twocell::kMaxWidth) return 2;

    for (int W = 6; W <= maxW; W += 2) {
        const auto schedule = oneesan::twocell::make_snake_schedule(W);
        if (!schedule.valid) fail("invalid schedule", W);
        if (schedule.size != W - 2) fail("pair count", W, schedule.size);

        std::vector<int> forward(static_cast<std::size_t>(W - 3), 0);
        std::vector<int> reverse(static_cast<std::size_t>(W - 3), 0);
        int right_turn = 0;
        int left_turn = 0;
        int forward_pairs = 0;
        int reverse_pairs = 0;

        for (int q = 0; q < schedule.size; ++q) {
            const auto p = schedule.pair[q];
            using Kind = oneesan::twocell::SnakePairKind;
            switch (p.kind) {
                case Kind::ForwardFusion2:
                    if (p.start < 0 || p.start + 1 > W - 4)
                        fail("forward pair range", W, p.start);
                    ++forward[p.start];
                    ++forward[p.start + 1];
                    ++forward_pairs;
                    break;
                case Kind::RightBoundary:
                    if (p.start != W - 4)
                        fail("right boundary start", W, p.start);
                    ++forward[W - 4];
                    ++right_turn;
                    break;
                case Kind::ReverseFusion2:
                    // low start s covers reverse destination-active indices
                    // s+1 and s: active s+2 -> s+1 -> s.
                    if (p.start < 1 || p.start + 1 > W - 4)
                        fail("reverse pair range", W, p.start);
                    ++reverse[p.start + 1];
                    ++reverse[p.start];
                    ++reverse_pairs;
                    break;
                case Kind::LeftBoundary:
                    if (p.start != 0)
                        fail("left boundary start", W, p.start);
                    ++reverse[0];
                    ++left_turn;
                    break;
            }
        }

        for (int i = 0; i <= W - 4; ++i) {
            if (forward[i] != 1) fail("forward K coverage", W, i);
            if (reverse[i] != 1) fail("reverse K coverage", W, i);
        }
        if (right_turn != 1 || left_turn != 1)
            fail("turn coverage", W);
        if (forward_pairs != (W - 4) / 2 ||
            reverse_pairs != (W - 4) / 2)
            fail("fusion pair count", W);

        const int total_k = 2 * (W - 3);
        const int total_turns = 2;
        const int total_ops = total_k + total_turns;
        if (2 * schedule.size != total_ops)
            fail("operator count", W);

        std::cout << "W=" << W
                  << " pairs=" << schedule.size
                  << " forward_fusion2=" << forward_pairs
                  << " reverse_fusion2=" << reverse_pairs
                  << " K_steps=" << total_k
                  << " turns=" << total_turns
                  << " total_ops=" << total_ops
                  << " coverage=EXACT\n";
    }

    const auto w28 = oneesan::twocell::make_snake_schedule(28);
    std::cout << "W=28 schedule:";
    for (int q = 0; q < w28.size; ++q)
        std::cout << ' '
                  << oneesan::twocell::snake_pair_kind_name(w28.pair[q].kind)
                  << ':' << w28.pair[q].start;
    std::cout << '\n';
    std::cout << "ALL_OK paired_stationary_snake_schedule=1\n";
    return 0;
}
