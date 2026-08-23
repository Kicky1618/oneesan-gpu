#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

static std::vector<MateID> enumerate_states(int width) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        if (pos < 0) {
            if (h == 0) out.push_back(m);
            return;
        }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, m);
        if (h > 0) self(self, pos - 1, h - 1,
                        m | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             m | (MateID(L) << (2 * pos)));
    };
    rec(rec, width - 1, 1, 0);
    return out;
}

static U64 state_count(int width) {
    std::vector<U64> cur(width + 3), nxt(width + 3);
    cur[1] = 1;
    for (int pos = 0; pos < width; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= width + 1; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static void verify_width(int W) {
    const auto main_states = enumerate_states(W);
    const auto blocked_states = enumerate_states(W - 1);

    // A complete Grid-FP row ends at p=1.  For p=1 every valid included
    // transition remains MAIN: include_horizontal() explicitly avoids shrink()
    // in NN/NR/NL/LL/RR/RL p=1 cases.  Existing BLOCKED input has only its
    // excluded branch and blocked_exclude() inserts N back into MAIN.
    // Therefore the output BLOCKED vector after p=1 is identically zero for
    // arbitrary input values, not merely for the particular counting DP seed.
    U64 valid_main = 0;
    U64 blocked_main_outputs = 0;
    for (MateID m : main_states) {
        const IncludeResult z = include_horizontal(m, W, 1);
        if (!z.valid) continue;
        ++valid_main;
        if (z.blocked) ++blocked_main_outputs;
    }
    if (blocked_main_outputs != 0) {
        std::cerr << "p=1 MAIN produced BLOCKED output W=" << W << '\n';
        std::exit(10);
    }

    U64 excluded_blocked_outputs = 0;
    for (MateID d : blocked_states) {
        const MateID z = blocked_exclude(d, 1);
        if (z == d) ++excluded_blocked_outputs; // impossible width mismatch guard
    }
    if (excluded_blocked_outputs != 0) {
        std::cerr << "blocked_exclude failed to return MAIN-width state W=" << W << '\n';
        std::exit(11);
    }

    if (main_states.size() != state_count(W)
        || blocked_states.size() != state_count(W - 1)) {
        std::cerr << "state enumeration mismatch W=" << W << '\n';
        std::exit(12);
    }

    std::cout << "row-boundary-blockzero W=" << W
              << " main=" << main_states.size()
              << " blocked=" << blocked_states.size()
              << " valid_main_p1=" << valid_main << '\n';
}

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 13;
    if (max_w < 4 || max_w > 13) return 1;
    for (int W = 4; W <= max_w; ++W) verify_width(W);

    const U64 M28 = state_count(28);
    const U64 D28 = state_count(27);
    if (M28 != 385719506620ULL || D28 != 135015505407ULL) {
        std::cerr << "n=27 state-count regression mismatch\n";
        return 2;
    }

    const long double full_io = 2.0L * (long double)(M28 + D28);
    const long double skip_io = 2.0L * (long double)M28 + (long double)D28;
    const long double ratio = skip_io / full_io;
    std::cout << "n27_full_high_io_state_words_per_row="
              << 2ULL * (M28 + D28)
              << " n27_skip_block_gather_words_per_row="
              << 2ULL * M28 + D28
              << " ratio=" << double(ratio)
              << " reduction=" << double(1.0L - ratio) << '\n';
    return 0;
}
