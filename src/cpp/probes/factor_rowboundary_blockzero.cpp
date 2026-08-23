#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unordered_set>
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
    const std::unordered_set<MateID> main_set(main_states.begin(), main_states.end());

    // A complete Grid-FP row ends at p=1. For p=1 every valid included
    // transition remains MAIN: include_horizontal() explicitly avoids shrink()
    // in NN/NR/NL/LL/RR/RL p=1 cases. Existing BLOCKED input has only its
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

    for (MateID d : blocked_states) {
        const MateID z = blocked_exclude(d, 1);
        if (!main_set.count(z)) {
            std::cerr << "blocked_exclude escaped MAIN state set W=" << W << '\n';
            std::exit(11);
        }
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

    constexpr int W = 28;
    constexpr long double PEER_FRACTION_BALANCED = 7.0L / 8.0L;
    const U64 M = state_count(W);
    const U64 D = state_count(W - 1);
    if (M != 385719506620ULL || D != 135015505407ULL) {
        std::cerr << "n=27 state-count regression mismatch\n";
        return 2;
    }

    const U64 full_words = 2ULL * (M + D);
    const U64 skip_words = 2ULL * M + D;
    if (full_words != 1041470024054ULL || skip_words != 906454518647ULL) {
        std::cerr << "n=27 HIGH I/O word-count regression mismatch\n";
        return 3;
    }

    const long double ratio = (long double)skip_words / (long double)full_words;
    const long double full_tib = (long double)full_words * 4.0L * W
        / (long double)(U64(1) << 40);
    const long double skip_tib = (long double)skip_words * 4.0L * W
        / (long double)(U64(1) << 40);
    const long double saved_tib = full_tib - skip_tib;
    const long double peer_full_tib = full_tib * PEER_FRACTION_BALANCED;
    const long double peer_skip_tib = skip_tib * PEER_FRACTION_BALANCED;

    std::cout << std::fixed << std::setprecision(9)
              << "n27_full_high_io_state_words_per_row=" << full_words
              << " n27_skip_block_gather_words_per_row=" << skip_words
              << " ratio=" << double(ratio)
              << " reduction=" << double(1.0L - ratio) << '\n'
              << "logical_high_io_tib_per_residue=" << double(full_tib)
              << " zero_block_gather_tib_per_residue=" << double(skip_tib)
              << " logical_saved_tib_per_residue=" << double(saved_tib) << '\n'
              << "balanced_7of8_peer_tib_before=" << double(peer_full_tib)
              << " balanced_7of8_peer_tib_after=" << double(peer_skip_tib)
              << " balanced_peer_saved_tib=" << double(peer_full_tib - peer_skip_tib)
              << '\n';
    return 0;
}
