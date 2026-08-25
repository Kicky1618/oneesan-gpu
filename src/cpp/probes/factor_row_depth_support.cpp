#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;
using StateSet = std::unordered_set<MateID>;

static int max_height(MateID m, int width) {
    int h = 1, mx = 1;
    for (int p = width - 1; p >= 0; --p) {
        const MateValue v = mget(m, p);
        if (v == R) --h;
        else if (v == L) { ++h; mx = std::max(mx, h); }
    }
    return mx;
}

static U64 capped_state_count(int width, int cap) {
    if (cap < 1) return 0;
    std::vector<U64> cur(cap + 1), nxt(cap + 1);
    cur[1] = 1;
    for (int pos = 0; pos < width; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= cap; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            if (h + 1 <= cap) nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static void step_position(StateSet& main, StateSet& block, int W, int p) {
    StateSet next_main = main; // excluded MAIN branch is identity
    StateSet next_block;
    next_main.reserve(main.size() * 2 + block.size() + 1);
    next_block.reserve(main.size() / 2 + 1);

    for (MateID m : main) {
        const IncludeResult z = include_horizontal(m, W, p);
        if (!z.valid) continue;
        if (z.blocked) next_block.insert(z.mate);
        else next_main.insert(z.mate);
    }
    for (MateID d : block) next_main.insert(blocked_exclude(d, p));
    main.swap(next_main);
    block.swap(next_block);
}

static void verify_width(int W, int low) {
    if (low < 1 || low >= W - 1) std::exit(20);
    StateSet main, block;
    main.insert(MateID(R) << (2 * (W - 1)));

    static const U64 W14_EXPECTED[] = {
        8192ULL, 80782ULL, 159094ULL, 190400ULL,
        196406ULL, 196924ULL, 196938ULL,
    };

    for (int row = 1; row <= W; ++row) {
        if (!block.empty()) {
            std::cerr << "row started with nonzero BLOCKED support W=" << W
                      << " row=" << row << '\n';
            std::exit(21);
        }

        for (int p = W - 1; p >= 1; --p) {
            step_position(main, block, W, p);
            if (p == low + 1) {
                for (MateID m : main) if (max_height(m, W) > row) {
                    std::cerr << "HIGH midpoint MAIN exceeds row-depth cap W=" << W
                              << " row=" << row << " p=" << p << '\n';
                    std::exit(22);
                }
                for (MateID d : block) if (max_height(d, W - 1) > row) {
                    std::cerr << "HIGH midpoint BLOCKED exceeds row-depth cap W=" << W
                              << " row=" << row << " p=" << p << '\n';
                    std::exit(23);
                }
            }
        }

        if (!block.empty()) {
            std::cerr << "p=1 left BLOCKED support W=" << W
                      << " row=" << row << '\n';
            std::exit(24);
        }
        const int cap = std::min(row, (W + 1) / 2);
        const U64 expected = capped_state_count(W, cap);
        if (main.size() != expected) {
            std::cerr << "row-boundary support != max-height cap W=" << W
                      << " row=" << row << " got=" << main.size()
                      << " expected=" << expected << '\n';
            std::exit(25);
        }
        for (MateID m : main) if (max_height(m, W) > row) {
            std::cerr << "row-boundary MAIN exceeds depth cap W=" << W
                      << " row=" << row << '\n';
            std::exit(26);
        }
        if (W == 14 && row <= 7 && main.size() != W14_EXPECTED[row - 1]) {
            std::cerr << "W14 pinned support mismatch row=" << row << '\n';
            std::exit(27);
        }
    }

    std::cout << "row-depth-support OK W=" << W << " low=" << low
              << " final_main=" << main.size() << '\n';
}

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 14;
    if (max_w < 6 || max_w > 14) return 1;
    for (int W = 6; W <= max_w; W += 2) verify_width(W, W / 2);

    constexpr int W = 28;
    constexpr int MAXH = 14;
    const U64 M = capped_state_count(W, MAXH);
    const U64 D = capped_state_count(W - 1, MAXH);
    if (M != 385719506620ULL || D != 135015505407ULL) return 2;

    U64 cap_words = 0;
    for (int row = 1; row <= W; ++row) {
        // Before HIGH: previous complete row implies BLOCKED=0. MAIN cannot
        // exceed max height row-1 (row 1 has the single initial state).
        const U64 gather_main = row == 1
            ? 1ULL : capped_state_count(W, std::min(row - 1, MAXH));
        // During one new grid row, frontier nesting can increase by at most one.
        // Thus the HIGH scatter never needs to transfer states above cap=row.
        const U64 scatter_main = capped_state_count(W, std::min(row, MAXH));
        const U64 scatter_block = capped_state_count(W - 1, std::min(row, MAXH));
        cap_words += gather_main + scatter_main + scatter_block;
    }

    const U64 dense_words = U64(W) * (2ULL * M + D); // v0.12 logical HIGH I/O
    if (dense_words != 25380726522116ULL
        || cap_words != 22074394853240ULL) {
        std::cerr << "n=27 row-depth traffic regression mismatch\n";
        return 3;
    }

    const long double ratio = (long double)cap_words / (long double)dense_words;
    const long double dense_tib = (long double)dense_words * 4.0L
        / (long double)(U64(1) << 40);
    const long double cap_tib = (long double)cap_words * 4.0L
        / (long double)(U64(1) << 40);

    std::cout << std::fixed << std::setprecision(9)
              << "n27_row_depth_dense_words=" << dense_words
              << " cap_words=" << cap_words
              << " cap_ratio=" << double(ratio)
              << " cap_reduction=" << double(1.0L - ratio) << '\n'
              << "v012_logical_high_io_tib=" << double(dense_tib)
              << " row_depth_cap_logical_tib=" << double(cap_tib)
              << " balanced_7of8_peer_tib=" << double(cap_tib * 7.0L / 8.0L)
              << '\n';

    std::cout << "n27_main_cap_counts=";
    for (int h = 1; h <= MAXH; ++h)
        std::cout << (h == 1 ? "" : ",") << capped_state_count(W, h);
    std::cout << '\n';
    return 0;
}
