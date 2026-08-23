#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

static U64 count_paths(int width) {
    std::vector<U64> cur(width + 3), nxt(width + 3);
    cur[1] = 1;
    for (int s = 0; s < width; ++s) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= width + 1; ++h) if (cur[h]) {
            nxt[h] += cur[h];       // N
            if (h > 0) nxt[h - 1] += cur[h]; // R
            nxt[h + 1] += cur[h];   // L
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static std::vector<U64> count_n_at_each_position(int width) {
    std::vector<std::vector<U64>> f(width + 1, std::vector<U64>(width + 3));
    std::vector<std::vector<U64>> b(width + 1, std::vector<U64>(width + 3));
    f[0][1] = 1;
    for (int s = 0; s < width; ++s) {
        for (int h = 0; h <= width + 1; ++h) if (f[s][h]) {
            f[s + 1][h] += f[s][h];
            if (h > 0) f[s + 1][h - 1] += f[s][h];
            f[s + 1][h + 1] += f[s][h];
        }
    }
    b[width][0] = 1;
    for (int s = width - 1; s >= 0; --s) {
        for (int h = 0; h <= width + 1; ++h) {
            U64 z = b[s + 1][h];
            if (h > 0) z += b[s + 1][h - 1];
            if (h + 1 <= width + 1) z += b[s + 1][h + 1];
            b[s][h] = z;
        }
    }

    // Position p is processed after width-1-p symbols from the HIGH end.
    std::vector<U64> out(width);
    for (int p = 0; p < width; ++p) {
        const int s = width - 1 - p;
        U64 z = 0;
        for (int h = 0; h <= width + 1; ++h)
            z += f[s][h] * b[s + 1][h]; // force this symbol to N
        out[p] = z;
    }
    return out;
}

static long double u128_ld(U128 x) {
    const U64 lo = U64(x);
    const U64 hi = U64(x >> 64);
    return static_cast<long double>(hi) * 18446744073709551616.0L
         + static_cast<long double>(lo);
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 40 || low < 1 || high < 1) return 1;

    const U64 main_states = count_paths(W);
    const U64 blocked_states = count_paths(W - 1);
    const auto ncount = count_n_at_each_position(W);
    for (int p = 1; p < W; ++p) {
        if (ncount[p] != blocked_states) {
            std::cerr << "delete-N bijection count mismatch p=" << p
                      << " got=" << ncount[p] << " blocked=" << blocked_states << '\n';
            return 2;
        }
    }

    U128 high_rep_steps_per_row = 0, low_rep_steps_per_row = 0;
    for (int p = W - 1; p >= low + 1; --p) high_rep_steps_per_row += ncount[p];
    for (int p = low; p >= 1; --p) low_rep_steps_per_row += ncount[p];
    const U128 high_rep_steps = high_rep_steps_per_row * U128(W);
    const U128 low_rep_steps = low_rep_steps_per_row * U128(W);
    const U128 reps = high_rep_steps + low_rep_steps;
    const U128 removed_dense_lookups = 2 * reps;

    // v0.4 orbit hot path: one active-code load for every main state and two
    // dense packed-rank loads for each representative. v0.5 replaces the code
    // load with aux for every state and the two rank loads with one descriptor
    // load for each representative. Logical load reduction is therefore one
    // uint32 per representative. This is NOT a prediction of physical HBM
    // traffic because HIGH rows broadcast heavily and caches can coalesce hits.
    const U128 logical_bytes_saved = 4 * reps;
    const long double tib = 1099511627776.0L;

    std::cout << std::fixed << std::setprecision(6)
              << "orbitaux-savings W=" << W << " low=" << low << " high=" << high << '\n'
              << "main_states=" << main_states << " blocked_states=" << blocked_states
              << " rep_fraction=" << double((long double)blocked_states / main_states) << '\n'
              << "high_rep_state_steps_per_residue=" << double(u128_ld(high_rep_steps))
              << " low_rep_state_steps_per_residue=" << double(u128_ld(low_rep_steps)) << '\n'
              << "removed_dense_rank_lookups_per_residue=" << double(u128_ld(removed_dense_lookups))
              << " removed_dense_rank_lookups_trillion="
              << double(u128_ld(removed_dense_lookups) / 1.0e12L) << '\n'
              << "logical_load_bytes_saved_tib="
              << double(u128_ld(logical_bytes_saved) / tib) << '\n';
    return 0;
}
