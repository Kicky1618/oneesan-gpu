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
using U128 = unsigned __int128;

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

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

static void verify_small_width(int W) {
    const auto main_states = enumerate_states(W);
    const auto block_states = enumerate_states(W - 1);
    const U64 sw2 = state_count(W - 2);

    for (int p = W - 1; p >= 2; --p) {
        std::unordered_set<MateID> blocked_dest;
        std::unordered_set<MateID> main_dest;
        std::unordered_set<MateID> exclude_image;
        std::unordered_set<MateID> nn_image;
        std::unordered_set<MateID> rnln_image;

        for (MateID d : block_states)
            exclude_image.insert(blocked_exclude(d, p));
        if (exclude_image.size() != block_states.size()) {
            std::cerr << "blocked_exclude not injective W=" << W << " p=" << p << '\n';
            std::exit(10);
        }

        U64 valid_main = 0;
        for (MateID m : main_states) {
            const MateValuePair w = mpair(m, p);
            const IncludeResult z = include_horizontal(m, W, p);
            if (!z.valid) continue;
            ++valid_main;
            if (z.blocked) {
                blocked_dest.insert(z.mate);
            } else {
                main_dest.insert(z.mate);
                if (w == NN) nn_image.insert(z.mate);
                else if (w == RN || w == LN) rnln_image.insert(z.mate);
            }
        }
        main_dest.insert(exclude_image.begin(), exclude_image.end());

        if (blocked_dest.size() != block_states.size()) {
            std::cerr << "blocked destination coverage mismatch W=" << W << " p=" << p << '\n';
            std::exit(11);
        }
        if (nn_image.size() != sw2) {
            std::cerr << "NN image size mismatch W=" << W << " p=" << p << '\n';
            std::exit(12);
        }
        for (MateID z : nn_image) if (exclude_image.count(z)) {
            std::cerr << "NN and blocked-exclude images overlap W=" << W << " p=" << p << '\n';
            std::exit(13);
        }
        for (MateID z : rnln_image) if (!exclude_image.count(z)) {
            std::cerr << "RN/LN image escaped blocked-exclude image W=" << W << " p=" << p << '\n';
            std::exit(14);
        }
        if (main_dest.size() != block_states.size() + sw2) {
            std::cerr << "main destination union mismatch W=" << W << " p=" << p << '\n';
            std::exit(15);
        }

        const U64 expected_valid_main = main_states.size() - sw2;
        if (valid_main != expected_valid_main) {
            std::cerr << "valid-main count mismatch W=" << W << " p=" << p << '\n';
            std::exit(16);
        }
        const U64 updates = valid_main + block_states.size();
        const U64 distinct = main_dest.size() + blocked_dest.size();
        const U64 expected_updates = main_states.size() + block_states.size() - sw2;
        const U64 expected_distinct = 2 * U64(block_states.size()) + sw2;
        if (updates != expected_updates || distinct != expected_distinct) {
            std::cerr << "aggregation formula mismatch W=" << W << " p=" << p << '\n';
            std::exit(17);
        }
    }
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int exhaustive_max = argc > 2 ? std::atoi(argv[2]) : 13;
    if (W < 4 || W > 30 || exhaustive_max < 4 || exhaustive_max > 13) return 1;

    for (int w = 4; w <= exhaustive_max; ++w) verify_small_width(w);

    const U64 m = state_count(W);
    const U64 d = state_count(W - 1);
    const U64 s2 = state_count(W - 2);
    const U64 updates_per_p = m + d - s2;
    const U64 distinct_dest_per_p = 2 * d + s2;
    const long double aggregate_ratio =
        (long double)distinct_dest_per_p / (long double)updates_per_p;
    const long double all_remote_tib =
        (long double)updates_per_p * (W - 1) * W * 4.0L / (long double)(U64(1) << 40);
    const long double all_remote_aggregated_tib =
        (long double)distinct_dest_per_p * (W - 1) * W * 4.0L
        / (long double)(U64(1) << 40);

    if (W == 28) {
        if (m != 385719506620ULL
            || d != 135015505407ULL
            || s2 != 47337954326ULL
            || updates_per_p != 473397057701ULL
            || distinct_dest_per_p != 317368965140ULL) {
            std::cerr << "n=27 HIGH aggregation regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "factor-highupdate-aggregation W=" << W
              << " exhaustive_verified_through=" << exhaustive_max << '\n'
              << "main_states=" << m
              << " blocked_states=" << d
              << " width_minus_2_states=" << s2 << '\n'
              << "updates_per_high_position=" << updates_per_p
              << " distinct_destinations_per_high_position=" << distinct_dest_per_p << '\n'
              << "ideal_destination_aggregation_ratio=" << double(aggregate_ratio)
              << " ideal_update_reduction=" << double(1.0L - aggregate_ratio) << '\n'
              << "all_remote_unaggregated_tib_per_residue=" << double(all_remote_tib)
              << " all_remote_one_message_per_destination_tib_per_residue="
              << double(all_remote_aggregated_tib) << '\n';
    return 0;
}
