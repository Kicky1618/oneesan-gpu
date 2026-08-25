#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;
using U128 = unsigned __int128;

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

static std::uint32_t flip_low(std::uint32_t lc, int low, int depth) {
    int s = depth;
    for (int pos = low - 1; pos >= 0; --pos) {
        const MateValue v = MateValue((lc >> (2 * pos)) & 3u);
        if (v == L) {
            ++s;
        } else if (v == R) {
            if (--s == 0) {
                const std::uint32_t z = 3u << (2 * pos);
                return (lc & ~z) | (std::uint32_t(L) << (2 * pos));
            }
        }
    }
    return 0xffffffffu;
}

static int cross_depth(MateID m, int low, int p) {
    MateID t = msetpair(m, p, NN);
    int q = p - 1, s = 1;
    for (;;) {
        --q;
        if (q < low) return s;
        const MateValue v = mget(t, q);
        if (v == L) ++s;
        else if (v == R) --s;
        if (!s) return 0;
    }
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || low >= 16 || high < 1 || high >= 16)
        return 1;

    const std::uint32_t low_masks = 1u << low;
    std::vector<std::vector<std::vector<std::uint32_t>>> low_groups(
        low + 1, std::vector<std::vector<std::uint32_t>>(low_masks));
    std::vector<std::uint32_t> representative(low + 1, 0xffffffffu);
    std::vector<U64> low_count(low + 1);
    std::unordered_map<std::uint32_t, std::uint16_t> low_mask_rank;
    low_mask_rank.reserve(1300000);

    for (int h0 = 0; h0 <= low; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code,
                       std::uint32_t mask) -> void {
            if (pos < 0) {
                if (h == 0) low_groups[h0][mask].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code, mask);
            if (h) self(self, pos - 1, h - 1,
                        code | (1u << (2 * pos)), mask | (1u << pos));
            self(self, pos - 1, h + 1,
                 code | (2u << (2 * pos)), mask | (1u << pos));
        };
        rec(rec, low - 1, h0, 0, 0);
        for (std::uint32_t mask = 0; mask < low_masks; ++mask) {
            const auto& g = low_groups[h0][mask];
            if (representative[h0] == 0xffffffffu && !g.empty())
                representative[h0] = g.front();
            for (std::uint32_t r = 0; r < g.size(); ++r) {
                if (r > 0xffffu) return 2;
                low_mask_rank.emplace(g[r], std::uint16_t(r));
                ++low_count[h0];
            }
        }
    }

    std::vector<std::vector<std::uint32_t>> high_codes(high + 2);
    auto rech = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            high_codes[h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1, code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, code | (2u << (2 * pos)));
    };
    rech(rech, high - 1, 1, 0);

    std::array<U64, 16> depth_rows{};
    std::array<U128, 16> depth_states{};
    std::vector<std::array<U64, 16>> rows_by_hs(low + 1);
    U64 block_rows = 0, cross_rows = 0, invalid_closure_rows = 0;
    U128 block_states = 0, cross_states = 0;
    const MateID low_mask_bits = (MateID(1) << (2 * low)) - 1;

    for (int p = W - 1; p >= low + 1; --p) {
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs > low || representative[hs] == 0xffffffffu) continue;
                const std::uint32_t lc = representative[hs];
                for (std::uint32_t hc : high_codes[he]) {
                    const MateID m = MateID(lc)
                        | (MateID(cv) << (2 * low))
                        | (MateID(hc) << (2 * (low + 1)));
                    const MateValuePair source = mpair(m, p);
                    if (source != LL && source != RR && source != RL) continue;
                    const IncludeResult z = include_horizontal(m, W, p);
                    if (!z.valid || !z.blocked) {
                        ++invalid_closure_rows;
                        continue;
                    }
                    const std::uint32_t lc2 = std::uint32_t(z.mate & low_mask_bits);
                    if (lc2 == lc) {
                        ++block_rows;
                        block_states += low_count[hs];
                    } else {
                        const int depth = cross_depth(m, low, p);
                        if (depth <= 0 || depth >= int(depth_rows.size())) return 3;
                        ++cross_rows;
                        ++depth_rows[size_t(depth)];
                        ++rows_by_hs[hs][size_t(depth)];
                        cross_states += low_count[hs];
                        depth_states[size_t(depth)] += low_count[hs];
                    }
                }
            }
        }
    }

    U64 max_target_rank = 0;
    U64 checked_cross_columns = 0;
    for (int hs = 0; hs <= low; ++hs) {
        for (int depth = 1; depth < int(depth_rows.size()); ++depth) {
            if (!rows_by_hs[hs][size_t(depth)]) continue;
            U64 valid = 0;
            for (std::uint32_t mask = 0; mask < low_masks; ++mask) {
                for (std::uint32_t lc : low_groups[hs][mask]) {
                    const std::uint32_t lc2 = flip_low(lc, low, depth);
                    if (lc2 == 0xffffffffu) continue;
                    const auto it = low_mask_rank.find(lc2);
                    if (it == low_mask_rank.end()) {
                        std::cerr << "HIGH CROSS target LOW code missing hs=" << hs
                                  << " depth=" << depth << '\n';
                        return 4;
                    }
                    max_target_rank = std::max<U64>(max_target_rank, it->second);
                    ++valid;
                }
            }
            if (valid != low_count[hs]) {
                std::cerr << "HIGH CROSS has invalid LOW columns hs=" << hs
                          << " depth=" << depth << " valid=" << valid
                          << " total=" << low_count[hs] << '\n';
                return 5;
            }
            checked_cross_columns += valid;
        }
    }

    int max_depth = 0;
    for (int d = 1; d < int(depth_rows.size()); ++d)
        if (depth_rows[size_t(d)]) max_depth = d;
    const U64 low_codes = low_mask_rank.size();
    const U64 direct_target_bytes = U64(max_depth) * low_codes * sizeof(std::uint16_t);
    const int hybrid_depth = std::min(6, max_depth);
    const U64 hybrid_bytes = U64(hybrid_depth) * low_codes * sizeof(std::uint16_t);
    U128 hybrid_states = 0;
    for (int d = 1; d <= hybrid_depth; ++d) hybrid_states += depth_states[size_t(d)];
    const U128 closure_states = block_states + cross_states;

    if (W == 28 && low == 14) {
        const std::array<U64, 13> expected_rows = {
            0, 673698, 518595, 358071, 220791, 120834, 58200,
            24387, 8748, 2625, 636, 117, 14
        };
        if (low_codes != 1201917ULL || block_rows != 7315213ULL
            || cross_rows != 1986716ULL || invalid_closure_rows != 0
            || block_states != U128(1285374327564ULL)
            || cross_states != U128(218576117914ULL)
            || closure_states != U128(1503950445478ULL)
            || max_depth != 12 || max_target_rank != 1000ULL
            || direct_target_bytes != 28846008ULL
            || hybrid_bytes != 14423004ULL
            || hybrid_states != U128(218380105092ULL)) {
            std::cerr << "n=27 HIGH closure CROSS aggregate regression mismatch\n";
            return 6;
        }
        for (int d = 1; d <= 12; ++d) {
            if (depth_rows[size_t(d)] != expected_rows[size_t(d)]) {
                std::cerr << "n=27 HIGH closure CROSS depth regression mismatch d="
                          << d << '\n';
                return 7;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "highclosure-cross W=" << W << " low=" << low
              << " high=" << high << '\n'
              << "low_codes=" << low_codes
              << " block_rows=" << block_rows
              << " cross_rows=" << cross_rows
              << " invalid_closure_rows=" << invalid_closure_rows << '\n'
              << "block_states=" << double(as_ld(block_states))
              << " cross_states=" << double(as_ld(cross_states))
              << " cross_state_fraction="
              << double(as_ld(cross_states) / as_ld(closure_states)) << '\n'
              << "max_cross_depth=" << max_depth
              << " max_target_mask_rank=" << max_target_rank
              << " checked_cross_columns=" << checked_cross_columns << '\n'
              << "direct_uint16_target_mib="
              << double(direct_target_bytes) / double(1ULL << 20)
              << " hybrid_depth6_mib="
              << double(hybrid_bytes) / double(1ULL << 20)
              << " hybrid_depth6_cross_coverage="
              << double(as_ld(hybrid_states) / as_ld(cross_states)) << '\n';
    for (int d = 1; d <= max_depth; ++d) {
        std::cout << "depth=" << d
                  << " rows=" << depth_rows[size_t(d)]
                  << " states=" << double(as_ld(depth_states[size_t(d)]))
                  << " state_fraction="
                  << double(as_ld(depth_states[size_t(d)]) / as_ld(cross_states))
                  << '\n';
    }
    return 0;
}
