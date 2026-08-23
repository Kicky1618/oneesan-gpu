#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using oneesan::gridfp::IncludeResult;
using oneesan::gridfp::MateID;
using oneesan::gridfp::MateValue;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;
using oneesan::gridfp::NN;
using oneesan::gridfp::mget;
using oneesan::gridfp::mset;
using oneesan::gridfp::msetpair;
using oneesan::gridfp::include_horizontal;
using oneesan::gridfp::blocked_exclude;

static int end_height(std::uint32_t code, int len) {
    int h = 1;
    for (int p = len - 1; p >= 0; --p) {
        const MateValue v = MateValue((code >> (2 * p)) & 3u);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return h;
}

static std::vector<std::vector<std::uint32_t>> build_high_all(int H) {
    std::vector<std::vector<std::uint32_t>> out(H + 3);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            out[h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1, code | (std::uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1, code | (std::uint32_t(L) << (2 * pos)));
    };
    rec(rec, H - 1, 1, 0);
    return out;
}

static std::vector<std::vector<std::uint32_t>> build_low_all(int Lw) {
    std::vector<std::vector<std::uint32_t>> out(Lw + 3);
    for (int h0 = 0; h0 <= Lw + 1; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) out[h0].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0) self(self, pos - 1, h - 1, code | (std::uint32_t(R) << (2 * pos)));
            self(self, pos - 1, h + 1, code | (std::uint32_t(L) << (2 * pos)));
        };
        rec(rec, Lw - 1, h0, 0);
    }
    return out;
}

struct HighRanks {
    std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> rank;
};

static HighRanks make_high_ranks(const std::vector<std::vector<std::uint32_t>>& all) {
    HighRanks r;
    r.rank.resize(all.size());
    for (std::size_t h = 0; h < all.size(); ++h)
        for (std::uint32_t i = 0; i < all[h].size(); ++i)
            r.rank[h].emplace(all[h][i], i);
    return r;
}

enum class Kind : std::uint8_t { Invalid, Main, Block, Cross };

struct Desc {
    Kind kind = Kind::Invalid;
    int block = 0;
    std::uint32_t high_rank = 0;
    int depth = 0;
};

static int cross_depth(MateID m, int p, int low) {
    MateID t = msetpair(m, p, NN);
    int q = p - 1;
    int s = 1;
    for (;;) {
        --q;
        if (q < low) return s;
        const MateValue v = mget(t, q);
        if (v == L) ++s;
        else if (v == R) --s;
        if (!s) return 0;
    }
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

static Desc build_main_desc(
    MateID representative, int W, int low, int high, int p,
    const HighRanks& ranks
) {
    const std::uint32_t LM = (1u << (2 * low)) - 1u;
    const std::uint32_t HM = (1u << (2 * high)) - 1u;
    const std::uint32_t lc = std::uint32_t(representative) & LM;
    const IncludeResult z = include_horizontal(representative, W, p);
    if (!z.valid) return {};

    const std::uint32_t lc2 = std::uint32_t(z.mate) & LM;
    const std::uint32_t hc2 = z.blocked
        ? std::uint32_t((z.mate >> (2 * low)) & HM)
        : std::uint32_t((z.mate >> (2 * (low + 1))) & HM);
    const int h2 = end_height(hc2, high);
    const auto it = ranks.rank[h2].find(hc2);
    if (it == ranks.rank[h2].end()) {
        std::cerr << "destination HIGH code missing\n";
        std::exit(20);
    }

    Desc d;
    d.high_rank = it->second;
    if (lc2 != lc) {
        if (!z.blocked) {
            std::cerr << "LOW-changing HIGH transition was not blocked\n";
            std::exit(21);
        }
        d.kind = Kind::Cross;
        d.block = h2;
        d.depth = cross_depth(representative, p, low);
        if (d.depth <= 0 || d.depth > 15) {
            std::cerr << "cross depth overflow " << d.depth << '\n';
            std::exit(22);
        }
    } else if (z.blocked) {
        d.kind = Kind::Block;
        d.block = h2;
    } else {
        d.kind = Kind::Main;
        const int cv2 = int(mget(z.mate, low));
        d.block = 3 * h2 + cv2;
    }
    return d;
}

static Desc build_block_desc(
    MateID representative, int low, int high, int p,
    const HighRanks& ranks
) {
    const std::uint32_t LM = (1u << (2 * low)) - 1u;
    const std::uint32_t HM = (1u << (2 * high)) - 1u;
    const std::uint32_t lc = std::uint32_t(representative) & LM;
    const MateID z = blocked_exclude(representative, p);
    if ((std::uint32_t(z) & LM) != lc) {
        std::cerr << "blocked descriptor unexpectedly changed LOW\n";
        std::exit(23);
    }
    const std::uint32_t hc2 = std::uint32_t((z >> (2 * (low + 1))) & HM);
    const int h2 = end_height(hc2, high);
    const auto it = ranks.rank[h2].find(hc2);
    if (it == ranks.rank[h2].end()) {
        std::cerr << "blocked destination HIGH code missing\n";
        std::exit(24);
    }
    Desc d;
    d.kind = Kind::Main;
    d.block = 3 * h2 + int(mget(z, low));
    d.high_rank = it->second;
    return d;
}

static bool same_result(MateID want, bool want_blocked, MateID got, bool got_blocked) {
    return want_blocked == got_blocked && want == got;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : 5;
    const int high = W - 1 - low;
    if (W < 4 || W > 14 || low < 1 || high < 1 || low >= 16 || high >= 16) {
        std::cerr << "usage: factor_highdesc_semantics [W<=14] [LOW]\n";
        return 1;
    }

    const auto high_all = build_high_all(high);
    const auto low_all = build_low_all(low);
    const HighRanks ranks = make_high_ranks(high_all);
    const std::uint32_t LM = (1u << (2 * low)) - 1u;

    std::uint64_t observations = 0;
    std::uint64_t compared = 0;
    std::uint64_t crosses = 0;
    std::uint64_t invalid = 0;

    for (int p = W - 1; p >= low + 1; --p) {
        // Main factor blocks: (ending HIGH height, center value).
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs >= int(low_all.size()) || low_all[hs].empty()) continue;
                const std::uint32_t rep_low = low_all[hs][0];
                for (std::uint32_t hc : high_all[he]) {
                    const MateID rep = MateID(rep_low)
                        | (MateID(cv) << (2 * low))
                        | (MateID(hc) << (2 * (low + 1)));
                    const Desc d = build_main_desc(rep, W, low, high, p, ranks);
                    ++observations;
                    if (d.kind == Kind::Cross) ++crosses;
                    if (d.kind == Kind::Invalid) ++invalid;

                    for (std::uint32_t lc : low_all[hs]) {
                        const MateID m = MateID(lc)
                            | (MateID(cv) << (2 * low))
                            | (MateID(hc) << (2 * (low + 1)));
                        const IncludeResult exact = include_horizontal(m, W, p);
                        if (d.kind == Kind::Invalid) {
                            if (exact.valid) {
                                std::cerr << "descriptor INVALID mismatch W=" << W
                                          << " p=" << p << " he=" << he << " cv=" << cv << '\n';
                                return 30;
                            }
                            ++compared;
                            continue;
                        }
                        if (!exact.valid) {
                            std::cerr << "descriptor valid but exact invalid\n";
                            return 31;
                        }

                        const int h2 = d.kind == Kind::Main ? d.block / 3 : d.block;
                        if (h2 < 0 || h2 >= int(high_all.size()) ||
                            d.high_rank >= high_all[h2].size()) {
                            std::cerr << "descriptor target HIGH rank out of range\n";
                            return 32;
                        }
                        const std::uint32_t hc2 = high_all[h2][d.high_rank];
                        MateID got = 0;
                        bool got_blocked = false;
                        if (d.kind == Kind::Main) {
                            const int cv2 = d.block % 3;
                            got = MateID(lc)
                                | (MateID(cv2) << (2 * low))
                                | (MateID(hc2) << (2 * (low + 1)));
                        } else if (d.kind == Kind::Block) {
                            got = MateID(lc) | (MateID(hc2) << (2 * low));
                            got_blocked = true;
                        } else {
                            const std::uint32_t lc2 = flip_low(lc, low, d.depth);
                            if (lc2 == 0xffffffffu) {
                                std::cerr << "CROSS low flip failed for legal source\n";
                                return 33;
                            }
                            got = MateID(lc2 & LM) | (MateID(hc2) << (2 * low));
                            got_blocked = true;
                        }
                        if (!same_result(exact.mate, exact.blocked, got, got_blocked)) {
                            std::cerr << "main descriptor semantic mismatch W=" << W
                                      << " p=" << p << " he=" << he << " cv=" << cv
                                      << " kind=" << int(d.kind) << '\n';
                            return 34;
                        }
                        ++compared;
                    }
                }
            }
        }

        // Blocked factor blocks: one block per ending HIGH height.
        for (int h = 0; h <= high + 1; ++h) {
            if (h >= int(low_all.size()) || low_all[h].empty()) continue;
            const std::uint32_t rep_low = low_all[h][0];
            for (std::uint32_t hc : high_all[h]) {
                const MateID rep = MateID(rep_low) | (MateID(hc) << (2 * low));
                const Desc d = build_block_desc(rep, low, high, p, ranks);
                for (std::uint32_t lc : low_all[h]) {
                    const MateID m = MateID(lc) | (MateID(hc) << (2 * low));
                    const MateID exact = blocked_exclude(m, p);
                    const int h2 = d.block / 3;
                    const int cv2 = d.block % 3;
                    if (h2 < 0 || h2 >= int(high_all.size()) ||
                        d.high_rank >= high_all[h2].size()) {
                        std::cerr << "blocked descriptor target out of range\n";
                        return 35;
                    }
                    const std::uint32_t hc2 = high_all[h2][d.high_rank];
                    const MateID got = MateID(lc)
                        | (MateID(cv2) << (2 * low))
                        | (MateID(hc2) << (2 * (low + 1)));
                    if (got != exact) {
                        std::cerr << "blocked descriptor semantic mismatch W=" << W
                                  << " p=" << p << " h=" << h << '\n';
                        return 36;
                    }
                    ++compared;
                }
            }
        }
    }

    std::cout << "factor-highdesc-semantics OK W=" << W
              << " low=" << low << " high=" << high
              << " observations=" << observations
              << " compared=" << compared
              << " cross=" << crosses
              << " invalid=" << invalid << '\n';
    return 0;
}
