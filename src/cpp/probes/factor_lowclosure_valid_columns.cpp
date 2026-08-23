#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

static std::vector<std::uint32_t> high_representatives(int high) {
    std::vector<std::uint32_t> rep(high + 2, 0xffffffffu);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            if (h >= 0 && h < int(rep.size()) && rep[h] == 0xffffffffu)
                rep[h] = code;
            return;
        }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1,
                    code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 0);
    return rep;
}

static std::vector<std::vector<std::uint32_t>> low_codes_by_start(int low) {
    std::vector<std::vector<std::uint32_t>> out(low + 1);
    for (int h0 = 0; h0 <= low; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) out[h0].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h) self(self, pos - 1, h - 1,
                        code | (1u << (2 * pos)));
            self(self, pos - 1, h + 1,
                 code | (2u << (2 * pos)));
        };
        rec(rec, low - 1, h0, 0);
    }
    return out;
}

static bool is_closure(MateValuePair w) {
    return w == LL || w == RR || w == RL;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || low >= 16 || high < 1 || high >= 16)
        return 1;

    const auto high_rep = high_representatives(high);
    const auto low_codes = low_codes_by_start(low);
    U64 expected = 0;

    for (int p = low; p >= 1; --p) {
        U64 closure = 0, invalid = 0;
        for (int he = 0; he <= high + 1; ++he) {
            if (he >= int(high_rep.size()) || high_rep[he] == 0xffffffffu) continue;
            const std::uint32_t hc = high_rep[he];
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                for (std::uint32_t lc : low_codes[hs]) {
                    const MateID m = MateID(lc)
                        | (MateID(cv) << (2 * low))
                        | (MateID(hc) << (2 * (low + 1)));
                    if (!is_closure(mpair(m, p))) continue;
                    ++closure;
                    if (!include_horizontal(m, W, p).valid) ++invalid;
                }
            }
        }
        if (invalid) {
            std::cerr << "invalid LOW closure columns p=" << p
                      << " invalid=" << invalid << '\n';
            return 2;
        }
        if (!expected) expected = closure;
        else if (closure != expected) {
            std::cerr << "valid LOW closure count depends on p: "
                      << closure << " vs " << expected << '\n';
            return 3;
        }
        std::cout << "p=" << p << " valid_closure_cols=" << closure << '\n';
    }

    if (W == 28 && low == 14 && expected != 1088282ULL) {
        std::cerr << "unexpected n=27 valid LOW closure count=" << expected << '\n';
        return 4;
    }
    std::cout << "factor-lowclosure-valid-columns OK W=" << W
              << " low=" << low << " high=" << high
              << " cols_per_p=" << expected << '\n';
    return 0;
}
