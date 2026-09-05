#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

#include <limits>

static int bits_u32(uint32_t x) {
    int b = 0;
    do { ++b; x >>= 1; } while (x);
    return b;
}
static int bits_signed(int lo, int hi) {
    for (int b = 1; b < 31; ++b) {
        const int mn = -(1 << (b - 1));
        const int mx = (1 << (b - 1)) - 1;
        if (lo >= mn && hi <= mx) return b;
    }
    return 31;
}
static uint32_t abstract_off(int n, int h) {
    uint32_t off = 0;
    for (int nn = 0; nn <= L; ++nn) {
        for (int hh = 0; hh < 16; ++hh) {
            if (nn == n && hh == h) return off;
            off += ballot_suffix(nn, hh);
        }
    }
    return off;
}
static uint64_t pack56(uint32_t start, uint32_t lcount, int delta,
                       uint32_t count, uint32_t aoff) {
    if (start >= (1u << 15) || lcount >= (1u << 3) ||
        delta < -(1 << 14) || delta >= (1 << 14) ||
        count == 0u || count >= (1u << 10) || aoff >= (1u << 13))
        std::exit(20);
    return uint64_t(start) |
           (uint64_t(lcount) << 15) |
           (uint64_t(uint32_t(delta) & 0x7fffu) << 18) |
           (uint64_t(count) << 33) |
           (uint64_t(aoff) << 43);
}
static int unpack_delta(uint64_t x) {
    uint32_t z = uint32_t((x >> 18) & 0x7fffu);
    if (z & 0x4000u) z |= 0xffff8000u;
    return int(int32_t(z));
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;
    std::vector<int32_t> base(size_t(NG) * S * LM, -1);
    auto bref = [&](int g, int h, uint32_t m) -> int32_t& {
        return base[(size_t(g) * S + size_t(h)) * LM + m];
    };
    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(owner[m]);
            const uint32_t cnt = uint32_t(f.low_mask_h[size_t(m) * S + size_t(h)].size());
            if (cnt) bref(g, h, m) = int32_t(next[g]);
            next[g] += cnt;
        }
    }

    uint32_t max_start = 0, max_count = 0, max_lcount = 0, max_aoff = 0;
    int min_delta = std::numeric_limits<int>::max();
    int max_delta = std::numeric_limits<int>::min();
    uint64_t groups = 0, exact = 0;
    for (int g = 0; g < NG; ++g) {
        for (int h = 0; h <= L + 1; ++h) {
            for (uint32_t m = 0; m < LM; ++m) {
                if (owner[m] != g) continue;
                const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
                if (v.empty()) continue;
                const uint32_t start = uint32_t(bref(g, h, m));
                const uint32_t count = uint32_t(v.size());
                const uint32_t n = uint32_t(__builtin_popcount(m));
                if (n < uint32_t(h) || ((n - uint32_t(h)) & 1u)) return 2;
                const uint32_t lcount = (n - uint32_t(h)) >> 1;
                const uint32_t aoff = abstract_off(int(n), h);
                int delta = 0;
                if (h + 2 <= L + 1) {
                    const int32_t b = bref(g, h + 2, m);
                    if (b >= 0) delta = int(b) - int(start);
                }
                max_start = std::max(max_start, start);
                max_count = std::max(max_count, count);
                max_lcount = std::max(max_lcount, lcount);
                max_aoff = std::max(max_aoff, aoff);
                min_delta = std::min(min_delta, delta);
                max_delta = std::max(max_delta, delta);

                const uint64_t p = pack56(start, lcount, delta, count, aoff);
                const uint32_t us = uint32_t(p & 0x7fffu);
                const uint32_t ul = uint32_t((p >> 15) & 0x7u);
                const int ud = unpack_delta(p);
                const uint32_t uc = uint32_t((p >> 33) & 0x3ffu);
                const uint32_t ua = uint32_t((p >> 43) & 0x1fffu);
                const uint32_t rebuilt_n = uint32_t(h) + 2u * ul;
                if (us != start || ul != lcount || ud != delta || uc != count ||
                    ua != aoff || rebuilt_n != n || (p >> 56) != 0u)
                    return 3;
                ++groups;
                ++exact;
            }
        }
    }

    if (groups != 69632ull || exact != groups || max_start != 29113u ||
        max_count != 1001u || max_lcount != 7u || max_aoff != 7059u ||
        min_delta != -12969 || max_delta != 14873)
        return 4;

    std::cout << "gridfp-rankformula-nometa-group56 OK"
              << " groups=" << groups
              << " max_start=" << max_start << " start_bits=" << bits_u32(max_start)
              << " max_lcount=" << max_lcount << " lcount_bits=" << bits_u32(max_lcount)
              << " min_delta=" << min_delta << " max_delta=" << max_delta
              << " delta_signed_bits=" << bits_signed(min_delta, max_delta)
              << " max_count=" << max_count << " count_bits=" << bits_u32(max_count)
              << " max_abstract_off=" << max_aoff << " abstract_off_bits=" << bits_u32(max_aoff)
              << " packed_bits=56 spare_bits=8 exact=" << exact
              << " n_reconstructed_from_h_lcount=1 self_group_index_removed=1"
              << " coop_leader_gi_register=1\n";
    return 0;
}
