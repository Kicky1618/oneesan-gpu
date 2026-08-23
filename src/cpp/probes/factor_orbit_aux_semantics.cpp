#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"
using namespace oneesan::gridfp;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

static U32 occ(U32 code, int len) {
    U32 m = 0;
    for (int i = 0; i < len; ++i) if (((code >> (2*i)) & 3u) != N) m |= 1u << i;
    return m;
}

struct SegTables {
    int len = 0;
    bool high = false;
    std::vector<std::vector<U32>> storage;
    std::vector<std::unordered_map<U32,U32>> rank;
};

static SegTables build_high(int len) {
    SegTables t; t.len = len; t.high = true; t.storage.resize(len+3); t.rank.resize(len+3);
    std::vector<std::vector<U32>> canonical(len+3);
    auto rec = [&](auto&& self, int pos, int h, U32 code) -> void {
        if (pos < 0) { canonical[h].push_back(code); return; }
        self(self, pos-1, h, code);
        if (h > 0) self(self, pos-1, h-1, code | (U32(R) << (2*pos)));
        self(self, pos-1, h+1, code | (U32(L) << (2*pos)));
    };
    rec(rec, len-1, 1, 0);
    for (int h = 0; h <= len+1; ++h) {
        for (U32 mask = 0; mask < (1u<<len); ++mask)
            for (U32 c : canonical[h]) if (occ(c,len) == mask) t.storage[h].push_back(c);
        for (U32 r = 0; r < t.storage[h].size(); ++r) t.rank[h][t.storage[h][r]] = r;
    }
    return t;
}

static SegTables build_low(int len) {
    SegTables t; t.len = len; t.storage.resize(len+3); t.rank.resize(len+3);
    for (int start = 0; start <= len+1; ++start) {
        std::vector<U32> canonical;
        auto rec = [&](auto&& self, int pos, int h, U32 code) -> void {
            if (pos < 0) { if (h == 0) canonical.push_back(code); return; }
            if (h < 0 || h > pos+1) return;
            self(self, pos-1, h, code);
            if (h > 0) self(self, pos-1, h-1, code | (U32(R) << (2*pos)));
            self(self, pos-1, h+1, code | (U32(L) << (2*pos)));
        };
        rec(rec, len-1, start, 0);
        for (U32 mask = 0; mask < (1u<<len); ++mask)
            for (U32 c : canonical) if (occ(c,len) == mask) t.storage[start].push_back(c);
        for (U32 r = 0; r < t.storage[start].size(); ++r) t.rank[start][t.storage[start][r]] = r;
    }
    return t;
}

static int end_height(U32 code, int len) {
    int h = 1;
    for (int p = len-1; p >= 0; --p) {
        const auto v = MateValue((code >> (2*p)) & 3u);
        if (v == R) --h; else if (v == L) ++h;
        if (h < 0) return -1;
    }
    return h;
}

static int center_shift(int cv) { return cv == int(L) ? 1 : cv == int(R) ? -1 : 0; }
static bool is_rep(MateValuePair w) { return w == NN || w == NR || w == NL; }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : 5;
    const int high = W - 1 - low;
    if (W < 4 || W > 14 || low < 1 || high < 1 || low >= 15 || high >= 16) return 1;

    const auto ht = build_high(high);
    const auto lt = build_low(low);
    U64 high_reps = 0, low_reps = 0, p1_pairs = 0;

    // HIGH aux: existing HighDesc is the included branch. Aux stores the other
    // orbit target: old blocked for NN, companion main for NR/NL.
    for (int he = 0; he <= high+1; ++he) for (int cv = 0; cv < 3; ++cv) {
        const int hs = he + center_shift(cv);
        if (hs < 0 || hs > low+1 || ht.storage[he].empty() || lt.storage[hs].empty()) continue;
        const U32 lc = lt.storage[hs].front();
        for (U32 hr = 0; hr < ht.storage[he].size(); ++hr) {
            const U32 hc = ht.storage[he][hr];
            const MateID m = MateID(lc) | (MateID(cv) << (2*low)) | (MateID(hc) << (2*(low+1)));
            for (int p = W-1; p >= low+1; --p) {
                const MateValuePair w = mpair(m,p); if (!is_rep(w)) continue; ++high_reps;
                const MateValuePair cw = w == NN ? LR : w == NR ? RN : LN;
                const MateID companion = msetpair(m,p,cw);
                const MateID dropped = mshrink(m,p);
                const IncludeResult z = include_horizontal(m,W,p);
                if (!z.valid) return 10;
                if (w == NN) {
                    if (z.blocked || z.mate != companion) return 11;
                    const U32 dhc = U32(dropped >> (2*low));
                    const int dh = end_height(dhc,high);
                    if (dh < 0 || !ht.rank[dh].count(dhc)) return 12;
                } else {
                    if (!z.blocked || z.mate != dropped) return 13;
                    const U32 chc = U32((companion >> (2*(low+1))) & ((U64(1)<<(2*high))-1));
                    const int che = end_height(chc,high);
                    const int ccv = int(mget(companion,low));
                    if (che < 0 || 3*che+ccv < 0 || !ht.rank[che].count(chc)) return 14;
                }
            }
        }
    }

    // LOW aux: LowDesc is again the included branch. For p>1 NR/NL aux is the
    // companion main; for NN and p=1 NR/NL aux is old blocked.
    for (int he = 0; he <= high+1; ++he) for (int cv = 0; cv < 3; ++cv) {
        const int hs = he + center_shift(cv);
        if (hs < 0 || hs > low+1 || ht.storage[he].empty() || lt.storage[hs].empty()) continue;
        const U32 hc = ht.storage[he].front();
        for (U32 lr = 0; lr < lt.storage[hs].size(); ++lr) {
            const U32 lc = lt.storage[hs][lr];
            const MateID m = MateID(lc) | (MateID(cv) << (2*low)) | (MateID(hc) << (2*(low+1)));
            for (int p = low; p >= 1; --p) {
                const MateValuePair w = mpair(m,p); if (!is_rep(w)) continue; ++low_reps;
                const MateValuePair cw = w == NN ? LR : w == NR ? RN : LN;
                const MateID companion = msetpair(m,p,cw);
                const MateID dropped = mshrink(m,p);
                const IncludeResult z = include_horizontal(m,W,p);
                if (!z.valid) return 20;
                if (w == NN) {
                    if (z.blocked || z.mate != companion) return 21;
                    const U32 dlc = U32(dropped & ((U64(1)<<(2*low))-1));
                    if (!lt.rank[he].count(dlc)) return 22;
                } else if (p == 1) {
                    ++p1_pairs;
                    if (z.blocked || z.mate != companion) return 23;
                    const U32 dlc = U32(dropped & ((U64(1)<<(2*low))-1));
                    if (!lt.rank[he].count(dlc)) return 24;
                } else {
                    if (!z.blocked || z.mate != dropped) return 25;
                    const U32 clc = U32(companion & ((U64(1)<<(2*low))-1));
                    const int ccv = int(mget(companion,low));
                    const int chs = he + center_shift(ccv);
                    if (chs < 0 || chs > low+1 || !lt.rank[chs].count(clc)) return 26;
                }
                const U32 h2 = z.blocked ? U32(z.mate >> (2*low))
                                         : U32((z.mate >> (2*(low+1))) & ((U64(1)<<(2*high))-1));
                if (h2 != hc) return 27;
            }
        }
    }

    std::cout << "factor-orbit-aux-semantics OK W=" << W << " low=" << low << " high=" << high
              << " high_reps=" << high_reps << " low_reps=" << low_reps
              << " p1_pairs=" << p1_pairs << '\n';
    return 0;
}
