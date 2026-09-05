#include "../../common/gridfp_closure_pattern10.hpp"
#include "../../common/gridfp_transition_reverse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#ifndef LOW_LUT_K
#define LOW_LUT_K 14
#endif
#ifndef HIGH_LUT_K
#define HIGH_LUT_K 13
#endif
#ifndef TARGET_W
#define TARGET_W (LOW_LUT_K + HIGH_LUT_K + 1)
#endif

using namespace oneesan::gridfp;

static constexpr std::uint32_t pow3_const(int n) {
    return n <= 0 ? 1u : 3u * pow3_const(n - 1);
}

static std::uint32_t low_key(MateID x) {
    std::uint32_t key = 0, w = 1;
    for (int i = 0; i < LOW_LUT_K; ++i, w *= 3u) key += std::uint32_t(mget(x, i)) * w;
    return key;
}

static std::uint32_t high_key(MateID x) {
    std::uint32_t key = 0, w = 1;
    for (int i = 1; i <= HIGH_LUT_K; ++i, w *= 3u) key += std::uint32_t(mget(x, i)) * w;
    return key;
}

static bool low_factor_pos(int pos) { return pos >= 0 && pos < LOW_LUT_K; }
static bool high_factor_pos(int pos) { return pos >= 1 && pos <= HIGH_LUT_K; }
static std::uint32_t low_weight(int pos) { return pow3_const(pos); }
static std::uint32_t high_weight(int pos) { return pow3_const(pos - 1); }

struct Stats {
    std::uint64_t states = 0;
    std::uint64_t sources = 0;
    std::uint32_t max_sources = 0;
    std::int64_t min_delta = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_delta = std::numeric_limits<std::int64_t>::min();
};

static void fail(const char* side, const char* kind, MateID d, int p, int q,
                 MateID x, std::uint32_t base, std::int64_t delta,
                 std::uint32_t got_key, std::uint32_t pred_center, std::uint32_t got_center) {
    std::cerr << "ternary delta mismatch side=" << side << " kind=" << kind
              << " p=" << p << " q=" << q
              << " d=" << d << " source=" << x
              << " base=" << base << " delta=" << delta
              << " predicted_key=" << (std::int64_t(base) + delta)
              << " actual_key=" << got_key
              << " predicted_center=" << pred_center
              << " actual_center=" << got_center << '\n';
    std::exit(2);
}

static void account(Stats& s, std::int64_t delta) {
    ++s.sources;
    s.min_delta = std::min(s.min_delta, delta);
    s.max_delta = std::max(s.max_delta, delta);
}

static void check_low_source(Stats& s, MateID d, int p, MateValuePair pair,
                             int q, MateValue q_new, const char* kind) {
    const std::uint32_t base = low_key(d);
    MateID x = msetpair(d, p, pair);
    if (q >= 0) x = mset(x, q, q_new);

    std::int64_t delta = 0;
    const int a = p - 1, b = p;
    const std::uint32_t pa = std::uint32_t(pair) & 3u;
    const std::uint32_t pb = (std::uint32_t(pair) >> 2) & 3u;
    if (low_factor_pos(a)) delta += std::int64_t(pa) * low_weight(a);
    if (low_factor_pos(b)) delta += std::int64_t(pb) * low_weight(b);
    if (q >= 0 && low_factor_pos(q)) {
        const int oldv = int(mget(d, q));
        delta += std::int64_t(int(q_new) - oldv) * low_weight(q);
    }

    std::uint32_t center = std::uint32_t(mget(d, LOW_LUT_K));
    if (a == LOW_LUT_K) center = pa;
    if (b == LOW_LUT_K) center = pb;
    if (q == LOW_LUT_K) center = std::uint32_t(q_new);

    const std::int64_t predicted = std::int64_t(base) + delta;
    const std::uint32_t got_key = low_key(x);
    const std::uint32_t got_center = std::uint32_t(mget(x, LOW_LUT_K));
    if (predicted < 0 || predicted >= std::int64_t(pow3_const(LOW_LUT_K)) ||
        std::uint32_t(predicted) != got_key || center != got_center) {
        fail("low", kind, d, p, q, x, base, delta, got_key, center, got_center);
    }
    account(s, delta);
}

static void check_high_source(Stats& s, MateID d, int p, MateValuePair pair,
                              int q, MateValue q_new, const char* kind) {
    const std::uint32_t base = high_key(d);
    MateID x = msetpair(d, p, pair);
    if (q >= 0) x = mset(x, q, q_new);

    std::int64_t delta = 0;
    const int a = p - 1, b = p;
    const std::uint32_t pa = std::uint32_t(pair) & 3u;
    const std::uint32_t pb = (std::uint32_t(pair) >> 2) & 3u;
    if (high_factor_pos(a)) delta += std::int64_t(pa) * high_weight(a);
    if (high_factor_pos(b)) delta += std::int64_t(pb) * high_weight(b);
    if (q >= 0 && high_factor_pos(q)) {
        const int oldv = int(mget(d, q));
        delta += std::int64_t(int(q_new) - oldv) * high_weight(q);
    }

    std::uint32_t center = std::uint32_t(mget(d, 0));
    if (a == 0) center = pa;
    if (b == 0) center = pb;
    if (q == 0) center = std::uint32_t(q_new);

    const std::int64_t predicted = std::int64_t(base) + delta;
    const std::uint32_t got_key = high_key(x);
    const std::uint32_t got_center = std::uint32_t(mget(x, 0));
    if (predicted < 0 || predicted >= std::int64_t(pow3_const(HIGH_LUT_K)) ||
        std::uint32_t(predicted) != got_key || center != got_center) {
        fail("high", kind, d, p, q, x, base, delta, got_key, center, got_center);
    }
    account(s, delta);
}

static void check_low(Stats& s, MateID d, int p) {
    const std::uint16_t id = closure_pattern10_encode(d, LOW_LUT_K + 1, p);
    if (id == CLOSURE_PATTERN10_NONE) return;
    std::uint16_t lm = 0, rm = 0;
    closure_pattern10_decode(id, LOW_LUT_K + 1, p, lm, rm);
    ++s.states;
    std::uint32_t before = std::uint32_t(s.sources);
    check_low_source(s, d, p, RL, -1, N, "rl");
    while (lm) {
        int i = __builtin_ctz(unsigned(lm));
        lm = std::uint16_t(lm & (lm - 1));
        check_low_source(s, d, p, LL, p - 2 - i, R, "ll");
    }
    while (rm) {
        int i = __builtin_ctz(unsigned(rm));
        rm = std::uint16_t(rm & (rm - 1));
        check_low_source(s, d, p, RR, p + 1 + i, L, "rr");
    }
    s.max_sources = std::max(s.max_sources, std::uint32_t(s.sources) - before);
}

static void check_high(Stats& s, MateID d, int p) {
    const std::uint16_t id = closure_pattern10_encode(d, HIGH_LUT_K + 1, p);
    if (id == CLOSURE_PATTERN10_NONE) return;
    std::uint16_t lm = 0, rm = 0;
    closure_pattern10_decode(id, HIGH_LUT_K + 1, p, lm, rm);
    ++s.states;
    std::uint32_t before = std::uint32_t(s.sources);
    check_high_source(s, d, p, RL, -1, N, "rl");
    while (lm) {
        int i = __builtin_ctz(unsigned(lm));
        lm = std::uint16_t(lm & (lm - 1));
        check_high_source(s, d, p, LL, p - 2 - i, R, "ll");
    }
    while (rm) {
        int i = __builtin_ctz(unsigned(rm));
        rm = std::uint16_t(rm & (rm - 1));
        check_high_source(s, d, p, RR, p + 1 + i, L, "rr");
    }
    s.max_sources = std::max(s.max_sources, std::uint32_t(s.sources) - before);
}

static std::vector<std::vector<MateID>> enumerate_low_codes() {
    constexpr int K = LOW_LUT_K;
    std::vector<std::vector<MateID>> out(K + 2);
    auto rec = [&](auto&& self, int pos, int h, MateID code, int h0) -> void {
        if (pos < 0) { if (h == 0) out[h0].push_back(code); return; }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, code, h0);
        if (h > 0) self(self, pos - 1, h - 1, code | (MateID(R) << (2 * pos)), h0);
        self(self, pos - 1, h + 1, code | (MateID(L) << (2 * pos)), h0);
    };
    for (int h0 = 0; h0 <= K + 1; ++h0) rec(rec, K - 1, h0, 0, h0);
    return out;
}

static std::vector<std::vector<MateID>> enumerate_high_codes() {
    constexpr int K = HIGH_LUT_K;
    std::vector<std::vector<MateID>> out(K + 2);
    auto rec = [&](auto&& self, int pos, int h, MateID code) -> void {
        if (pos < 0) { if (h >= 0 && h < int(out.size())) out[h].push_back(code); return; }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1, code | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1, code | (MateID(L) << (2 * pos)));
    };
    rec(rec, K - 1, 1, 0);
    return out;
}

int main() {
    static_assert(LOW_LUT_K > 0 && HIGH_LUT_K > 0);
    static_assert(LOW_LUT_K <= 14 && HIGH_LUT_K <= 14);
    static_assert(TARGET_W == LOW_LUT_K + HIGH_LUT_K + 1);

    auto low = enumerate_low_codes();
    auto high = enumerate_high_codes();
    Stats ls, hs;

    // Forward LOW p=1 main-state special case.
    for (int he = 0; he <= HIGH_LUT_K + 1; ++he) {
        for (int c = int(N); c <= int(L); ++c) {
            int h = he + (c == int(L) ? 1 : c == int(R) ? -1 : 0);
            if (h < 0 || h >= int(low.size())) continue;
            for (MateID code : low[h]) check_low(ls, code | (MateID(c) << (2 * LOW_LUT_K)), 1);
        }
    }
    // Forward blocked + reverse LOW share the same reconstructed full partial.
    for (int p = 1; p <= LOW_LUT_K; ++p) {
        for (const auto& bucket : low) for (MateID code : bucket) {
            MateID d = blocked_exclude_reverse(code, LOW_LUT_K + 1, p);
            check_low(ls, d, p);
        }
    }

    // Forward/reverse blocked HIGH.
    for (int p = 1; p <= HIGH_LUT_K; ++p) {
        for (const auto& bucket : high) for (MateID code : bucket) {
            MateID d = blocked_exclude_reverse(code, HIGH_LUT_K + 1, p);
            check_high(hs, d, p);
        }
    }
    // Reverse HIGH edge: center is explicit and factor is shifted by one.
    for (const auto& bucket : high) for (MateID code : bucket) {
        for (int c = int(N); c <= int(L); ++c) check_high(hs, MateID(c) | (code << 2), HIGH_LUT_K);
    }

    std::cout << "gridfp-closure-ternary-delta OK W=" << TARGET_W
              << " low_states=" << ls.states
              << " high_states=" << hs.states
              << " low_sources=" << ls.sources
              << " high_sources=" << hs.sources
              << " low_max_sources=" << ls.max_sources
              << " high_max_sources=" << hs.max_sources
              << " low_delta_min=" << ls.min_delta
              << " low_delta_max=" << ls.max_delta
              << " high_delta_min=" << hs.min_delta
              << " high_delta_max=" << hs.max_delta
              << " payload_only=1 mateid_source_rebuild_required=0"
              << '\n';
    return 0;
}
