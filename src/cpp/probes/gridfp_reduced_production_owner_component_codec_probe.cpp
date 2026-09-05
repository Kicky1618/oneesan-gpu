#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <set>

namespace {

struct SrRange {
    Rank begin = 0;
    Rank end = 0;
};

__int128 ceil_div_signed(__int128 a, __int128 b) {
    if (b <= 0) fail("ceil div denominator");
    if (a >= 0) return (a + b - 1) / b;
    return -((-a) / b);
}

Rank component_group_size(int L, int outer_ones) {
    __uint128_t total = 0;
    for (int local = 0; local <= L - 1; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        const Rank pc = catalan((occupied + 1) / 2);
        const Rank supports = choose_u64(L - 1, local) - choose_u64(L - 3, local);
        total += __uint128_t(supports) * pc;
    }
    return static_cast<Rank>(total);
}

SrRange owner_sr_range(int L, int O, int r, int owner, int ngpu) {
    const Rank count = choose_u64(O, r);
    const Rank group = fixed_outer_group_size(L, r);
    const Rank total = total_grouped_states(L, O);
    const Rank prefix = group_prefix_before_r(L, O, r);
    const __int128 base = __int128(prefix) + __int128(group / 2);
    const __int128 t0 = ceil_div_signed(__int128(owner) * total, ngpu);
    const __int128 t1 = ceil_div_signed(__int128(owner + 1) * total, ngpu);
    __int128 a = ceil_div_signed(t0 - base, group);
    __int128 b = ceil_div_signed(t1 - base, group);
    a = std::max<__int128>(0, std::min<__int128>(count, a));
    b = std::max<__int128>(0, std::min<__int128>(count, b));
    if (b < a) fail("owner sr range order");
    return SrRange{static_cast<Rank>(a), static_cast<Rank>(b)};
}

std::uint32_t support_unrank_host(int len, int ones, Rank rank) {
    if (rank >= choose_u64(len, ones)) fail("support unrank host range");
    std::uint32_t mask = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
        const Rank z = choose_u64(rem, left);
        if (rank < z) continue;
        rank -= z;
        mask |= std::uint32_t(1) << pos;
        --left;
    }
    if (left != 0 || rank != 0) fail("support unrank host final");
    return mask;
}

std::uint32_t conditioned_support_unrank_host(
    int len, int ones, int mark0, int mark1, Rank rank
) {
    const Rank total = choose_u64(len, ones) - choose_u64(len - 2, ones);
    if (rank >= total) fail("conditioned support range");
    std::uint32_t support = 0;
    int left = ones;
    bool seen_mark = false;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
        const int future_marks = (mark0 > pos ? 1 : 0) + (mark1 > pos ? 1 : 0);
        Rank zero_count = choose_u64(rem, left);
        if (!seen_mark) zero_count -= choose_u64(rem - future_marks, left);
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
        if (pos == mark0 || pos == mark1) seen_mark = true;
    }
    if (left != 0 || rank != 0 || !seen_mark) fail("conditioned support final");
    return support;
}

MateID primitive_unrank_host(int occupied, Rank rank) {
    ProductionFactorTables t(std::max(occupied, 1));
    const auto p = t.primitive_unrank(occupied, rank);
    MateID m = 0;
    for (int pos = 0; pos < occupied; ++pos)
        m |= MateID(p[static_cast<std::size_t>(pos)]) << (2 * (occupied - 1 - pos));
    return m;
}

MateID materialize_label_host(std::uint32_t full_support, int W, int missing_bit, Rank pr) {
    std::uint32_t lr_support = 0;
    int lp = 0;
    int occupied = 0;
    for (int pos = 0; pos < W; ++pos) {
        const int physical_bit = W - 1 - pos;
        if (physical_bit == missing_bit) continue;
        if ((full_support >> physical_bit) & 1u) {
            lr_support |= std::uint32_t(1) << lp;
            ++occupied;
        }
        ++lp;
    }
    ProductionFactorTables t(W);
    const auto primitive = t.primitive_unrank(occupied, pr);
    MateID label = 0;
    int q = 0;
    for (int pos = 0; pos < W - 1; ++pos) {
        if (((lr_support >> pos) & 1u) == 0) continue;
        const int bit = W - 2 - pos;
        label |= MateID(primitive[static_cast<std::size_t>(q++)]) << (2 * bit);
    }
    if (q != occupied) fail("materialize label occupied");
    return label;
}

struct OwnerComponentCodec {
    int W;
    int tile_start;
    int K;
    bool reverse;
    int owner;
    int ngpu;

    int L() const { return K + 2; }
    int O() const { return W - L(); }
    int lo() const { return reverse ? tile_start - 1 : tile_start - K - 1; }
    int hi() const { return lo() + L() - 1; }

    Rank size() const {
        Rank z = 0;
        for (int r = 0; r <= O(); ++r) {
            const SrRange a = owner_sr_range(L(), O(), r, owner, ngpu);
            z += (a.end - a.begin) * component_group_size(L(), r);
        }
        return z;
    }

    MateID label_unrank(int p, Rank rank) const {
        int r = -1;
        SrRange range{};
        Rank group_count = 0;
        for (int t = 0; t <= O(); ++t) {
            const SrRange a = owner_sr_range(L(), O(), t, owner, ngpu);
            const Rank cg = component_group_size(L(), t);
            const Rank n = (a.end - a.begin) * cg;
            if (rank < n) { r = t; range = a; group_count = cg; break; }
            rank -= n;
        }
        if (r < 0 || group_count == 0) fail("owner component outer sector");
        const Rank outer_sr = range.begin + rank / group_count;
        Rank within = rank % group_count;
        const std::uint32_t outer = support_unrank_host(O(), r, outer_sr);

        int local_ones = -1;
        Rank local_sr = 0, pr = 0;
        for (int l = 0; l <= L() - 1; ++l) {
            const int occupied = r + l;
            if (!(occupied & 1)) continue;
            const Rank pc = catalan((occupied + 1) / 2);
            const Rank supports = choose_u64(L() - 1, l) - choose_u64(L() - 3, l);
            const Rank n = supports * pc;
            if (within < n) {
                local_ones = l;
                local_sr = within / pc;
                pr = within % pc;
                break;
            }
            within -= n;
        }
        if (local_ones < 0) fail("owner component local sector");

        const int missing = reverse ? p - 1 : p;
        const int mark_a = reverse ? p : p - 1;
        const int mark_b = reverse ? p + 1 : p - 2;
        int mark0 = -1, mark1 = -1, avail = 0;
        std::vector<int> local_bits;
        local_bits.reserve(static_cast<std::size_t>(L() - 1));
        for (int bit = lo(); bit <= hi(); ++bit) {
            if (bit == missing) continue;
            if (bit == mark_a) mark0 = avail;
            if (bit == mark_b) mark1 = avail;
            local_bits.push_back(bit);
            ++avail;
        }
        if (mark0 < 0 || mark1 < 0) fail("owner component marks outside window");
        const std::uint32_t local = conditioned_support_unrank_host(
            L() - 1, local_ones, mark0, mark1, local_sr);

        std::uint32_t full = 0;
        int oq = 0;
        for (int bit = 0; bit < W; ++bit) {
            if (bit >= lo() && bit <= hi()) continue;
            if ((outer >> oq) & 1u) full |= std::uint32_t(1) << bit;
            ++oq;
        }
        for (int j = 0; j < L() - 1; ++j)
            if ((local >> j) & 1u) full |= std::uint32_t(1) << local_bits[static_cast<std::size_t>(j)];
        if ((full >> missing) & 1u) fail("owner component missing occupied");
        return materialize_label_host(full, W, missing, pr);
    }
};

void verify_owner_codec(int W, int K, bool reverse, int ngpu) {
    const int tile_start = reverse ? 1 : W - 1;
    const int begin = reverse ? 1 : W - 1;
    const int end = reverse ? std::min(W - 3, K) : std::max(3, W - K);
    const int delta = reverse ? 1 : -1;
    const auto labels = gen_words(W - 1);
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);

    for (int p = begin;; p += delta) {
        std::set<MateID> seen;
        Rank total = 0;
        for (int g = 0; g < ngpu; ++g) {
            OwnerComponentCodec codec{W, tile_start, K, reverse, g, ngpu};
            const Rank n = codec.size();
            total += n;
            for (Rank r = 0; r < n; ++r) {
                const MateID v = codec.label_unrank(p, r);
                if (!seen.insert(v).second) fail("owner component duplicate label");
                const int other = reverse ? p : p - 2;
                if (mget(v, p - 1) == N && mget(v, other) == N)
                    fail("owner component ineligible label");
                const Key seed = reverse
                    ? (mget(v,p-1)!=N ? Key{true,v} : Key{false,blocked_exclude_reverse(v,W,p)})
                    : (mget(v,p-1)!=N ? Key{true,v} : Key{false,blocked_exclude(v,p)});
                const GroupedRank gr = grouped_rank(
                    seed, tables, W, p, reverse, tile_start, K, ngpu, plan);
                if (gr.owner != g) fail("owner component seed owner mismatch");
            }
        }
        const Rank want = labels.size() - gen_words(W - 3).size();
        if (total != want || seen.size() != want) fail("owner component total coverage");
        if (p == end) break;
    }
}

void print_w28_load() {
    constexpr int W = 28, K = 13, ngpu = 8;
    Rank total = 0, lo = std::numeric_limits<Rank>::max(), hi = 0;
    for (int g = 0; g < ngpu; ++g) {
        OwnerComponentCodec codec{W, W - 1, K, false, g, ngpu};
        const Rank n = codec.size();
        total += n;
        lo = std::min(lo, n);
        hi = std::max(hi, n);
        std::cout << "W=28 K=13 gpu=" << g
                  << " local_components=" << n
                  << " fraction=" << double(n) / 118389089432.0
                  << " table_bytes=0\n";
    }
    if (total != 118389089432ULL) fail("W28 owner component total");
    std::cout << "W=28 K=13 component_min=" << lo
              << " component_max=" << hi
              << " spread_fraction=" << double(hi-lo)/double(total)
              << " duplicate_component_scans=0"
              << " global_components=" << total << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 12 || ngpu < 2 || ngpu > 16) return 2;

    for (int W = 7; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            verify_owner_codec(W, K, false, ngpu);
            verify_owner_codec(W, K, true, ngpu);
            std::cout << "W=" << W << " K=" << K
                      << " owner_local_component_codec=OK"
                      << " forward=OK reverse=OK"
                      << " component_table_bytes=0\n";
        }
    }
    print_w28_load();
    std::cout << "ALL_OK production_owner_local_component_codec=1\n";
    return 0;
}
