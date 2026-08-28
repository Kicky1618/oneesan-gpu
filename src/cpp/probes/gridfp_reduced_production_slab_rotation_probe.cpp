#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_slab_cycle_probe_main_unused
#include "gridfp_reduced_production_slab_cycle_probe.cpp"
#pragma pop_macro("main")

#include <numeric>

namespace {

std::uint32_t rotate_support(std::uint32_t x, int len, int shift) {
    if (len <= 0 || len >= 32) fail("slab rotate length");
    shift %= len;
    if (shift < 0) shift += len;
    const std::uint32_t mask = (std::uint32_t(1) << len) - 1u;
    x &= mask;
    if (!shift) return x;
    return ((x << shift) & mask) | (x >> (len - shift));
}

std::uint32_t erase_support_bits(
    std::uint32_t full,
    int W,
    int a,
    int b
) {
    if (a > b) std::swap(a, b);
    std::uint32_t out = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == a || bit == b) continue;
        if ((full >> bit) & 1u) out |= std::uint32_t(1) << q;
        ++q;
    }
    if (q != W - 2) fail("slab erase support width");
    return out;
}

struct RotationStats {
    Rank slabs = 0;
    Rank main_slabs = 0;
    Rank blocked_slabs = 0;
    Rank main_fixed = 0;
    Rank blocked_fixed = 0;
    Rank max_values = 0;
};

RotationStats verify_rotation_formula(int W, bool reverse, int ngpu) {
    if ((W & 1) || W < 8) fail("slab rotation expects even W>=8");
    const int K = W / 2 - 1;
    const int q = W / 2;
    const int old_start = reverse ? 1 : W - 1;
    const int new_start = q;
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    ProductionFactorTables tables(W);

    struct Bases { Rank old_base = 0; Rank new_base = 0; Rank values = 0; };
    std::map<SlabKey, Bases> by_key;
    std::map<Rank, SlabKey> old_occupant;

    for (Key k : layout(gen_words(W), gen_words(W - 1), q)) {
        const MateID full = embed_full(k, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const int occupied = __builtin_popcount(support);
        const Rank pc = catalan((occupied + 1) / 2);
        const Rank pr = tables.primitive_rank(full, W);
        const GroupedRank a = grouped_rank(
            k, tables, W, q, reverse, old_start, K, ngpu, plan);
        const GroupedRank b = grouped_rank(
            k, tables, W, q, reverse, new_start, K, ngpu, plan);
        const Rank old_global = grouped_global_rank(a, plan);
        const Rank new_global = grouped_global_rank(b, plan);
        const SlabKey key{k.blocked, support};
        const Bases want{old_global - pr, new_global - pr, pc};
        auto [it, inserted] = by_key.emplace(key, want);
        if (!inserted && (it->second.old_base != want.old_base ||
                          it->second.new_base != want.new_base ||
                          it->second.values != want.values))
            fail("slab rotation bases depend on primitive rank");
    }
    for (const auto& [key, x] : by_key) {
        if (!old_occupant.emplace(x.old_base, key).second)
            fail("slab rotation duplicate old physical slab");
    }

    RotationStats st;
    st.slabs = by_key.size();
    const int main_shift = reverse ? -K : K;
    const int main_order = W / std::gcd(W, K);
    const int missing = reverse ? q - 1 : q;
    const int fixed = reverse ? q : q - 1;
    if (W - 2 != 2 * K) fail("slab blocked half-turn geometry");

    for (const auto& [key, x] : by_key) {
        const auto oit = old_occupant.find(x.new_base);
        if (oit == old_occupant.end()) fail("slab rotation destination not old slab");
        const SlabKey& next = oit->second;
        st.max_values = std::max(st.max_values, x.values);
        if (!key.blocked) {
            ++st.main_slabs;
            if (next.blocked) fail("slab rotation main changed coordinate type");
            const std::uint32_t expected = rotate_support(key.support, W, main_shift);
            if (next.support != expected)
                fail(std::string(reverse ? "reverse" : "forward") +
                     " main slab is not fixed support rotation W=" + std::to_string(W));
            if (next.support == key.support) ++st.main_fixed;

            std::uint32_t z = key.support;
            for (int step = 0; step < main_order; ++step)
                z = rotate_support(z, W, main_shift);
            if (z != key.support) fail("slab main rotation order formula");
        } else {
            ++st.blocked_slabs;
            if (!next.blocked) fail("slab rotation blocked changed coordinate type");
            if (((key.support >> missing) & 1u) != 0 ||
                ((key.support >> fixed) & 1u) == 0 ||
                ((next.support >> missing) & 1u) != 0 ||
                ((next.support >> fixed) & 1u) == 0)
                fail("slab blocked fixed 10 support condition");
            const std::uint32_t free0 = erase_support_bits(key.support, W, missing, fixed);
            const std::uint32_t free1 = erase_support_bits(next.support, W, missing, fixed);
            // W-2 = 2K, so +K and -K are the same half-turn.
            if (free1 != rotate_support(free0, W - 2, K))
                fail(std::string(reverse ? "reverse" : "forward") +
                     " blocked slab free support is not half rotation W=" + std::to_string(W));
            if (free0 == free1) ++st.blocked_fixed;
            if (rotate_support(free1, W - 2, K) != free0)
                fail("slab blocked rotation is not involution");
        }
    }

    const Rank want_main = Rank(1) << (W - 1);
    const Rank want_blocked = Rank(1) << (W - 3);
    if (st.main_slabs != want_main || st.blocked_slabs != want_blocked ||
        st.slabs != want_main + want_blocked)
        fail("slab rotation closed-form slab count");

    std::cout << "W=" << W
              << " direction=" << (reverse ? "reverse" : "forward")
              << " main_shift=" << main_shift
              << " main_cycle_order_bound=" << main_order
              << " blocked_free_bits=" << W - 2
              << " blocked_shift=" << K
              << " blocked_cycle_order_bound=2"
              << " slabs=" << st.slabs
              << " main_fixed=" << st.main_fixed
              << " blocked_fixed=" << st.blocked_fixed
              << " max_slab_values=" << st.max_values
              << " permutation_table_bytes=0"
              << " visited_bitset_required=0"
              << " support_rotation_formula=OK\n";
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 12 || ngpu < 2 || ngpu > 16) return 2;

    for (int W = 8; W <= maxW; W += 2) {
        const RotationStats f = verify_rotation_formula(W, false, ngpu);
        const RotationStats r = verify_rotation_formula(W, true, ngpu);
        if (f.slabs != r.slabs || f.main_slabs != r.main_slabs ||
            f.blocked_slabs != r.blocked_slabs || f.max_values != r.max_values)
            fail("slab rotation reflection statistics differ");
    }

    constexpr int W = 28;
    constexpr int K = 13;
    constexpr int main_order = W / 1; // gcd(28,13)=1
    constexpr Rank main_slabs = Rank(1) << 27;
    constexpr Rank blocked_slabs = Rank(1) << 25;
    constexpr Rank max_values = 2674440ULL;
    constexpr Rank scratch_bytes = max_values * 4ULL;
    std::cout << "W=28_theory q=14"
              << " forward_main_support_rotate=+13"
              << " reverse_main_support_rotate=-13"
              << " main_cycle_order_bound=" << main_order
              << " blocked_free_support_bits=26"
              << " blocked_support_rotate=13"
              << " blocked_cycle_order_bound=2"
              << " main_slabs=" << main_slabs
              << " blocked_slabs=" << blocked_slabs
              << " max_slab_values=" << max_values
              << " max_u32_scratch_MiB=" << double(scratch_bytes) / double(1ULL << 20)
              << " permutation_table_bytes=0"
              << " visited_bitset_bytes=0"
              << "\n";
    std::cout << "ALL_OK production_support_slab_rotation=1\n";
    return 0;
}
