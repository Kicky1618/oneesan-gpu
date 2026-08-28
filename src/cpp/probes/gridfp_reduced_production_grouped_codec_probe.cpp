#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_multigpu_owner_probe_main_unused
#include "gridfp_reduced_production_multigpu_owner_probe.cpp"
#pragma pop_macro("main")

namespace {

struct OwnerPlan {
    std::vector<Rank> begin;
    std::vector<Rank> size;
};

OwnerPlan make_owner_plan(int W, int K, int ngpu) {
    const int L = K + 2;
    const int O = W - L;
    const Rank total = total_grouped_states(L, O);
    OwnerPlan plan;
    plan.begin.assign(static_cast<std::size_t>(ngpu), std::numeric_limits<Rank>::max());
    plan.size.assign(static_cast<std::size_t>(ngpu), 0);

    Rank prefix = 0;
    int last_owner = -1;
    for (int r = 0; r <= O; ++r) {
        const Rank g = fixed_outer_group_size(L, r);
        const Rank count = choose_u64(O, r);
        for (Rank sr = 0; sr < count; ++sr) {
            const Rank base = prefix + sr * g;
            const Rank midpoint = base + g / 2;
            int owner = int((__uint128_t(midpoint) * ngpu) / total);
            if (owner >= ngpu) owner = ngpu - 1;
            if (owner < last_owner) fail("owner assignment not monotone");
            last_owner = owner;
            if (plan.begin[static_cast<std::size_t>(owner)] == std::numeric_limits<Rank>::max())
                plan.begin[static_cast<std::size_t>(owner)] = base;
            plan.size[static_cast<std::size_t>(owner)] += g;
        }
        prefix += count * g;
    }
    if (prefix != total) fail("owner plan total");
    for (int g = 0; g < ngpu; ++g)
        if (plan.begin[static_cast<std::size_t>(g)] == std::numeric_limits<Rank>::max())
            plan.begin[static_cast<std::size_t>(g)] = 0;
    return plan;
}

Rank rank_bits_mask(std::uint32_t mask, int len, int ones) {
    Rank rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        const int rem = len - pos - 1;
        rank += choose_u64(rem, left);
        --left;
    }
    if (left != 0) fail("group local support rank");
    return rank;
}

Rank group_local_sector_offset(int L, int outer_ones, int local_ones) {
    __uint128_t off = 0;
    for (int l = 0; l < local_ones; ++l) {
        const int occupied = outer_ones + l;
        if (!(occupied & 1)) continue;
        const Rank pc = catalan((occupied + 1) / 2);
        off += __uint128_t(choose_u64(L, l) + choose_u64(L - 2, l - 1)) * pc;
    }
    return static_cast<Rank>(off);
}

struct GroupedRank {
    int owner = -1;
    Rank local = 0;
    Rank group_global_base = 0;
    Rank within_group = 0;
};

GroupedRank grouped_rank(
    Key k,
    const ProductionFactorTables& tables,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const OwnerPlan& plan
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = reverse ? tile_start + K : tile_start;
    if (lo < 0 || hi >= W || hi - lo + 1 != L) fail("grouped rank tile window");

    const MateID full = embed_full(k, W, q, reverse);
    const std::uint32_t support = occupancy_mask(full, W);
    const std::uint32_t outer = compress_outside_window(support, W, lo, hi);
    const int outer_ones = __builtin_popcount(outer);
    const Rank group = fixed_outer_group_size(L, outer_ones);
    const Rank sr_outer = support_rank_fixed(outer, O, outer_ones);
    const Rank prefix = group_prefix_before_r(L, O, outer_ones);
    const Rank group_base = prefix + sr_outer * group;
    const int owner = weighted_owner(outer, L, O, ngpu);

    std::uint32_t local_mask = 0;
    int local_ones = 0;
    for (int bit = lo; bit <= hi; ++bit) {
        if ((support >> bit) & 1u) {
            local_mask |= std::uint32_t(1) << (bit - lo);
            ++local_ones;
        }
    }
    const int occupied = outer_ones + local_ones;
    if (!(occupied & 1)) fail("grouped rank occupied parity");
    const Rank pc = catalan((occupied + 1) / 2);
    const Rank pr = tables.primitive_rank(full, W);
    Rank within = group_local_sector_offset(L, outer_ones, local_ones);

    if (!k.blocked) {
        const Rank sr = rank_bits_mask(local_mask, L, local_ones);
        within += sr * pc + pr;
    } else {
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_bit = reverse ? q : q - 1;
        if (((support >> missing_bit) & 1u) != 0 || ((support >> fixed_bit) & 1u) == 0)
            fail("grouped blocked fixed support");
        const int missing_pos = missing_bit - lo;
        const int fixed_pos = fixed_bit - lo;
        if (missing_pos < 0 || missing_pos >= L || fixed_pos < 0 || fixed_pos >= L)
            fail("grouped blocked fixed bits outside tile");
        std::uint32_t compact = 0;
        int cp = 0;
        for (int pos = 0; pos < L; ++pos) {
            if (pos == missing_pos || pos == fixed_pos) continue;
            if ((local_mask >> pos) & 1u) compact |= std::uint32_t(1) << cp;
            ++cp;
        }
        const Rank sr = rank_bits_mask(compact, L - 2, local_ones - 1);
        within += choose_u64(L, local_ones) * pc + sr * pc + pr;
    }

    if (within >= group) fail("grouped within rank range");
    const Rank begin = plan.begin[static_cast<std::size_t>(owner)];
    const Rank local = (group_base - begin) + within;
    if (local >= plan.size[static_cast<std::size_t>(owner)]) fail("grouped owner local range");
    return GroupedRank{owner, local, group_base, within};
}

void verify_grouped_tile(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int tile_start,
    int K,
    bool reverse,
    int ngpu
) {
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    std::vector<std::vector<std::uint8_t>> seen(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g)
        seen[static_cast<std::size_t>(g)].assign(
            static_cast<std::size_t>(plan.size[static_cast<std::size_t>(g)]), 0);

    // Every quotient position in the tile is a different permutation of the
    // same per-owner storage capacity. Verify a full bijection at each q.
    int q = tile_start;
    for (int step = 0; step <= K; ++step) {
        for (auto& v : seen) std::fill(v.begin(), v.end(), 0);
        Rank count = 0;
        for (Key k : layout(main, block, q)) {
            const GroupedRank r = grouped_rank(
                k, tables, W, q, reverse, tile_start, K, ngpu, plan);
            auto& cell = seen[static_cast<std::size_t>(r.owner)][static_cast<std::size_t>(r.local)];
            if (cell++) fail("grouped local rank collision");
            ++count;
        }
        if (count != tables.size()) fail("grouped local rank count");
        for (int g = 0; g < ngpu; ++g) {
            for (std::uint8_t x : seen[static_cast<std::size_t>(g)])
                if (x != 1) fail("grouped local rank hole");
        }
        if (step == K) break;
        q += reverse ? 1 : -1;
    }

    // Every transition remains on one owner, but local offsets may permute.
    q = tile_start;
    for (int step = 0; step < K; ++step) {
        const int next = reverse ? q + 1 : q - 1;
        for (Key src : layout(main, block, q)) {
            const GroupedRank sr = grouped_rank(
                src, tables, W, q, reverse, tile_start, K, ngpu, plan);
            for (const auto& [dst, c] : reduced_step_basis(src, W, q, reverse)) {
                if (c != 1 && c != -1) fail("grouped transition coefficient");
                const GroupedRank dr = grouped_rank(
                    dst, tables, W, next, reverse, tile_start, K, ngpu, plan);
                if (dr.owner != sr.owner) fail("grouped transition crosses owner");
            }
        }
        q = next;
    }
}

void print_w28_grouped_plan(int K, int ngpu) {
    const OwnerPlan p = make_owner_plan(28, K, ngpu);
    Rank sum = 0;
    for (int g = 0; g < ngpu; ++g) {
        sum += p.size[static_cast<std::size_t>(g)];
        std::cout << "W=28 K=" << K
                  << " gpu=" << g
                  << " owner_begin=" << p.begin[static_cast<std::size_t>(g)]
                  << " local_states=" << p.size[static_cast<std::size_t>(g)]
                  << " local_u32_GiB=" << double(p.size[static_cast<std::size_t>(g)]) * 4.0 /
                                           double(1ULL << 30)
                  << "\n";
    }
    if (sum != 473397057701ULL) fail("W28 grouped plan total");
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 11 || ngpu < 2 || ngpu > 16) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 7; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            for (int start = K + 2; start <= W - 1; ++start)
                verify_grouped_tile(words[W], words[W - 1], W, start, K, false, ngpu);
            for (int start = 1; start <= W - K - 2; ++start)
                verify_grouped_tile(words[W], words[W - 1], W, start, K, true, ngpu);
            std::cout << "W=" << W
                      << " K=" << K
                      << " grouped_codec_bijection=1"
                      << " owner_local_rank_dense=1"
                      << " intra_tile_owner_crossings=0"
                      << " forward=OK reverse=OK\n";
        }
    }

    print_w28_grouped_plan(11, 8);
    print_w28_grouped_plan(14, 8);
    std::cout << "ALL_OK production_grouped_multigpu_codec=1"
              << " owner_begin_table_u64_per_tile=8\n";
    return 0;
}
