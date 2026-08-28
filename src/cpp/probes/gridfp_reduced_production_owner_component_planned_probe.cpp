#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_owner_component_codec_probe_main_unused
#include "gridfp_reduced_production_owner_component_codec_probe.cpp"
#pragma pop_macro("main")

namespace {

struct PlannedOwnerCodec {
    OwnerComponentCodec base;
    std::vector<Rank> prefix;
    std::vector<Rank> sr_begin;
    std::vector<Rank> component_group;

    PlannedOwnerCodec(OwnerComponentCodec b) : base(b) {
        prefix.assign(static_cast<std::size_t>(base.O() + 2), 0);
        sr_begin.assign(static_cast<std::size_t>(base.O() + 1), 0);
        component_group.assign(static_cast<std::size_t>(base.O() + 1), 0);
        for (int r = 0; r <= base.O(); ++r) {
            const SrRange a = owner_sr_range(base.L(), base.O(), r, base.owner, base.ngpu);
            const Rank cg = component_group_size(base.L(), r);
            sr_begin[static_cast<std::size_t>(r)] = a.begin;
            component_group[static_cast<std::size_t>(r)] = cg;
            prefix[static_cast<std::size_t>(r + 1)] =
                prefix[static_cast<std::size_t>(r)] + (a.end - a.begin) * cg;
        }
        if (prefix.back() != base.size()) fail("planned owner size mismatch");
    }

    static int compact_index(int physical_bit, int lo, int missing) {
        return physical_bit - lo - (physical_bit > missing ? 1 : 0);
    }

    MateID label_unrank(int p, Rank rank) const {
        int r = -1;
        Rank local = 0;
        for (int t = 0; t <= base.O(); ++t) {
            if (rank < prefix[static_cast<std::size_t>(t + 1)]) {
                r = t;
                local = rank - prefix[static_cast<std::size_t>(t)];
                break;
            }
        }
        if (r < 0) fail("planned owner outer rank");
        const Rank cg = component_group[static_cast<std::size_t>(r)];
        if (!cg) fail("planned owner zero component group");
        const Rank outer_sr = sr_begin[static_cast<std::size_t>(r)] + local / cg;
        Rank within = local % cg;
        const std::uint32_t outer = support_unrank_host(base.O(), r, outer_sr);

        int local_ones = -1;
        Rank local_sr = 0, pr = 0;
        for (int l = 0; l <= base.L() - 1; ++l) {
            const int occupied = r + l;
            if (!(occupied & 1)) continue;
            const Rank pc = catalan((occupied + 1) / 2);
            const Rank supports = choose_u64(base.L() - 1, l) - choose_u64(base.L() - 3, l);
            const Rank n = supports * pc;
            if (within < n) {
                local_ones = l;
                local_sr = within / pc;
                pr = within % pc;
                break;
            }
            within -= n;
        }
        if (local_ones < 0) fail("planned owner local rank");

        const int missing = base.reverse ? p - 1 : p;
        const int mark_a = base.reverse ? p : p - 1;
        const int mark_b = base.reverse ? p + 1 : p - 2;
        const int mark0 = compact_index(mark_a, base.lo(), missing);
        const int mark1 = compact_index(mark_b, base.lo(), missing);
        if (mark0 < 0 || mark0 >= base.L() - 1 ||
            mark1 < 0 || mark1 >= base.L() - 1 || mark0 == mark1)
            fail("planned owner mark index");
        const std::uint32_t local_support = conditioned_support_unrank_host(
            base.L() - 1, local_ones, mark0, mark1, local_sr);

        std::uint32_t full = 0;
        int oq = 0;
        for (int bit = 0; bit < base.W; ++bit) {
            if (bit >= base.lo() && bit <= base.hi()) continue;
            if ((outer >> oq) & 1u) full |= std::uint32_t(1) << bit;
            ++oq;
        }
        int cp = 0;
        for (int bit = base.lo(); bit <= base.hi(); ++bit) {
            if (bit == missing) continue;
            if ((local_support >> cp) & 1u) full |= std::uint32_t(1) << bit;
            ++cp;
        }
        if ((full >> missing) & 1u) fail("planned owner missing occupied");
        return materialize_label_host(full, base.W, missing, pr);
    }
};

void verify_planned_owner_codec(int W, int K, bool reverse, int ngpu) {
    const int tile_start = reverse ? 1 : W - 1;
    const int begin = reverse ? 1 : W - 1;
    const int end = reverse ? std::min(W - 3, K) : std::max(3, W - K);
    const int delta = reverse ? 1 : -1;
    for (int owner = 0; owner < ngpu; ++owner) {
        OwnerComponentCodec base{W, tile_start, K, reverse, owner, ngpu};
        PlannedOwnerCodec planned(base);
        for (int p = begin;; p += delta) {
            for (Rank r = 0; r < base.size(); ++r) {
                const MateID a = base.label_unrank(p, r);
                const MateID b = planned.label_unrank(p, r);
                if (a != b)
                    fail("planned owner label mismatch W=" + std::to_string(W) +
                         " p=" + std::to_string(p) +
                         " owner=" + std::to_string(owner) +
                         " rank=" + std::to_string(r));
            }
            if (p == end) break;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 12 || ngpu < 2 || ngpu > 16) return 2;

    for (int W = 7; W <= maxW; ++W) {
        const int K = std::min(4, W - 3);
        verify_planned_owner_codec(W, K, false, ngpu);
        verify_planned_owner_codec(W, K, true, ngpu);
        std::cout << "W=" << W
                  << " K=" << K
                  << " ngpu=" << ngpu
                  << " planned_owner_unrank=OK"
                  << " local_position_array=0"
                  << " per_component_owner_boundary_division=0"
                  << " forward_reverse=OK\n";
    }
    std::cout << "ALL_OK production_owner_component_planned_codec=1\n";
    return 0;
}
