#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <map>
#include <tuple>

namespace {

struct RunKey {
    std::uint32_t support = 0;
    bool blocked = false;
    bool operator<(const RunKey& o) const {
        return support != o.support ? support < o.support : blocked < o.blocked;
    }
};

struct RunSeen {
    int src_owner = -1;
    int dst_owner = -1;
    Rank src_base = 0;
    Rank dst_base = 0;
    Rank pc = 0;
    std::vector<std::uint8_t> primitive_seen;
};

struct ExactRunStats {
    Rank states = 0;
    Rank runs = 0;
    Rank moved_states = 0;
    Rank moved_runs = 0;
    Rank max_run = 0;
};

ExactRunStats verify_split_runs(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int Kold,
    bool reverse,
    int ngpu
) {
    const int total_steps = W - 3;
    const int Knew = total_steps - Kold;
    if (Kold <= 0 || Knew <= 0) fail("run split width");
    const int q = reverse ? 1 + Kold : W - 1 - Kold;
    const int old_start = reverse ? 1 : W - 1;
    const int new_start = q;

    ProductionFactorTables tables(W);
    const OwnerPlan old_plan = make_owner_plan(W, Kold, ngpu);
    const OwnerPlan new_plan = make_owner_plan(W, Knew, ngpu);
    std::map<RunKey, RunSeen> runs;

    ExactRunStats st;
    for (Key k : layout(main, block, q)) {
        const MateID full = embed_full(k, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const int occupied = __builtin_popcount(support);
        if (!(occupied & 1)) fail("run occupied parity");
        const Rank pc = catalan((occupied + 1) / 2);
        const Rank pr = tables.primitive_rank(full, W);
        if (pr >= pc) fail("run primitive range");

        const GroupedRank sr = grouped_rank(
            k, tables, W, q, reverse, old_start, Kold, ngpu, old_plan);
        const GroupedRank dr = grouped_rank(
            k, tables, W, q, reverse, new_start, Knew, ngpu, new_plan);
        if (sr.local < pr || dr.local < pr) fail("run base underflow");

        const RunKey rk{support, k.blocked};
        auto [it, inserted] = runs.emplace(rk, RunSeen{});
        RunSeen& z = it->second;
        if (inserted) {
            z.src_owner = sr.owner;
            z.dst_owner = dr.owner;
            z.src_base = sr.local - pr;
            z.dst_base = dr.local - pr;
            z.pc = pc;
            z.primitive_seen.assign(static_cast<std::size_t>(pc), 0);
        } else {
            if (z.src_owner != sr.owner || z.dst_owner != dr.owner ||
                z.src_base + pr != sr.local || z.dst_base + pr != dr.local || z.pc != pc)
                fail("primitive run is not contiguous");
        }
        if (z.primitive_seen[static_cast<std::size_t>(pr)]++) fail("duplicate primitive in run");
        ++st.states;
    }

    for (const auto& [rk, z] : runs) {
        (void)rk;
        for (std::uint8_t x : z.primitive_seen) if (x != 1) fail("primitive run hole");
        ++st.runs;
        st.max_run = std::max(st.max_run, z.pc);
        if (z.src_owner != z.dst_owner) {
            ++st.moved_runs;
            st.moved_states += z.pc;
        }
    }

    const Rank expected_runs = Rank(5) << (W - 3);
    if (st.runs != expected_runs) fail("total primitive run formula");
    if (st.states != tables.size()) fail("primitive run state coverage");
    return st;
}

struct RunOwnerHist {
    std::vector<std::vector<Rank>> by_owner_ones;
};

RunOwnerHist run_owner_hist_for_common(
    std::uint32_t common_mask,
    int common_bits,
    int exclusive_bits,
    int L,
    int ngpu
) {
    const int O = common_bits + exclusive_bits;
    RunOwnerHist h;
    h.by_owner_ones.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<Rank>(static_cast<std::size_t>(exclusive_bits + 1), 0));
    const Rank count = Rank(1) << exclusive_bits;
    for (Rank x = 0; x < count; ++x) {
        const std::uint32_t compact = common_mask | (std::uint32_t(x) << common_bits);
        const int owner = weighted_owner(compact, L, O, ngpu);
        ++h.by_owner_ones[static_cast<std::size_t>(owner)]
                         [static_cast<std::size_t>(__builtin_popcountll(x))];
    }
    return h;
}

Rank run_overlap_state_weight(int base_occupied) {
    if (base_occupied & 1) {
        return catalan((base_occupied + 1) / 2) +
               catalan((base_occupied + 3) / 2);
    }
    return 3 * catalan((base_occupied + 2) / 2);
}

struct RunTraffic {
    Rank total_states = 0;
    Rank moved_states = 0;
    Rank total_runs = 0;
    Rank moved_runs = 0;
    Rank pair_states[8][8]{};
    Rank pair_runs[8][8]{};
};

RunTraffic run_traffic_model(int W, int Kold, int ngpu) {
    if (ngpu != 8) fail("run traffic matrix currently fixed to 8 GPUs");
    const int Knew = W - 3 - Kold;
    const int common_bits = W - (Kold + Knew + 2);
    if (common_bits != 1) fail("two-tile run model expects one common bit");
    const int Lold = Kold + 2;
    const int Lnew = Knew + 2;

    RunTraffic out;
    for (std::uint32_t c = 0; c < 2; ++c) {
        const RunOwnerHist old_hist = run_owner_hist_for_common(c, 1, Knew, Lold, ngpu);
        const RunOwnerHist new_hist = run_owner_hist_for_common(c, 1, Kold, Lnew, ngpu);
        for (int so = 0; so < ngpu; ++so) {
            for (int bo = 0; bo <= Knew; ++bo) {
                const Rank cb = old_hist.by_owner_ones[static_cast<std::size_t>(so)]
                                                      [static_cast<std::size_t>(bo)];
                if (!cb) continue;
                for (int d = 0; d < ngpu; ++d) {
                    for (int ao = 0; ao <= Kold; ++ao) {
                        const Rank ca = new_hist.by_owner_ones[static_cast<std::size_t>(d)]
                                                          [static_cast<std::size_t>(ao)];
                        if (!ca) continue;
                        const int base_occupied = int(c) + bo + ao;
                        const Rank state_weight = run_overlap_state_weight(base_occupied);
                        const Rank run_weight = (base_occupied & 1) ? 2 : 3;
                        const Rank pairs = cb * ca;
                        const Rank states = pairs * state_weight;
                        const Rank runs = pairs * run_weight;
                        out.total_states += states;
                        out.total_runs += runs;
                        out.pair_states[so][d] += states;
                        out.pair_runs[so][d] += runs;
                        if (so != d) {
                            out.moved_states += states;
                            out.moved_runs += runs;
                        }
                    }
                }
            }
        }
    }
    const Rank expected_runs = Rank(5) << (W - 3);
    if (out.total_runs != expected_runs) fail("run model total runs");
    return out;
}

void print_w28_run_model(int Kold, int ngpu) {
    const int Knew = 25 - Kold;
    const RunTraffic r = run_traffic_model(28, Kold, ngpu);
    if (r.total_states != 473397057701ULL) fail("W28 run state total");
    const double moved_tib = double(r.moved_states) * 4.0 / double(1ULL << 40);
    const double avg_run = r.moved_runs ? double(r.moved_states) * 4.0 / double(r.moved_runs) : 0.0;
    std::cout << "W=28 tile_split=" << Kold << '+' << Knew
              << " total_runs=" << r.total_runs
              << " moved_runs=" << r.moved_runs
              << " moved_states=" << r.moved_states
              << " moved_TiB=" << moved_tib
              << " avg_moved_run_bytes=" << avg_run
              << " base_support_warps=" << (Rank(1) << 26)
              << " metadata_bytes=0"
              << "\n";

    Rank max_pair_states = 0, max_pair_runs = 0;
    int max_s = -1, max_d = -1;
    for (int s = 0; s < 8; ++s) {
        for (int d = 0; d < 8; ++d) {
            if (s == d) continue;
            if (r.pair_states[s][d] > max_pair_states) {
                max_pair_states = r.pair_states[s][d];
                max_pair_runs = r.pair_runs[s][d];
                max_s = s;
                max_d = d;
            }
        }
    }
    std::cout << "W=28 heaviest_peer=" << max_s << "->" << max_d
              << " states=" << max_pair_states
              << " GiB=" << double(max_pair_states) * 4.0 / double(1ULL << 30)
              << " runs=" << max_pair_runs
              << " avg_run_bytes=" << (max_pair_runs ? double(max_pair_states) * 4.0 / double(max_pair_runs) : 0.0)
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 11 || ngpu != 8) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 7; W <= maxW; ++W) {
        const int total = W - 3;
        for (int Kold = 2; Kold <= total - 2; ++Kold) {
            for (bool reverse : {false, true}) {
                const ExactRunStats s = verify_split_runs(
                    words[W], words[W - 1], W, Kold, reverse, ngpu);
                const RunTraffic model = run_traffic_model(W, Kold, ngpu);
                if (s.states != model.total_states || s.runs != model.total_runs ||
                    s.moved_states != model.moved_states || s.moved_runs != model.moved_runs)
                    fail("exact/model run traffic mismatch");
                std::cout << "W=" << W
                          << " split=" << Kold << '+' << (total - Kold)
                          << " direction=" << (reverse ? "reverse" : "forward")
                          << " states=" << s.states
                          << " runs=" << s.runs
                          << " moved_runs=" << s.moved_runs
                          << " primitive_contiguous=1"
                          << " source_destination_same_run_length=1"
                          << " model=OK\n";
            }
        }
    }

    print_w28_run_model(11, ngpu);
    print_w28_run_model(14, ngpu);
    std::cout << "ALL_OK production_tile_primitive_runs=1"
              << " persistent_base_support_enumeration=1"
              << " p2p_metadata_bytes=0\n";
    return 0;
}
