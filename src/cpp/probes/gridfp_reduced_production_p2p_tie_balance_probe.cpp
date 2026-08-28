#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_modal_probe_main_unused
#include "gridfp_reduced_production_p2p_modal_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

std::uint32_t p2p_tie_hash_cpu(std::uint32_t support, bool blocked, bool reverse) {
    std::uint32_t x = support;
    x ^= blocked ? 0x85ebca6bu : 0u;
    x ^= reverse ? 0xc2b2ae35u : 0u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int hash_modal_tie_owner(
    const std::array<int, 8>& counts,
    int ngpu,
    std::uint32_t support,
    bool blocked,
    bool reverse
) {
    int max_count = 0;
    for (int g = 0; g < ngpu; ++g)
        max_count = std::max(max_count, counts[static_cast<std::size_t>(g)]);
    std::array<int, 8> tie{};
    int nt = 0;
    for (int g = 0; g < ngpu; ++g)
        if (counts[static_cast<std::size_t>(g)] == max_count)
            tie[static_cast<std::size_t>(nt++)] = g;
    if (!nt) fail("tie hash empty modal set");
    return tie[static_cast<std::size_t>(
        p2p_tie_hash_cpu(support, blocked, reverse) % std::uint32_t(nt))];
}

struct TieBalanceStats {
    Rank cycles = 0;
    Rank tied_cycles = 0;
    __uint128_t low_remote_ops = 0;
    __uint128_t hash_remote_ops = 0;
    __uint128_t greedy_remote_ops = 0;
    std::array<__uint128_t, 8> low_work{};
    std::array<__uint128_t, 8> hash_work{};
    std::array<__uint128_t, 8> greedy_work{};
    std::array<Rank, 8> low_cycles{};
    std::array<Rank, 8> hash_cycles{};
    std::array<Rank, 8> greedy_cycles{};
};

std::pair<__uint128_t, __uint128_t> minmax_u128(
    const std::array<__uint128_t, 8>& a,
    int ngpu
) {
    __uint128_t lo = a[0], hi = a[0];
    for (int g = 1; g < ngpu; ++g) {
        lo = std::min(lo, a[static_cast<std::size_t>(g)]);
        hi = std::max(hi, a[static_cast<std::size_t>(g)]);
    }
    return {lo, hi};
}

TieBalanceStats verify_tie_balance(int W, bool reverse, int ngpu) {
    const int K = (W - 2) / 2;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);

    struct Rec { int owner = -1; Rank pc = 0; };
    std::map<RunKey, Rec> runs;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const Rank pc = catalan((__builtin_popcount(support) + 1) / 2);
        const GroupedRank gr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, plan);
        const RunKey rk{support, key.blocked};
        auto [it, inserted] = runs.emplace(rk, Rec{gr.owner, pc});
        if (!inserted && (it->second.owner != gr.owner || it->second.pc != pc))
            fail("tie-balance run inconsistency");
    }

    TieBalanceStats out;
    std::set<RunKey> seen;
    for (const auto& [candidate, rec0] : runs) {
        if (seen.count(candidate)) continue;
        std::vector<RunKey> cycle;
        RunKey cur = candidate;
        do {
            if (!runs.count(cur)) fail("tie-balance cycle escaped run set");
            if (!seen.insert(cur).second) fail("tie-balance cycle overlap");
            cycle.push_back(cur);
            cur = shifted_next_key(cur, W, q, K, K, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("tie-balance cycle too long");
        } while (!(cur.support == candidate.support && cur.blocked == candidate.blocked));

        const Rank pc = rec0.pc;
        std::array<int, 8> counts{};
        for (const RunKey& r : cycle) {
            const auto it = runs.find(r);
            if (it == runs.end() || it->second.pc != pc)
                fail("tie-balance primitive count changed");
            ++counts[static_cast<std::size_t>(it->second.owner)];
        }
        int max_count = 0;
        for (int g = 0; g < ngpu; ++g)
            max_count = std::max(max_count, counts[static_cast<std::size_t>(g)]);
        std::array<int, 8> ties{};
        int nties = 0;
        for (int g = 0; g < ngpu; ++g)
            if (counts[static_cast<std::size_t>(g)] == max_count)
                ties[static_cast<std::size_t>(nties++)] = g;
        if (nties > 1) ++out.tied_cycles;

        const int low = ties[0];
        const int hash = hash_modal_tie_owner(
            counts, ngpu, candidate.support, candidate.blocked, reverse);
        int greedy = ties[0];
        for (int i = 1; i < nties; ++i) {
            const int g = ties[static_cast<std::size_t>(i)];
            if (out.greedy_work[static_cast<std::size_t>(g)] <
                out.greedy_work[static_cast<std::size_t>(greedy)])
                greedy = g;
        }
        if (counts[static_cast<std::size_t>(hash)] != max_count ||
            counts[static_cast<std::size_t>(greedy)] != max_count)
            fail("tie-balance selected non-modal executor");

        const Rank remote_runs = cycle.size() - Rank(max_count);
        const __uint128_t remote_ops = __uint128_t(2) * remote_runs * pc;
        const __uint128_t work = __uint128_t(cycle.size()) * pc;
        out.low_remote_ops += remote_ops;
        out.hash_remote_ops += remote_ops;
        out.greedy_remote_ops += remote_ops;
        out.low_work[static_cast<std::size_t>(low)] += work;
        out.hash_work[static_cast<std::size_t>(hash)] += work;
        out.greedy_work[static_cast<std::size_t>(greedy)] += work;
        ++out.low_cycles[static_cast<std::size_t>(low)];
        ++out.hash_cycles[static_cast<std::size_t>(hash)];
        ++out.greedy_cycles[static_cast<std::size_t>(greedy)];
        ++out.cycles;
    }
    if (seen.size() != runs.size()) fail("tie-balance run coverage");
    if (out.low_remote_ops != out.hash_remote_ops ||
        out.low_remote_ops != out.greedy_remote_ops)
        fail("tie balancing changed remote traffic");
    return out;
}

double u128_double(__uint128_t x) {
    return static_cast<double>(x);
}

void print_balance(const char* name,
                   const std::array<__uint128_t, 8>& work,
                   int ngpu) {
    const auto [lo, hi] = minmax_u128(work, ngpu);
    __uint128_t total = 0;
    for (int g = 0; g < ngpu; ++g) total += work[static_cast<std::size_t>(g)];
    const double avg = u128_double(total) / double(ngpu);
    std::cout << ' ' << name << "_min_work=" << u128_double(lo)
              << ' ' << name << "_max_work=" << u128_double(hi)
              << ' ' << name << "_max_over_avg="
              << (avg ? u128_double(hi) / avg : 0.0)
              << ' ' << name << "_spread=" << u128_double(hi - lo);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8) return 2;

    for (int W = 8; W <= maxW; W += 2) {
        for (bool reverse : {false, true}) {
            const TieBalanceStats s = verify_tie_balance(W, reverse, ngpu);
            std::cout << "W=" << W
                      << " K=" << ((W - 2) / 2)
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " cycles=" << s.cycles
                      << " tied_cycles=" << s.tied_cycles
                      << " tied_fraction="
                      << (s.cycles ? double(s.tied_cycles) / double(s.cycles) : 0.0)
                      << " remote_ops_equal=1";
            print_balance("low", s.low_work, ngpu);
            print_balance("hash", s.hash_work, ngpu);
            print_balance("greedy", s.greedy_work, ngpu);
            std::cout << "\n";
        }
    }
    std::cout << "W=28_plan tie_hash=canonical_support_mix"
              << " preserves_exact_modal_remote_traffic=1"
              << " sequential_greedy_is_builder_only_reference=1\n";
    std::cout << "ALL_OK production_p2p_modal_tie_balance=1\n";
    return 0;
}
