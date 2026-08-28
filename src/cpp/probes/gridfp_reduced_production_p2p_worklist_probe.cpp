#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_modal_probe_main_unused
#include "gridfp_reduced_production_p2p_modal_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

static constexpr int WORK_DESC_BASE_BITS = 26;
static constexpr std::uint32_t WORK_DESC_BASE_MASK =
    (std::uint32_t(1) << WORK_DESC_BASE_BITS) - 1u;

std::uint32_t expand_descriptor_base(Rank compact, int W, int q) {
    std::uint32_t full = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((compact >> cp) & 1ULL) full |= std::uint32_t(1) << bit;
        ++cp;
    }
    if (cp != W - 2) fail("worklist base expansion width");
    return full;
}

int descriptor_run_seeds(
    Rank compact, int W, int q, bool reverse, RunKey (&out)[3]
) {
    const std::uint32_t base = expand_descriptor_base(compact, W, q);
    const int fixed = reverse ? q : q - 1;
    const int missing = reverse ? q - 1 : q;
    const bool odd = (__builtin_popcountll(compact) & 1) != 0;
    if (odd) {
        out[0] = RunKey{base, false};
        out[1] = RunKey{
            base | (std::uint32_t(1) << (q - 1)) |
                   (std::uint32_t(1) << q), false};
        return 2;
    }
    const std::uint32_t fixed_support = base | (std::uint32_t(1) << fixed);
    out[0] = RunKey{fixed_support, false};
    out[1] = RunKey{fixed_support, true};
    out[2] = RunKey{base | (std::uint32_t(1) << missing), false};
    return 3;
}

std::pair<int, bool> cycle_length_and_leader(
    RunKey root, int W, int q, int K, bool reverse
) {
    RunKey cur = root;
    std::uint32_t minimum = root.support;
    int len = 0;
    do {
        cur = shifted_next_key(cur, W, q, K, K, reverse);
        minimum = std::min(minimum, cur.support);
        ++len;
        if (len > RP_MAX_W) fail("worklist cycle length");
    } while (!(cur.support == root.support && cur.blocked == root.blocked));
    return {len, minimum == root.support};
}

int support_owner(
    std::uint32_t support, int W, int tile_start, int K, bool reverse, int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    const std::uint32_t outer = compress_outside_window(support, W, lo, hi);
    return weighted_owner(outer, L, O, ngpu);
}

std::uint32_t pack_work_descriptor(Rank base_rank, int ri) {
    if (base_rank > WORK_DESC_BASE_MASK || ri < 0 || ri > 3)
        fail("work descriptor range");
    return static_cast<std::uint32_t>(base_rank) |
           (static_cast<std::uint32_t>(ri) << WORK_DESC_BASE_BITS);
}

void unpack_work_descriptor(std::uint32_t desc, Rank& base_rank, int& ri) {
    base_rank = desc & WORK_DESC_BASE_MASK;
    ri = int(desc >> WORK_DESC_BASE_BITS);
}

struct WorklistStats {
    Rank runs = 0;
    Rank fixed_cycles = 0;
    Rank descriptors = 0;
    Rank main_descriptors = 0;
    Rank blocked_descriptors = 0;
    std::array<Rank, 8> by_owner{};
    __uint128_t modal_remote_ops = 0;
    __uint128_t boundary_payload_ops = 0;
};

WorklistStats verify_worklist(int W, bool reverse, int ngpu) {
    const int K = (W - 2) / 2;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    const Rank base_count = Rank(1) << (W - 2);

    WorklistStats out;
    std::set<std::uint32_t> descriptors;
    for (Rank base_rank = 0; base_rank < base_count; ++base_rank) {
        RunKey seeds[3]{};
        const int nr = descriptor_run_seeds(base_rank, W, q, reverse, seeds);
        out.runs += nr;
        for (int ri = 0; ri < nr; ++ri) {
            const RunKey root = seeds[ri];
            const auto [len, leader] = cycle_length_and_leader(
                root, W, q, K, reverse);
            if (!leader) continue;
            if (len == 1) {
                ++out.fixed_cycles;
                continue;
            }

            const std::uint32_t desc = pack_work_descriptor(base_rank, ri);
            if (!descriptors.insert(desc).second) fail("duplicate work descriptor");
            Rank decoded_base = 0;
            int decoded_ri = -1;
            unpack_work_descriptor(desc, decoded_base, decoded_ri);
            if (decoded_base != base_rank || decoded_ri != ri)
                fail("work descriptor round trip");
            RunKey decoded_seeds[3]{};
            const int decoded_nr = descriptor_run_seeds(
                decoded_base, W, q, reverse, decoded_seeds);
            if (decoded_ri >= decoded_nr ||
                decoded_seeds[decoded_ri].support != root.support ||
                decoded_seeds[decoded_ri].blocked != root.blocked)
                fail("work descriptor reconstruction");

            std::array<int, 8> counts{};
            std::array<int, RP_MAX_W> owners{};
            RunKey cur = root;
            const Rank pc = catalan((__builtin_popcount(root.support) + 1) / 2);
            for (int h = 0; h < len; ++h) {
                const int owner = support_owner(
                    cur.support, W, old_start, K, reverse, ngpu);
                if (owner < 0 || owner >= ngpu) fail("worklist owner range");
                owners[static_cast<std::size_t>(h)] = owner;
                ++counts[static_cast<std::size_t>(owner)];
                cur = shifted_next_key(cur, W, q, K, K, reverse);
            }
            int exec = 0;
            for (int g = 1; g < ngpu; ++g)
                if (counts[static_cast<std::size_t>(g)] >
                    counts[static_cast<std::size_t>(exec)]) exec = g;
            ++out.by_owner[static_cast<std::size_t>(exec)];
            const Rank remote_runs = len - counts[static_cast<std::size_t>(exec)];
            Rank boundaries = 0;
            for (int h = 0; h < len; ++h)
                boundaries += owners[static_cast<std::size_t>(h)] !=
                    owners[static_cast<std::size_t>((h + 1) % len)];
            out.modal_remote_ops += __uint128_t(2) * remote_runs * pc;
            out.boundary_payload_ops += __uint128_t(boundaries) * pc;
            ++out.descriptors;
            if (root.blocked) ++out.blocked_descriptors;
            else ++out.main_descriptors;
        }
    }

    const Rank expected_runs = Rank(5) << (W - 3);
    if (out.runs != expected_runs) fail("worklist run count");

    const ModalTraffic modal = verify_modal_traffic(W, K, reverse, ngpu);
    if (out.descriptors + out.fixed_cycles != modal.cycles)
        fail("worklist cycle coverage count");
    if (out.modal_remote_ops != modal.modal_remote_ops)
        fail("worklist modal traffic mismatch");
    if (out.boundary_payload_ops != modal.boundary_payload_ops)
        fail("worklist boundary payload mismatch");
    Rank owner_sum = 0;
    for (int g = 0; g < ngpu; ++g) owner_sum += out.by_owner[static_cast<std::size_t>(g)];
    if (owner_sum != out.descriptors) fail("worklist owner partition");
    return out;
}

Rank w28_main_cycle_count() {
    // Burnside over odd-weight 28-bit supports under cyclic rotation.  A
    // rotation fixes odd-weight strings only when its repetition count is odd.
    Rank fixed_sum = 0;
    for (int shift = 0; shift < 28; ++shift) {
        const int d = std::gcd(28, shift);
        const int repeats = 28 / d;
        if (repeats & 1) fixed_sum += Rank(1) << (d - 1);
    }
    if (fixed_sum % 28) fail("W28 main Burnside divisibility");
    return fixed_sum / 28;
}

void print_w28_worklist_theory() {
    const Rank main_cycles = w28_main_cycle_count();
    const Rank blocked_runs = Rank(1) << 25; // even compact parity + fixed occupied bit
    const Rank blocked_fixed = Rank(1) << 13; // two 13-bit halves equal
    const Rank blocked_nontrivial = (blocked_runs - blocked_fixed) / 2;
    const Rank descriptors = main_cycles + blocked_nontrivial;
    const Rank bytes = descriptors * sizeof(std::uint32_t);
    std::cout << "W=28 K=13"
              << " main_cycle_descriptors=" << main_cycles
              << " blocked_nontrivial_descriptors=" << blocked_nontrivial
              << " skipped_blocked_fixed_cycles=" << blocked_fixed
              << " total_work_descriptors=" << descriptors
              << " descriptor_bytes=" << bytes
              << " descriptor_MiB=" << double(bytes) / double(1ULL << 20)
              << " avg_descriptor_MiB_per_8gpu="
              << double(bytes) / 8.0 / double(1ULL << 20)
              << " descriptor_format=base26_plus_ri2_u32"
              << " reusable_across_rows_and_residues=1\n";
    if (main_cycles != 4793492ULL || blocked_nontrivial != 16773120ULL ||
        descriptors != 21566612ULL || bytes != 86266448ULL)
        fail("W28 compact worklist theory constants");
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8) return 2;

    for (int W = 8; W <= maxW; W += 2) {
        for (bool reverse : {false, true}) {
            const WorklistStats s = verify_worklist(W, reverse, ngpu);
            std::cout << "W=" << W
                      << " K=" << ((W - 2) / 2)
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " runs=" << s.runs
                      << " fixed_cycles_skipped=" << s.fixed_cycles
                      << " descriptors=" << s.descriptors
                      << " main_descriptors=" << s.main_descriptors
                      << " blocked_descriptors=" << s.blocked_descriptors
                      << " descriptor_bytes=" << s.descriptors * sizeof(std::uint32_t)
                      << " modal_traffic_exact=1"
                      << " descriptor_bijection=1 owner_partition=1\n";
        }
    }
    print_w28_worklist_theory();
    std::cout << "ALL_OK production_p2p_compact_worklist=1\n";
    return 0;
}
