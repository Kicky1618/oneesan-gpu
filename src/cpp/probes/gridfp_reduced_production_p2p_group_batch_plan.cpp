#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_multigpu_owner_probe_main_unused
#include "gridfp_reduced_production_multigpu_owner_probe.cpp"
#pragma pop_macro("main")

#include <iomanip>

namespace {

struct GroupPlanItem {
    int outer_ones = 0;
    Rank outer_sr = 0;
    Rank state_words = 0;
    Rank support_slabs = 0;
};

struct BatchPlan {
    Rank state_capacity_words = 0;
    Rank descriptor_capacity = 0;
    Rank groups = 0;
};

Rank support_slabs_per_outer_group(int L, int outer_ones) {
    __uint128_t total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        total += choose_u64(L, local);
        if (local >= 1) total += choose_u64(L - 2, local - 1);
    }
    return static_cast<Rank>(total);
}

std::vector<std::vector<GroupPlanItem>> build_owner_groups(
    int W,
    int K,
    int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    const Rank total = total_grouped_states(L, O);
    std::vector<std::vector<GroupPlanItem>> owners(
        static_cast<std::size_t>(ngpu));

    Rank prefix = 0;
    for (int r = 0; r <= O; ++r) {
        const Rank group_words = fixed_outer_group_size(L, r);
        const Rank group_slabs = support_slabs_per_outer_group(L, r);
        const Rank count = choose_u64(O, r);
        for (Rank sr = 0; sr < count; ++sr) {
            const __uint128_t midpoint =
                __uint128_t(prefix) + __uint128_t(sr) * group_words +
                group_words / 2;
            int owner = int(midpoint * ngpu / total);
            if (owner >= ngpu) owner = ngpu - 1;
            owners[static_cast<std::size_t>(owner)].push_back(
                GroupPlanItem{r, sr, group_words, group_slabs});
        }
        prefix += count * group_words;
    }
    return owners;
}

std::vector<BatchPlan> pack_owner_groups(
    const std::vector<GroupPlanItem>& groups,
    Rank scratch_budget_words
) {
    std::vector<BatchPlan> out;
    BatchPlan cur;
    for (const auto& group : groups) {
        if (group.state_words > scratch_budget_words)
            fail("group exceeds static scratch budget");
        if (cur.groups &&
            cur.state_capacity_words + group.state_words > scratch_budget_words) {
            out.push_back(cur);
            cur = BatchPlan{};
        }
        cur.state_capacity_words += group.state_words;
        cur.descriptor_capacity += group.support_slabs;
        ++cur.groups;
    }
    if (cur.groups) out.push_back(cur);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int K = argc > 2 ? std::atoi(argv[2]) : 13;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const double scratch_gib = argc > 4 ? std::atof(argv[4]) : 32.0;
    const Rank descriptor_bytes = argc > 5
        ? static_cast<Rank>(std::strtoull(argv[5], nullptr, 10)) : 32ULL;
    if (W < 7 || W > 28 || K < 1 || K + 2 > W ||
        ngpu < 2 || ngpu > 64 || scratch_gib <= 0.0 || !descriptor_bytes)
        return 2;

    const int L = K + 2;
    const int O = W - L;
    const Rank total_states = total_grouped_states(L, O);
    const Rank scratch_budget_words = static_cast<Rank>(
        scratch_gib * double(1ULL << 30) / sizeof(std::uint32_t));
    const auto owners = build_owner_groups(W, K, ngpu);

    Rank total_groups = 0;
    Rank total_slabs = 0;
    Rank state_sum = 0;
    Rank max_group_words = 0;
    Rank max_owner_state = 0;
    Rank max_owner_slabs = 0;
    Rank max_batch_words = 0;
    Rank max_batch_desc = 0;
    std::size_t max_rounds = 0;

    std::cout << std::fixed << std::setprecision(6);
    for (int g = 0; g < ngpu; ++g) {
        Rank owner_state = 0;
        Rank owner_slabs = 0;
        for (const auto& item : owners[static_cast<std::size_t>(g)]) {
            owner_state += item.state_words;
            owner_slabs += item.support_slabs;
            max_group_words = std::max(max_group_words, item.state_words);
        }
        const auto batches = pack_owner_groups(
            owners[static_cast<std::size_t>(g)], scratch_budget_words);
        Rank owner_max_batch_words = 0;
        Rank owner_max_batch_desc = 0;
        for (const auto& batch : batches) {
            owner_max_batch_words = std::max(
                owner_max_batch_words, batch.state_capacity_words);
            owner_max_batch_desc = std::max(
                owner_max_batch_desc, batch.descriptor_capacity);
        }
        max_batch_words = std::max(max_batch_words, owner_max_batch_words);
        max_batch_desc = std::max(max_batch_desc, owner_max_batch_desc);
        max_rounds = std::max(max_rounds, batches.size());
        max_owner_state = std::max(max_owner_state, owner_state);
        max_owner_slabs = std::max(max_owner_slabs, owner_slabs);
        total_groups += owners[static_cast<std::size_t>(g)].size();
        total_slabs += owner_slabs;
        state_sum += owner_state;

        std::cout << "group-batch-owner"
                  << " gpu=" << g
                  << " groups=" << owners[static_cast<std::size_t>(g)].size()
                  << " support_slabs=" << owner_slabs
                  << " state_GiB="
                  << double(owner_state) * 4.0 / double(1ULL << 30)
                  << " batches=" << batches.size()
                  << " max_static_scratch_GiB="
                  << double(owner_max_batch_words) * 4.0 / double(1ULL << 30)
                  << " max_descriptor_GiB="
                  << double(owner_max_batch_desc * descriptor_bytes) /
                         double(1ULL << 30)
                  << '\n';
    }

    const Rank expected_groups = Rank(1) << O;
    const Rank expected_slabs = 5ULL * (Rank(1) << (W - 3));
    if (total_groups != expected_groups || total_slabs != expected_slabs ||
        state_sum != total_states)
        fail("group batch exact totals");

    const double b300_gib = 288e9 / double(1ULL << 30);
    const double peak_gib =
        double(max_owner_state) * 4.0 / double(1ULL << 30) +
        double(max_batch_words) * 4.0 / double(1ULL << 30) +
        double(max_batch_desc * descriptor_bytes) / double(1ULL << 30);

    std::cout << "gridfp-p2p-group-batch-plan"
              << " W=" << W
              << " Kwin=" << K
              << " local_window=" << L
              << " outer_bits=" << O
              << " ngpu=" << ngpu
              << " states=" << total_states
              << " outer_groups=" << total_groups
              << " support_slabs=" << total_slabs
              << " scratch_budget_GiB=" << scratch_gib
              << " max_group_GiB="
              << double(max_group_words) * 4.0 / double(1ULL << 30)
              << " max_rounds=" << max_rounds
              << " max_owner_support_slabs=" << max_owner_slabs
              << " max_batch_static_scratch_GiB="
              << double(max_batch_words) * 4.0 / double(1ULL << 30)
              << " max_batch_descriptor_GiB="
              << double(max_batch_desc * descriptor_bytes) /
                     double(1ULL << 30)
              << " B300_GiB=" << b300_gib
              << " conservative_peak_GiB=" << peak_gib
              << " B300_headroom_GiB=" << (b300_gib - peak_gib)
              << " count_passes=0"
              << " static_capacity_exact_upper_bound=1"
              << " native_peer_atomics_required=0\n";

    if (W == 28 && K == 13 && ngpu == 8 && total_states != 473397057701ULL)
        return 3;
    std::cout << "ALL_OK gridfp_p2p_group_batch_plan=1\n";
    return 0;
}
