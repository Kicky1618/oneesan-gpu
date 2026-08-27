#pragma once

// Pure host policy used by v0.78: LPT load remains the primary invariant.
// Among GPUs tied at the exact same minimum load, first reuse the job's graph
// class, then maximize caller-supplied local-I/O work. Choosing only among equal
// minimum loads preserves the complete LPT load multiset.
template<class Job, class LocalScore>
static MaskShardHighStaticLptSchedule
maskshard_build_high_lpt_from_weights_affinity_locality(
    const std::vector<std::uint64_t>& weight,
    const std::vector<Job>& jobs,
    int ngpu,
    LocalScore local_score
) {
    if (weight.size() != jobs.size() || ngpu < 1) {
        std::cerr << "HIGH affinity-locality LPT invalid input weight="
                  << weight.size() << " jobs=" << jobs.size()
                  << " ngpu=" << ngpu << '\n';
        std::exit(396);
    }
    std::vector<std::size_t> order(weight.size());
    for (std::size_t q = 0; q < order.size(); ++q) order[q] = q;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return weight[a] != weight[b] ? weight[a] > weight[b] : a < b;
    });

    MaskShardHighStaticLptSchedule out;
    out.jobs_by_gpu.resize(std::size_t(ngpu));
    out.work_by_gpu.assign(std::size_t(ngpu), 0);
    std::vector<std::array<std::uint8_t, LOW_LUT_K + 1>> seen_class(
        std::size_t(ngpu));
    for (auto& seen : seen_class) seen.fill(0);

    for (std::size_t q : order) {
        std::uint64_t min_load = out.work_by_gpu[0];
        for (int d = 1; d < ngpu; ++d)
            min_load = std::min(min_load, out.work_by_gpu[std::size_t(d)]);

        const int pc = maskshard_high_cap_lpt_popcount(jobs[q].low_mask);
        if (pc < 0 || pc > LOW_LUT_K) {
            std::cerr << "HIGH affinity-locality invalid popcount="
                      << pc << " q=" << q << '\n';
            std::exit(397);
        }
        int best = -1;
        bool best_reuse = false;
        std::uint64_t best_local = 0;
        for (int d = 0; d < ngpu; ++d) {
            if (out.work_by_gpu[std::size_t(d)] != min_load) continue;
            const bool reuse = seen_class[std::size_t(d)][std::size_t(pc)] != 0;
            const std::uint64_t local = local_score(q, d);
            if (best < 0 || reuse > best_reuse
                || (reuse == best_reuse && local > best_local)) {
                best = d;
                best_reuse = reuse;
                best_local = local;
            }
        }
        if (best < 0) {
            std::cerr << "HIGH affinity-locality found no minimum-load GPU q="
                      << q << '\n';
            std::exit(398);
        }

        const std::uint64_t w = weight[q];
        std::uint64_t& load = out.work_by_gpu[std::size_t(best)];
        if (w > std::numeric_limits<std::uint64_t>::max() - load
            || w > std::numeric_limits<std::uint64_t>::max() - out.total_work) {
            std::cerr << "HIGH affinity-locality work overflow q=" << q << '\n';
            std::exit(399);
        }
        out.jobs_by_gpu[std::size_t(best)].push_back(q);
        load += w;
        out.total_work += w;
        seen_class[std::size_t(best)][std::size_t(pc)] = 1;
    }
    if (!out.work_by_gpu.empty()) {
        const auto mm = std::minmax_element(
            out.work_by_gpu.begin(), out.work_by_gpu.end());
        out.min_work = *mm.first;
        out.max_work = *mm.second;
    }
    return out;
}
