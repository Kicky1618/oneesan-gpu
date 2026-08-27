#pragma once

#ifndef MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY
#error "capture-affinity layer requires MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY"
#endif
#ifndef MASKSHARD_HIGH_CAP_LPT_EXACT_CLOSURE_LANES
#error "capture-affinity layer requires v0.76 exact closure lane weights"
#endif

template<class Job>
static MaskShardHighStaticLptSchedule
maskshard_build_high_lpt_from_weights_affinity(
    const std::vector<std::uint64_t>& weight,
    const std::vector<Job>& jobs,
    int ngpu
) {
    if (weight.size() != jobs.size() || ngpu < 1) {
        std::cerr << "HIGH capture-affinity LPT invalid input weight="
                  << weight.size() << " jobs=" << jobs.size()
                  << " ngpu=" << ngpu << '\n';
        std::exit(385);
    }
    std::vector<std::size_t> order(weight.size());
    for (std::size_t q = 0; q < order.size(); ++q) order[q] = q;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return weight[a] != weight[b] ? weight[a] > weight[b] : a < b;
    });

    MaskShardHighStaticLptSchedule out;
    out.jobs_by_gpu.resize(std::size_t(ngpu));
    out.work_by_gpu.assign(std::size_t(ngpu), 0);
    std::vector<std::array<std::uint8_t, LOW_LUT_K + 1>> seen_class;
    seen_class.resize(std::size_t(ngpu));
    for (auto& seen : seen_class) seen.fill(0);

    for (std::size_t q : order) {
        int default_best = 0;
        for (int d = 1; d < ngpu; ++d) {
            if (out.work_by_gpu[std::size_t(d)]
                < out.work_by_gpu[std::size_t(default_best)]) {
                default_best = d;
            }
        }
        const std::uint64_t min_load = out.work_by_gpu[std::size_t(default_best)];
        const int pc = maskshard_high_cap_lpt_popcount(jobs[q].low_mask);
        if (pc < 0 || pc > LOW_LUT_K) {
            std::cerr << "HIGH capture-affinity LPT invalid popcount="
                      << pc << " q=" << q << '\n';
            std::exit(386);
        }

        int best = default_best;
        if (!seen_class[std::size_t(best)][std::size_t(pc)]) {
            for (int d = 0; d < ngpu; ++d) {
                if (out.work_by_gpu[std::size_t(d)] == min_load
                    && seen_class[std::size_t(d)][std::size_t(pc)]) {
                    best = d;
                    break;
                }
            }
        }
        const std::uint64_t w = weight[q];
        std::uint64_t& load = out.work_by_gpu[std::size_t(best)];
        if (w > std::numeric_limits<std::uint64_t>::max() - load
            || w > std::numeric_limits<std::uint64_t>::max() - out.total_work) {
            std::cerr << "HIGH capture-affinity LPT work overflow q=" << q << '\n';
            std::exit(387);
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

template<class Job>
struct MaskShardHighCapLptState : MaskShardHighCapLptStateV076<Job> {
    using Base = MaskShardHighCapLptStateV076<Job>;
    std::uint64_t plain_cap_capture_classes = 0;
    std::uint64_t affinity_selected_caps = 0;
    bool affinity_prepared = false;

    static bool same_load_multiset(
        const MaskShardHighStaticLptSchedule& a,
        const MaskShardHighStaticLptSchedule& b
    ) {
        if (a.total_work != b.total_work || a.min_work != b.min_work
            || a.max_work != b.max_work
            || a.work_by_gpu.size() != b.work_by_gpu.size()) return false;
        auto aw = a.work_by_gpu;
        auto bw = b.work_by_gpu;
        std::sort(aw.begin(), aw.end());
        std::sort(bw.begin(), bw.end());
        return aw == bw;
    }

    void prepare() {
        if (affinity_prepared) return;
        Base::prepare();
        if (!Base::jobs || Base::ngpu < 1) {
            std::cerr << "HIGH capture-affinity LPT missing v0.76 state\n";
            std::exit(388);
        }

        plain_cap_capture_classes = Base::cap_capture_classes;
        affinity_selected_caps = 0;
        std::uint64_t selected_total_classes = 0;
        auto& orbit = maskshard_row_depth_orbit_compact_cache();
        std::array<std::vector<FBlock>, LOW_LUT_K + 1> closure_blocks_by_class;
        for (int k = 0; k <= LOW_LUT_K; ++k) {
            const std::uint32_t mask = k
                ? ((std::uint32_t(1) << k) - 1u) : 0u;
            closure_blocks_by_class[std::size_t(k)] =
                make_factor_main_blocks(true, mask);
        }

        std::vector<std::uint64_t> weight(Base::jobs->size(), 0);
        for (int cap = 1; cap <= Base::FULL_CAP; ++cap) {
            for (std::size_t q = 0; q < Base::jobs->size(); ++q) {
                const Job& job = (*Base::jobs)[q];
                std::array<Code, HIGH_LUT_K + 3> orbit_prefix{};
                std::array<std::uint16_t, HIGH_LUT_K + 2> orbit_low_count{};
                const Code orbit_tasks = orbit.make_job_plan(
                    job.low_mask, cap, orbit_prefix, orbit_low_count);
                std::uint64_t work = 0;
                Base::checked_add(work, 2ULL * std::uint64_t(job.main_n));
#ifdef MASKSHARD_LAZY_ZERO_BLOCK_INIT
                Base::checked_add(work, std::uint64_t(job.block_n));
#else
                Base::checked_add(work, 2ULL * std::uint64_t(job.block_n));
#endif
                if (std::uint64_t(orbit_tasks)
                    > std::numeric_limits<std::uint64_t>::max()
                          / std::uint64_t(HIGH_LUT_K)) {
                    std::cerr << "HIGH capture-affinity orbit work overflow mask="
                              << job.low_mask << " cap=" << cap << '\n';
                    std::exit(389);
                }
                Base::checked_add(work,
                    std::uint64_t(orbit_tasks) * std::uint64_t(HIGH_LUT_K));
                const int pc = maskshard_high_cap_lpt_popcount(job.low_mask);
                if (pc < 0 || pc > LOW_LUT_K) {
                    std::cerr << "HIGH capture-affinity invalid popcount=" << pc << '\n';
                    std::exit(390);
                }
                Base::checked_add(work, Base::exact_closure_lane_work(
                    closure_blocks_by_class[std::size_t(pc)],
                    orbit_low_count, cap));
                weight[q] = work;
            }

            auto& selected = Base::by_cap[std::size_t(cap - 1)];
            const auto plain = selected;
            const auto affinity = maskshard_build_high_lpt_from_weights_affinity(
                weight, *Base::jobs, Base::ngpu);
            if (!same_load_multiset(plain, affinity)) {
                std::cerr << "HIGH capture-affinity changed load multiset cap="
                          << cap << '\n';
                std::exit(391);
            }
            const std::uint64_t plain_classes = Base::class_count(plain, *Base::jobs);
            const std::uint64_t affinity_classes =
                Base::class_count(affinity, *Base::jobs);
            if (affinity_classes < plain_classes) {
                selected = affinity;
                ++affinity_selected_caps;
            }
            const std::uint64_t selected_classes =
                Base::class_count(selected, *Base::jobs);
            if (selected_classes > plain_classes) {
                std::cerr << "HIGH capture-affinity class regression cap="
                          << cap << '\n';
                std::exit(392);
            }
            Base::checked_add(selected_total_classes, selected_classes);
            std::cerr << "fullorbit-batch HIGH capture affinity cap=" << cap
                      << " plain_classes=" << plain_classes
                      << " affinity_classes=" << affinity_classes
                      << " selected_classes=" << selected_classes
                      << " selected=" << (affinity_classes < plain_classes ? 1 : 0)
                      << '\n';
        }
        if (selected_total_classes > plain_cap_capture_classes) {
            std::cerr << "HIGH capture-affinity aggregate class regression\n";
            std::exit(393);
        }
        Base::cap_capture_classes = selected_total_classes;
        affinity_prepared = true;
        std::cerr << "fullorbit-batch HIGH capture affinity plain_cap_classes="
                  << plain_cap_capture_classes
                  << " selected_cap_classes=" << Base::cap_capture_classes
                  << " savings="
                  << (plain_cap_capture_classes - Base::cap_capture_classes)
                  << " selected_caps=" << affinity_selected_caps
                  << " load_multiset=identical\n";
    }
};

template<class Job>
struct MaskShardHighCapLptJobsProxy {
    std::shared_ptr<MaskShardHighCapLptState<Job>> state;
    const std::vector<std::size_t>& operator[](std::size_t gpu) const {
        if (!state || !state->affinity_prepared) {
            std::cerr << "HIGH capture-affinity schedule used before prepare\n";
            std::exit(394);
        }
        const int row = G_MS_HIGH_CAP_LPT_ROW.load(std::memory_order_acquire);
        if (row < 0 || row >= TARGET_W || gpu >= std::size_t(state->ngpu)) {
            std::cerr << "HIGH capture-affinity invalid row/GPU row="
                      << row << " gpu=" << gpu << '\n';
            std::exit(395);
        }
        const int cap = std::min(row + 1, MaskShardHighCapLptState<Job>::FULL_CAP);
        return state->by_cap[std::size_t(cap - 1)].jobs_by_gpu[gpu];
    }
};

template<class Job>
struct MaskShardHighCapLptSchedule {
    std::shared_ptr<MaskShardHighCapLptState<Job>> state;
    MaskShardHighCapLptJobsProxy<Job> jobs_by_gpu;
    std::uint64_t total_work = 0;
    std::uint64_t min_work = 0;
    std::uint64_t max_work = 0;
    void prepare() const { state->prepare(); }
};

template<class Job>
static MaskShardHighCapLptSchedule<Job> maskshard_build_high_cap_lpt_schedule(
    const std::vector<Job>& jobs, int ngpu
) {
    auto state = std::make_shared<MaskShardHighCapLptState<Job>>();
    state->jobs = &jobs;
    state->ngpu = ngpu;
    state->baseline = maskshard_build_high_static_lpt_schedule(jobs, ngpu);
    MaskShardHighCapLptSchedule<Job> out;
    out.state = state;
    out.jobs_by_gpu.state = state;
    out.total_work = state->baseline.total_work;
    out.min_work = state->baseline.min_work;
    out.max_work = state->baseline.max_work;
    return out;
}

template<class Schedule>
static void maskshard_prepare_highclosure_rowdepth_compact_cap_lpt(
    const HighDescHost& high_desc, int ngpu, const Schedule& schedule
) {
    maskshard_prepare_highclosure_rowdepth_compact_launch(high_desc, ngpu);
    schedule.prepare();
}

[[maybe_unused]] static void maskshard_set_row_depth_fblock_io_row_cap_lpt(
    int zero_based_row
) {
    G_MS_HIGH_CAP_LPT_ROW.store(zero_based_row, std::memory_order_release);
    maskshard_set_row_depth_exact_io_row(zero_based_row);
}

#define maskshard_build_high_static_lpt_schedule \
        maskshard_build_high_cap_lpt_schedule
#define maskshard_prepare_highclosure_rowdepth_compact(high_desc, ngpu) \
        maskshard_prepare_highclosure_rowdepth_compact_cap_lpt( \
            high_desc, ngpu, high_schedule)
#define maskshard_set_row_depth_fblock_io_row \
        maskshard_set_row_depth_fblock_io_row_cap_lpt
