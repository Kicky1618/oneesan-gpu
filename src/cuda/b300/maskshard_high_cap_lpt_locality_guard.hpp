#pragma once

#ifndef MASKSHARD_HIGH_CAP_LPT_LOCALITY_GUARD
#error "v0.78 locality guard requires MASKSHARD_HIGH_CAP_LPT_LOCALITY_GUARD"
#endif
#ifndef MASKSHARD_LAZY_ZERO_BLOCK_INIT
#error "v0.78 exact peer-I/O model currently assumes lazy BLOCKED gather removal"
#endif

#include "maskshard_high_lpt_locality_policy.hpp"

static const MaskShardLayout* G_MS_HIGH_CAP_LPT_LOCALITY_SHARD = nullptr;

#ifdef report_high_mask_shard_layout
static void maskshard_report_high_mask_shard_layout_locality(
    const MaskShardLayout& shard
) {
    report_high_mask_shard_layout(shard);
    G_MS_HIGH_CAP_LPT_LOCALITY_SHARD = &shard;
}
#undef report_high_mask_shard_layout
#define report_high_mask_shard_layout maskshard_report_high_mask_shard_layout_locality
#else
#error "v0.78 locality guard requires row-depth shard-layout setup hook"
#endif

struct MaskShardHighCapLptLocalityCache {
    static constexpr int FULL_CAP = TARGET_W / 2;
    static constexpr int HIGH_HEIGHTS = HIGH_LUT_K + 2;
    using OwnerCaps = std::array<std::uint64_t, 8>;
    std::array<std::array<OwnerCaps, FULL_CAP + 1>, LOW_LUT_K + 1> local_io{};
    std::array<std::array<std::uint64_t, FULL_CAP + 1>, LOW_LUT_K + 1> total_io{};
    bool built = false;

    static void add_checked(std::uint64_t& dst, std::uint64_t x, const char* what) {
        if (x > std::numeric_limits<std::uint64_t>::max() - dst) {
            std::cerr << "HIGH locality I/O overflow at " << what << '\n';
            std::exit(400);
        }
        dst += x;
    }

    void build(const MaskShardLayout& shard) {
        if (built) return;
        if (shard.ngpu < 1 || shard.ngpu > 8
            || shard.masks != (std::uint32_t(1) << HIGH_LUT_K)) {
            std::cerr << "HIGH locality invalid shard layout ngpu=" << shard.ngpu
                      << " masks=" << shard.masks << '\n';
            std::exit(401);
        }
        auto& exact = maskshard_row_depth_exact_cache();
        exact.build();
        auto& orbit = maskshard_row_depth_orbit_compact_cache();
        orbit.build();

        using CapCounts = std::array<std::uint32_t, FULL_CAP + 1>;
        using HeightCaps = std::array<CapCounts, HIGH_HEIGHTS>;
        std::array<HeightCaps, 8> high_active{};
        constexpr int S = FactorTablesHost::STRIDE;
        constexpr int H = HIGH_LUT_K;

        for (std::uint32_t mask = 0; mask < shard.masks; ++mask) {
            const int owner = int(shard.owner[mask]);
            if (owner < 0 || owner >= shard.ngpu) {
                std::cerr << "HIGH locality invalid owner mask=" << mask
                          << " owner=" << owner << '\n';
                std::exit(402);
            }
            for (int he = 0; he < HIGH_HEIGHTS; ++he) {
                const std::size_t ix = std::size_t(mask) * S + std::size_t(he);
                const std::uint32_t a = G_FACTOR.high_mask_off[ix];
                const std::uint32_t n = factor_count(G_FACTOR.high_mask_off, mask, he);
                for (std::uint32_t r = 0; r < n; ++r) {
                    const std::uint32_t code = G_FACTOR.high_mask_codes[a + r];
                    const int peak = int(exact.peak_code(code, H, 1));
                    if (peak < 0 || peak > FULL_CAP) {
                        std::cerr << "HIGH locality peak out of range mask=" << mask
                                  << " he=" << he << " peak=" << peak << '\n';
                        std::exit(403);
                    }
                    for (int cap = std::max(1, peak); cap <= FULL_CAP; ++cap)
                        ++high_active[std::size_t(owner)][std::size_t(he)]
                                     [std::size_t(cap)];
                }
            }
        }

        auto active_for = [&](
            const std::vector<FBlock>& blocks,
            std::uint32_t low_mask,
            int cap,
            int owner,
            bool blocked
        ) -> std::uint64_t {
            cap = std::max(1, std::min(cap, FULL_CAP));
            std::uint64_t total = 0;
            for (const FBlock& b : blocks) {
                if (!b.stride) continue;
                const int low_h = blocked ? int(b.he) : int(b.hs);
                if (int(b.he) < 0 || int(b.he) >= HIGH_HEIGHTS
                    || low_h < 0 || low_h > LOW_LUT_K + 1) {
                    std::cerr << "HIGH locality invalid FBlock heights he="
                              << int(b.he) << " low_h=" << low_h << '\n';
                    std::exit(404);
                }
                const std::uint64_t hc = high_active[std::size_t(owner)]
                    [std::size_t(b.he)][std::size_t(cap)];
                const std::uint64_t lc = orbit.low_count[
                    MaskShardRowDepthOrbitCompactCache::low_count_index(
                        low_mask, low_h, cap)];
                if (hc && lc > std::numeric_limits<std::uint64_t>::max() / hc) {
                    std::cerr << "HIGH locality active product overflow\n";
                    std::exit(405);
                }
                add_checked(total, hc * lc, "active_for");
            }
            return total;
        };

        for (int k = 0; k <= LOW_LUT_K; ++k) {
            const std::uint32_t low_mask = k
                ? ((std::uint32_t(1) << k) - 1u) : 0u;
            const auto main_blocks = make_factor_main_blocks(true, low_mask);
            const auto block_blocks = make_factor_block_blocks(true, low_mask);
            for (int cap = 1; cap <= FULL_CAP; ++cap) {
                const int gather_cap = cap < FULL_CAP
                    ? std::max(cap - 1, 1) : std::max(FULL_CAP - 1, 1);
                for (int d = 0; d < shard.ngpu; ++d) {
                    const std::uint64_t main_gather = active_for(
                        main_blocks, low_mask, gather_cap, d, false);
                    const std::uint64_t main_scatter = active_for(
                        main_blocks, low_mask, cap, d, false);
                    const std::uint64_t block_scatter = active_for(
                        block_blocks, low_mask, cap, d, true);
                    std::uint64_t local = 0;
                    add_checked(local, main_gather, "main gather");
                    add_checked(local, main_scatter, "main scatter");
                    add_checked(local, block_scatter, "block scatter");
                    if (cap == FULL_CAP) {
                        const std::uint64_t extra_rows =
                            std::uint64_t(TARGET_W - FULL_CAP);
                        const std::uint64_t main_full = main_scatter;
                        const std::uint64_t block_full = block_scatter;
                        if (main_full > (std::numeric_limits<std::uint64_t>::max()
                                         - block_full) / 2ULL) {
                            std::cerr << "HIGH locality saturated row overflow\n";
                            std::exit(406);
                        }
                        const std::uint64_t full_row = 2ULL * main_full + block_full;
                        if (extra_rows && full_row
                            > std::numeric_limits<std::uint64_t>::max() / extra_rows) {
                            std::cerr << "HIGH locality saturated aggregate overflow\n";
                            std::exit(407);
                        }
                        add_checked(local, extra_rows * full_row,
                                    "saturated extra rows");
                    }
                    local_io[std::size_t(k)][std::size_t(cap)][std::size_t(d)] = local;
                    add_checked(total_io[std::size_t(k)][std::size_t(cap)],
                                local, "owner sum");
                }
            }
        }
        built = true;
        std::cerr << "fullorbit-batch HIGH locality host table classes="
                  << (LOW_LUT_K + 1) << " caps=" << FULL_CAP
                  << " owners=" << shard.ngpu
                  << " gpu_metadata_bytes=0\n";
    }

    template<class Job>
    std::uint64_t peer_io(
        const MaskShardHighStaticLptSchedule& plan,
        const std::vector<Job>& jobs,
        int cap
    ) const {
        if (!built || cap < 1 || cap > FULL_CAP) {
            std::cerr << "HIGH locality peer model used before build/cap="
                      << cap << '\n';
            std::exit(408);
        }
        std::uint64_t peer = 0;
        for (std::size_t d = 0; d < plan.jobs_by_gpu.size(); ++d) {
            if (d >= 8) {
                std::cerr << "HIGH locality peer model GPU overflow\n";
                std::exit(409);
            }
            for (std::size_t q : plan.jobs_by_gpu[d]) {
                const int pc = maskshard_high_cap_lpt_popcount(jobs[q].low_mask);
                const std::uint64_t total = total_io[std::size_t(pc)][std::size_t(cap)];
                const std::uint64_t local = local_io[std::size_t(pc)]
                    [std::size_t(cap)][d];
                if (local > total) {
                    std::cerr << "HIGH locality local exceeds total\n";
                    std::exit(410);
                }
                add_checked(peer, total - local, "peer sum");
            }
        }
        return peer;
    }
};

static MaskShardHighCapLptLocalityCache& maskshard_high_cap_lpt_locality_cache() {
    static MaskShardHighCapLptLocalityCache cache;
    return cache;
}

template<class Job>
struct MaskShardHighCapLptState : MaskShardHighCapLptStateV077<Job> {
    using Base = MaskShardHighCapLptStateV077<Job>;
    std::uint64_t peer_io_v077 = 0;
    std::uint64_t peer_io_v078 = 0;
    std::uint64_t locality_selected_caps = 0;
    bool locality_prepared = false;

    void prepare() {
        if (locality_prepared) return;
        Base::prepare();
        if (!G_MS_HIGH_CAP_LPT_LOCALITY_SHARD) {
            std::cerr << "HIGH locality missing captured shard layout\n";
            std::exit(411);
        }
        auto& locality = maskshard_high_cap_lpt_locality_cache();
        locality.build(*G_MS_HIGH_CAP_LPT_LOCALITY_SHARD);
        auto& orbit = maskshard_row_depth_orbit_compact_cache();

        std::array<std::vector<FBlock>, LOW_LUT_K + 1> closure_blocks_by_class;
        for (int k = 0; k <= LOW_LUT_K; ++k) {
            const std::uint32_t mask = k
                ? ((std::uint32_t(1) << k) - 1u) : 0u;
            closure_blocks_by_class[std::size_t(k)] =
                make_factor_main_blocks(true, mask);
        }

        std::vector<std::uint64_t> weight(Base::jobs->size(), 0);
        std::uint64_t selected_classes_total = 0;
        peer_io_v077 = peer_io_v078 = 0;
        locality_selected_caps = 0;
        for (int cap = 1; cap <= Base::FULL_CAP; ++cap) {
            for (std::size_t q = 0; q < Base::jobs->size(); ++q) {
                const Job& job = (*Base::jobs)[q];
                std::array<Code, HIGH_LUT_K + 3> orbit_prefix{};
                std::array<std::uint16_t, HIGH_LUT_K + 2> orbit_low_count{};
                const Code orbit_tasks = orbit.make_job_plan(
                    job.low_mask, cap, orbit_prefix, orbit_low_count);
                std::uint64_t work = 0;
                Base::checked_add(work, 2ULL * std::uint64_t(job.main_n));
                Base::checked_add(work, std::uint64_t(job.block_n));
                if (std::uint64_t(orbit_tasks)
                    > std::numeric_limits<std::uint64_t>::max()
                          / std::uint64_t(HIGH_LUT_K)) {
                    std::cerr << "HIGH locality orbit work overflow mask="
                              << job.low_mask << " cap=" << cap << '\n';
                    std::exit(412);
                }
                Base::checked_add(work,
                    std::uint64_t(orbit_tasks) * std::uint64_t(HIGH_LUT_K));
                const int pc = maskshard_high_cap_lpt_popcount(job.low_mask);
                Base::checked_add(work, Base::exact_closure_lane_work(
                    closure_blocks_by_class[std::size_t(pc)],
                    orbit_low_count, cap));
                weight[q] = work;
            }

            auto& selected = Base::by_cap[std::size_t(cap - 1)];
            const auto v077 = selected;
            const auto candidate =
                maskshard_build_high_lpt_from_weights_affinity_locality(
                    weight, *Base::jobs, Base::ngpu,
                    [&](std::size_t q, int d) -> std::uint64_t {
                        const int pc = maskshard_high_cap_lpt_popcount(
                            (*Base::jobs)[q].low_mask);
                        return locality.local_io[std::size_t(pc)]
                            [std::size_t(cap)][std::size_t(d)];
                    });
            if (!Base::same_load_multiset(v077, candidate)) {
                std::cerr << "HIGH locality changed load multiset cap="
                          << cap << '\n';
                std::exit(413);
            }
            const std::uint64_t v077_classes =
                Base::class_count(v077, *Base::jobs);
            const std::uint64_t candidate_classes =
                Base::class_count(candidate, *Base::jobs);
            const std::uint64_t v077_peer =
                locality.peer_io(v077, *Base::jobs, cap);
            const std::uint64_t candidate_peer =
                locality.peer_io(candidate, *Base::jobs, cap);
            const bool adopt = candidate_classes <= v077_classes
                && candidate_peer <= v077_peer
                && (candidate_classes < v077_classes || candidate_peer < v077_peer);
            if (adopt) {
                selected = candidate;
                ++locality_selected_caps;
            }
            const std::uint64_t final_classes =
                Base::class_count(selected, *Base::jobs);
            const std::uint64_t final_peer =
                locality.peer_io(selected, *Base::jobs, cap);
            if (final_classes > v077_classes || final_peer > v077_peer) {
                std::cerr << "HIGH locality final Pareto regression cap="
                          << cap << '\n';
                std::exit(414);
            }
            Base::checked_add(selected_classes_total, final_classes);
            Base::checked_add(peer_io_v077, v077_peer);
            Base::checked_add(peer_io_v078, final_peer);
            std::cerr << "fullorbit-batch HIGH locality cap=" << cap
                      << " graph_v077=" << v077_classes
                      << " graph_candidate=" << candidate_classes
                      << " graph_final=" << final_classes
                      << " peer_v077=" << v077_peer
                      << " peer_candidate=" << candidate_peer
                      << " peer_final=" << final_peer
                      << " selected=" << (adopt ? 1 : 0) << '\n';
        }
        if (selected_classes_total > Base::cap_capture_classes
            || peer_io_v078 > peer_io_v077) {
            std::cerr << "HIGH locality aggregate regression\n";
            std::exit(415);
        }
        const std::uint64_t graph_v077 = Base::cap_capture_classes;
        Base::cap_capture_classes = selected_classes_total;
        locality_prepared = true;
        std::cerr << "fullorbit-batch HIGH locality graph_v077=" << graph_v077
                  << " graph_v078=" << Base::cap_capture_classes
                  << " graph_savings="
                  << (graph_v077 - Base::cap_capture_classes)
                  << " peer_v077=" << peer_io_v077
                  << " peer_v078=" << peer_io_v078
                  << " peer_savings=" << (peer_io_v077 - peer_io_v078)
                  << " selected_caps=" << locality_selected_caps
                  << " load_multiset=identical gpu_metadata_bytes=0\n";
    }
};

template<class Job>
struct MaskShardHighCapLptJobsProxy {
    std::shared_ptr<MaskShardHighCapLptState<Job>> state;
    const std::vector<std::size_t>& operator[](std::size_t gpu) const {
        if (!state || !state->locality_prepared) {
            std::cerr << "HIGH locality schedule used before prepare\n";
            std::exit(416);
        }
        const int row = G_MS_HIGH_CAP_LPT_ROW.load(std::memory_order_acquire);
        if (row < 0 || row >= TARGET_W || gpu >= std::size_t(state->ngpu)) {
            std::cerr << "HIGH locality invalid row/GPU row="
                      << row << " gpu=" << gpu << '\n';
            std::exit(417);
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
