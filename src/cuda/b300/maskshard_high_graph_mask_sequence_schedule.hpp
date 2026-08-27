#pragma once

#ifndef MASKSHARD_HIGH_GRAPH_MASK_SEQUENCE
#error "HIGH graph mask schedule requires MASKSHARD_HIGH_GRAPH_MASK_SEQUENCE"
#endif
#ifndef MASKSHARD_HIGH_CAP_LPT_LOCALITY_GUARD
#error "v0.79 currently layers on the v0.78 locality scheduler"
#endif

template<class Job>
struct MaskShardHighCapLptState : MaskShardHighCapLptStateV078<Job> {
    using Base = MaskShardHighCapLptStateV078<Job>;
    bool mask_sequence_prepared = false;

    void prepare() {
        if (mask_sequence_prepared) return;
        Base::prepare();
        maskshard_high_graph_mask_sequence_install(*this);
        mask_sequence_prepared = true;
    }
};

template<class Job>
struct MaskShardHighCapLptJobsProxy {
    std::shared_ptr<MaskShardHighCapLptState<Job>> state;

    const std::vector<std::size_t>& operator[](std::size_t gpu) const {
        if (!state || !state->mask_sequence_prepared) {
            std::cerr << "HIGH graph mask sequence schedule used before prepare\n";
            std::exit(422);
        }
        const int row = G_MS_HIGH_CAP_LPT_ROW.load(std::memory_order_acquire);
        if (row < 0 || row >= TARGET_W || gpu >= std::size_t(state->ngpu)) {
            std::cerr << "HIGH graph mask sequence invalid row/GPU row="
                      << row << " gpu=" << gpu << '\n';
            std::exit(423);
        }
        const int cap = std::min(
            row + 1, MaskShardHighCapLptState<Job>::FULL_CAP);
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
static MaskShardHighCapLptSchedule<Job>
maskshard_build_high_cap_lpt_schedule(
    const std::vector<Job>& jobs, int ngpu
) {
    auto state = std::make_shared<MaskShardHighCapLptState<Job>>();
    state->jobs = &jobs;
    state->ngpu = ngpu;
    state->baseline = maskshard_build_high_static_lpt_schedule_v074(jobs, ngpu);

    MaskShardHighCapLptSchedule<Job> out;
    out.state = state;
    out.jobs_by_gpu.state = state;
    out.total_work = state->baseline.total_work;
    out.min_work = state->baseline.min_work;
    out.max_work = state->baseline.max_work;
    return out;
}

static void maskshard_set_row_depth_fblock_io_row_mask_sequence(
    int zero_based_row
) {
    maskshard_set_row_depth_fblock_io_row_cap_lpt_v078(zero_based_row);
    maskshard_high_graph_mask_sequence_set_row(zero_based_row);
}

#define maskshard_build_high_static_lpt_schedule \
        maskshard_build_high_cap_lpt_schedule
#define maskshard_prepare_highclosure_rowdepth_compact(high_desc, ngpu) \
        maskshard_prepare_highclosure_rowdepth_compact_cap_lpt_v078( \
            high_desc, ngpu, high_schedule)
#define maskshard_set_row_depth_fblock_io_row \
        maskshard_set_row_depth_fblock_io_row_mask_sequence
