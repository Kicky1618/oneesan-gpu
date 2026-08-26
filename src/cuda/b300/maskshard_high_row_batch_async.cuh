#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>

#ifndef MASKSHARD_HIGH_ROW_BATCH_ASYNC
#error "HIGH row-batch async header requires MASKSHARD_HIGH_ROW_BATCH_ASYNC"
#endif
#ifndef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#error "HIGH row-batch async requires compact HIGH orbit metadata"
#endif
#ifndef MASKSHARD_HIGH_GROUP_SIZE_CACHE
#error "HIGH row-batch async requires v0.60 cached BLOCKED group sizes"
#endif

// The original compact-orbit host helper constructs two stack arrays and uses
// synchronous cudaMemcpyToSymbol() for every HIGH job. That would both break
// the source lifetime required by an asynchronous row batch and reintroduce a
// host/device barrier between jobs. Precompute every unsaturated (mask, cap)
// plan once, retain it for the process lifetime, and enqueue both constant
// updates on stream 0. At full cap the compact kernel uses physical BLOCKED
// order and reads neither plan array, so no plan copy is needed at all.
struct MaskShardHighRowBatchPlan {
    std::array<Code, HIGH_LUT_K + 3> prefix{};
    std::array<std::uint16_t, HIGH_LUT_K + 2> low_count{};
    Code total = 0;
};

struct MaskShardHighRowBatchPlanCache {
    static constexpr int FULL_CAP = TARGET_W / 2;
    static constexpr int CAP_STRIDE = FULL_CAP;
    static constexpr std::uint32_t NMASK = 1u << LOW_LUT_K;
    static constexpr std::size_t PLAN_COUNT =
        std::size_t(NMASK) * std::size_t(CAP_STRIDE);

#ifdef MASKSHARD_HIGH_PINNED_CONFIG
    MaskShardHighRowBatchPlan* plan = nullptr;
#else
    std::vector<MaskShardHighRowBatchPlan> plan;
#endif
    bool built = false;

    static std::size_t index(std::uint32_t mask, int cap) {
        return std::size_t(mask) * CAP_STRIDE + std::size_t(cap);
    }

    MaskShardHighRowBatchPlan& at(std::size_t i) {
#ifdef MASKSHARD_HIGH_PINNED_CONFIG
        return plan[i];
#else
        return plan[i];
#endif
    }
    const MaskShardHighRowBatchPlan& at(std::size_t i) const {
#ifdef MASKSHARD_HIGH_PINNED_CONFIG
        return plan[i];
#else
        return plan[i];
#endif
    }

    void allocate() {
#ifdef MASKSHARD_HIGH_PINNED_CONFIG
        if (plan) return;
        void* p = nullptr;
        ck(cudaHostAlloc(&p, PLAN_COUNT * sizeof(MaskShardHighRowBatchPlan),
                         cudaHostAllocPortable),
           "HIGH row-batch pinned compact plans");
        plan = static_cast<MaskShardHighRowBatchPlan*>(p);
        for (std::size_t i = 0; i < PLAN_COUNT; ++i)
            ::new (static_cast<void*>(plan + i)) MaskShardHighRowBatchPlan{};
#else
        plan.resize(PLAN_COUNT);
#endif
    }

    void build() {
        if (built) return;
        auto& compact = maskshard_row_depth_orbit_compact_cache();
        compact.build();
        allocate();
        for (std::uint32_t mask = 0; mask < NMASK; ++mask) {
            for (int cap = 1; cap < FULL_CAP; ++cap) {
                auto& p = at(index(mask, cap));
                p.total = compact.make_job_plan(
                    mask, cap, p.prefix, p.low_count);
            }
        }
        built = true;
        std::cerr << "HIGH row-batch compact-plan cache entries=" << PLAN_COUNT
                  << " plan_bytes=" << sizeof(MaskShardHighRowBatchPlan)
                  << " host_mib="
                  << double(PLAN_COUNT * sizeof(MaskShardHighRowBatchPlan))
                       / double(1ULL << 20)
                  << " saturated_plan_copies=0"
#ifdef MASKSHARD_HIGH_PINNED_CONFIG
                  << " pinned=1 contiguous=1"
#else
                  << " pinned=0"
#endif
                  << '\n';
    }
};

static MaskShardHighRowBatchPlanCache G_MS_HIGH_ROW_BATCH_PLAN_CACHE{};

static void maskshard_report_high_mask_shard_layout_row_batch_async(
    const MaskShardLayout& s
) {
    report_high_mask_shard_layout(s);
    G_MS_HIGH_ROW_BATCH_PLAN_CACHE.build();
}
#ifdef report_high_mask_shard_layout
#undef report_high_mask_shard_layout
#endif
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_row_batch_async

static Code maskshard_configure_row_depth_compact_group_row_batch_async(
    std::uint32_t mask, int cap
) {
    auto& cache = G_MS_HIGH_ROW_BATCH_PLAN_CACHE;
    if (!cache.built || mask >= cache.NMASK) {
        std::cerr << "HIGH row-batch compact plan unavailable mask=" << mask << '\n';
        std::exit(367);
    }
    cap = std::max(0, std::min(cap, cache.FULL_CAP));
    if (cap >= cache.FULL_CAP)
        return G_MS_HIGH_GROUP_SIZE_CACHE.block_size[mask];

    const auto& p = cache.at(cache.index(mask, cap));
    ck(cudaMemcpyToSymbolAsync(
           D_MS_ROW_DEPTH_COMPACT_TASK_PREFIX,
           p.prefix.data(), sizeof(p.prefix), 0,
           cudaMemcpyHostToDevice, cudaStream_t(0)),
       "HIGH row-batch compact task prefix");
    ck(cudaMemcpyToSymbolAsync(
           D_MS_ROW_DEPTH_COMPACT_JOB_LOW_COUNT,
           p.low_count.data(), sizeof(p.low_count), 0,
           cudaMemcpyHostToDevice, cudaStream_t(0)),
       "HIGH row-batch compact LOW counts");
    return p.total;
}

// Redirect only the shared host call parsed after this header. The original
// v0.19 helper remains intact for all older variants.
#define maskshard_configure_row_depth_compact_group \
        maskshard_configure_row_depth_compact_group_row_batch_async
