#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32_directmask.cuh"

#ifndef P10DC_WARPSTRIPED_ILP
#define P10DC_WARPSTRIPED_ILP 1
#endif
static_assert(P10DC_WARPSTRIPED_ILP == 1 || P10DC_WARPSTRIPED_ILP == 2 ||
              P10DC_WARPSTRIPED_ILP == 4,
              "P10DC_WARPSTRIPED_ILP must be 1, 2, or 4");

#if P10DC_RANKCHUNK32_DIRECTMASK && P10DC_WARPSTRIPED_ILP > 1
static_assert(P10DC_RANKCHUNK32_ALIGN32,
              "directmask ILP>1 requires height-aligned rankchunk metadata");
static_assert(P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE <= 7u,
              "directmask ILP assumes the global ordinal mask fits in 7 bits");

// Resolve ILP independent LOW-rank streams together. With height alignment,
// each 32-lane stripe is contained in one metadata block (also for block64:
// a stripe starts at offset 0 or 32), so one elected lane can broadcast the
// block base for each independent stripe.
template<int ILP>
__device__ __forceinline__ void p10dc_rankchunk32_directmask_rows_ilp(
    uint32_t h, const uint32_t (&lr)[ILP], const uint8_t (&need)[ILP],
    const uint16_t* (&row)[ILP]
) {
    static_assert(ILP == 2 || ILP == 4);
    const unsigned active = __activemask();
    const unsigned lane = unsigned(threadIdx.x) & 31u;

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
        row[j] = nullptr;
        const unsigned wanted = __ballot_sync(active, need[j] != 0u);
        uint32_t prefix = 0u, block_index = 0u, local_base = 0u;
        if (need[j]) {
            const uint32_t compact = D_P10DC_LOW_RANKCHUNK_HOFF[h] + lr[j];
            const uint32_t meta = __ldg(D_P10DC_LOW_RANKCHUNKMETA32 + compact);
            prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;
            block_index = compact >> P10DC_RANKCHUNK32_BLOCK_LOG2;
        }
        if (wanted) {
            const unsigned leader = unsigned(__ffs(int(wanted)) - 1);
            if (need[j] && lane == leader)
                local_base = __ldg(D_P10DC_LOW_RANKCHUNKBLOCK16 + block_index);
            const uint32_t block_base = __shfl_sync(active, local_base, int(leader));
            if (need[j]) row[j] = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
        }
    }
}

// Batch the independent memory chains in lock-step:
//   local rows: issue ILP Count loads before consuming them;
//   directmask: issue ILP byte loads;
//   rank rows: issue ILP rank16 loads for an ordinal;
//   source rows: issue ILP source32 loads before reducing them.
// This deliberately spends registers to expose memory-level parallelism on B300.
template<int ILP>
__device__ __forceinline__ void p10dc_direct_resolved_high_plan_sum_rankchunk32_directmask_ilp(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db,
    const uint32_t (&lr)[ILP], const uint8_t (&valid)[ILP], Count (&out)[ILP]
) {
    static_assert(ILP == 2 || ILP == 4);
#if GPU_DIRECT_PM_ACCUM
    uint64_t accum[ILP]{};
#else
    Count accum[ILP]{};
#endif

#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < c.local_n) {
            Count v[ILP]{};
#pragma unroll
            for (int j = 0; j < ILP; ++j)
                if (valid[j]) v[j] = c.local_base[i][lr[j]];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                if (!valid[j]) continue;
#if GPU_DIRECT_PM_ACCUM
                accum[j] += uint64_t(v[j]);
#else
                accum[j] = gpu_direct_add(accum[j], v[j]);
#endif
            }
        }
    }

    if (c.cross_depth) {
        uint8_t pending[ILP]{};
        uint8_t need[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (valid[j]) {
                pending[j] = p10dc_low_rankchunk32_directmask_load(
                    db.hs, lr[j], c.cross_depth);
                need[j] = pending[j] != 0u;
            }
        }

        const uint16_t* rank_row[ILP]{};
        p10dc_rankchunk32_directmask_rows_ilp<ILP>(db.hs, lr, need, rank_row);

#pragma unroll
        for (uint32_t ordinal = 0;
             ordinal < P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE; ++ordinal) {
            uint16_t source_rank[ILP]{};
            uint8_t take[ILP]{};
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                take[j] = uint8_t(need[j] &&
                    ((pending[j] & uint8_t(1u << ordinal)) != 0u));
                if (take[j]) source_rank[j] = rank_row[j][ordinal];
            }

            Count v[ILP]{};
#pragma unroll
            for (int j = 0; j < ILP; ++j)
                if (take[j]) v[j] = c.cross_base[uint32_t(source_rank[j])];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                if (!take[j]) continue;
#if GPU_DIRECT_PM_ACCUM
                accum[j] += uint64_t(v[j]);
#else
                accum[j] = gpu_direct_add(accum[j], v[j]);
#endif
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
        if (!valid[j]) {
            out[j] = 0;
            continue;
        }
#if GPU_DIRECT_PM_ACCUM
        out[j] = gpu_direct_pm_reduce_u64(accum[j]);
#else
        out[j] = accum[j];
#endif
    }
}
#endif
