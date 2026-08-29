#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32_directmask.cuh"

#ifndef P10DC_WARPSTRIPED_ILP
#define P10DC_WARPSTRIPED_ILP 1
#endif
static_assert(P10DC_WARPSTRIPED_ILP == 1 || P10DC_WARPSTRIPED_ILP == 2 ||
              P10DC_WARPSTRIPED_ILP == 4,
              "P10DC_WARPSTRIPED_ILP must be 1, 2, or 4");

#if P10DC_RANKCHUNK32_DIRECTMASK && P10DC_WARPSTRIPED_ILP > 1
static_assert(P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE <= 7u,
              "directmask ILP assumes the global ordinal mask fits in 7 bits");

// Batch independent memory chains in lock-step. Directmask mode carries an
// absolute rankstream offset per LOW rank, so the ILP path no longer touches
// rankchunk metadata, block-base tables, or warp shuffles at all. All selected
// rank16 entries are prefetched before beginning source32 gathers, removing the
// rank16 latency from the source-gather issue loop at the cost of registers.
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
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            if (valid[j])
                pending[j] = p10dc_low_rankchunk32_directmask_load(
                    db.hs, lr[j], c.cross_depth);

        const uint16_t* rank_row[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            if (pending[j])
                rank_row[j] = p10dc_low_rankchunk32_directoff_row(db.hs, lr[j]);

        constexpr uint32_t MAXR = P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE;
        uint16_t source_rank[ILP][MAXR]{};
#pragma unroll
        for (uint32_t ordinal = 0; ordinal < MAXR; ++ordinal) {
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                if (pending[j] & uint8_t(1u << ordinal))
                    source_rank[j][ordinal] = rank_row[j][ordinal];
            }
        }

#pragma unroll
        for (uint32_t ordinal = 0; ordinal < MAXR; ++ordinal) {
            Count v[ILP]{};
            uint8_t take[ILP]{};
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                take[j] = uint8_t((pending[j] & uint8_t(1u << ordinal)) != 0u);
                if (take[j])
                    v[j] = c.cross_base[uint32_t(source_rank[j][ordinal])];
            }
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
