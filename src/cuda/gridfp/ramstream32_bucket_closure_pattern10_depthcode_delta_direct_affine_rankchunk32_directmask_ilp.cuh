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

// Batch independent memory chains in lock-step. In rankplane mode each warp
// reads the same ordinal plane at consecutive compact indices, exposing a
// coalesced rank16 load before the random source32 gather. Without rankplane,
// keep the direct-offset path as an A/B fallback.
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
        uint32_t compact[ILP]{};
        uint8_t pending[ILP]{};
        uint8_t lane_any = 0u;
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (valid[j]) {
                compact[j] = p10dc_low_rankchunk32_directcompact(db.hs, lr[j]);
                pending[j] = p10dc_low_rankchunk32_directmask_load_compact(
                    compact[j], c.cross_depth);
            }
            lane_any = uint8_t(lane_any | uint8_t(pending[j] != 0u));
        }

        const unsigned active = __activemask();
        if (__any_sync(active, lane_any != 0u)) {
#if !P10DC_RANKCHUNK32_RANKPLANE
            const uint16_t* rank_row[ILP]{};
#pragma unroll
            for (int j = 0; j < ILP; ++j)
                if (pending[j])
                    rank_row[j] = p10dc_low_rankchunk32_directoff_row_compact(compact[j]);
#endif

            constexpr uint32_t NRANK = P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE;
            constexpr uint32_t RANK_STORAGE = NRANK ? NRANK : 1u;
            uint16_t source_rank[ILP][RANK_STORAGE]{};
#pragma unroll
            for (uint32_t ordinal = 0; ordinal < NRANK; ++ordinal) {
#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    if (pending[j] & uint8_t(1u << ordinal)) {
#if P10DC_RANKCHUNK32_RANKPLANE
                        source_rank[j][ordinal] =
                            p10dc_low_rankchunk32_directrank_load_compact(compact[j], ordinal);
#else
                        source_rank[j][ordinal] = rank_row[j][ordinal];
#endif
                    }
                }
            }

#pragma unroll
            for (uint32_t ordinal = 0; ordinal < NRANK; ++ordinal) {
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
