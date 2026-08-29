#pragma once

#ifndef P10DC_ORBITCTA_CTX
#error "pipe2 producer-warp columns require P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_COL_ILP
#define P10DC_ORBITCTA_COL_ILP 1
#endif
static_assert(P10DC_ORBITCTA_COL_ILP == 1 || P10DC_ORBITCTA_COL_ILP == 2 ||
              P10DC_ORBITCTA_COL_ILP == 4,
              "pipe2 producer-warp columns require ILP 1,2,4");

// Warp 0 is the producer. Remaining whole warps cover every column with a
// compact logical thread id, preserving 32-lane contiguous accesses inside each
// worker warp. This removes the standard pipe2 tail where warp 0 starts its own
// 1/8 share only after lane 0 finishes preparing the next orbit. A partial last
// warp would violate that contract for warp-level plan-sum helpers, so non-warp-
// aligned block sizes fall back to the ordinary exact column executor.
__device__ __forceinline__ void p10dc_orbitcta_flat_forward_columns_pipe2_producer_warp(
    P10DC_ORBITCTA_CTX& c
) {
    if (blockDim.x <= 32 || (uint32_t(blockDim.x) & 31u) != 0u) {
        p10dc_orbitcta_flat_forward_columns(c);
        return;
    }
    if (threadIdx.x < 32u) return;
    constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
    const uint32_t worker = uint32_t(threadIdx.x) - 32u;
    const uint32_t workers = uint32_t(blockDim.x) - 32u;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t group_step = workers * uint32_t(ILP);
    for (uint32_t base = worker; base < cols; base += group_step) {
        uint32_t lr[ILP]{};
        uint8_t valid[ILP]{};
        Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            lr[j] = base + uint32_t(j) * workers;
            valid[j] = uint8_t(lr[j] < cols);
            if (valid[j]) {
                x[j] = ip_base[lr[j]];
                old[j] = dp_base[lr[j]];
#if P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
            }
        }
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
#pragma unroll
        for (int j = 0; j < ILP; j += 2) {
            if constexpr (ILP > 1) {
                if (valid[j] && valid[j + 1]) {
                    P10DC_ORBITCTA_PLAN_SUM_PAIR(
                        c, c.db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                } else {
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
                    if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j + 1]);
                }
            } else if (valid[j]) {
                extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
#endif
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (!valid[j]) continue;
            if (kind == CPU_ORBIT_NN) {
#if P10DC_ORBITCTA_EARLY_JP
                jp_base[lr[j]] = gpu_direct_add(y[j], x[j]);
#else
                jp_base[lr[j]] = gpu_direct_add(jp_base[lr[j]], x[j]);
#endif
                ip_base[lr[j]] = gpu_direct_add(x[j], old[j]);
                dp_base[lr[j]] = extra[j];
            } else {
#if !P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
                ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], y[j]), old[j]);
                dp_base[lr[j]] = gpu_direct_add(x[j], extra[j]);
            }
        }
    }
}

__device__ __forceinline__ void p10dc_orbitcta_flat_reverse_columns_pipe2_producer_warp(
    P10DC_ORBITCTA_CTX& c, bool edge
) {
    if (blockDim.x <= 32 || (uint32_t(blockDim.x) & 31u) != 0u) {
        p10dc_orbitcta_flat_reverse_columns(c, edge);
        return;
    }
    if (threadIdx.x < 32u) return;
    constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
    const uint32_t worker = uint32_t(threadIdx.x) - 32u;
    const uint32_t workers = uint32_t(blockDim.x) - 32u;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t group_step = workers * uint32_t(ILP);
    const BucketPhysicalBlock& sum_db = edge ? c.xb : c.db;
    for (uint32_t base = worker; base < cols; base += group_step) {
        uint32_t lr[ILP]{};
        uint8_t valid[ILP]{};
        Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            lr[j] = base + uint32_t(j) * workers;
            valid[j] = uint8_t(lr[j] < cols);
            if (valid[j]) {
                x[j] = ip_base[lr[j]];
                old[j] = dp_base[lr[j]];
#if P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
            }
        }
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
#pragma unroll
        for (int j = 0; j < ILP; j += 2) {
            if constexpr (ILP > 1) {
                if (valid[j] && valid[j + 1]) {
                    P10DC_ORBITCTA_PLAN_SUM_PAIR(
                        c, sum_db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                } else {
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
                    if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j + 1]);
                }
            } else if (valid[j]) {
                extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
#endif
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (!valid[j]) continue;
            if (kind == CPU_ORBIT_NN) {
#if P10DC_ORBITCTA_EARLY_JP
                jp_base[lr[j]] = gpu_direct_add(y[j], x[j]);
#else
                jp_base[lr[j]] = gpu_direct_add(jp_base[lr[j]], x[j]);
#endif
                ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], old[j]), edge ? extra[j] : 0);
                dp_base[lr[j]] = edge ? 0 : extra[j];
            } else {
#if !P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
                ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], y[j]), old[j]);
                if (edge) {
                    jp_base[lr[j]] = gpu_direct_add(x[j], y[j]);
                    dp_base[lr[j]] = 0;
                } else {
                    dp_base[lr[j]] = gpu_direct_add(x[j], extra[j]);
                }
            }
        }
    }
}
