#pragma once

#ifndef P10DC_ORBITCTA_CTX
#error "flat quad columns require P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_PLAN_SUM
#error "flat quad columns require scalar plan-sum hook"
#endif
#ifndef P10DC_ORBITCTA_PLAN_SUM_QUAD
#error "flat quad columns require quad plan-sum hook"
#endif
#ifndef P10DC_ORBITCTA_COL_ILP
#define P10DC_ORBITCTA_COL_ILP 4
#endif
static_assert(P10DC_ORBITCTA_COL_ILP == 4,
              "flat quad columns require ORBITCTA_COL_ILP=4");

__device__ __forceinline__ void p10dc_orbitcta_flat_forward_columns(
    P10DC_ORBITCTA_CTX& c
) {
    constexpr int ILP = 4;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t lane_step = uint32_t(blockDim.x);
    const uint32_t group_step = lane_step * 4u;
    for (uint32_t base = uint32_t(threadIdx.x); base < cols; base += group_step) {
        uint32_t lr[ILP]{};
        uint8_t valid[ILP]{};
        Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            lr[j] = base + uint32_t(j) * lane_step;
            valid[j] = uint8_t(lr[j] < cols);
            if (valid[j]) {
                x[j] = ip_base[lr[j]];
                old[j] = dp_base[lr[j]];
#if P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
            }
        }
        if (valid[3]) {
            P10DC_ORBITCTA_PLAN_SUM_QUAD(
                c, c.db, lr[0], lr[1], lr[2], lr[3],
                extra[0], extra[1], extra[2], extra[3]);
        } else {
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
            if (valid[0] && valid[1])
                P10DC_ORBITCTA_PLAN_SUM_PAIR(c, c.db, lr[0], lr[1], extra[0], extra[1]);
            else if (valid[0])
                extra[0] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[0]);
            if (valid[2] && valid[3])
                P10DC_ORBITCTA_PLAN_SUM_PAIR(c, c.db, lr[2], lr[3], extra[2], extra[3]);
            else if (valid[2])
                extra[2] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[2]);
#else
#pragma unroll
            for (int j = 0; j < ILP; ++j)
                if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
#endif
        }
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

__device__ __forceinline__ void p10dc_orbitcta_flat_reverse_columns(
    P10DC_ORBITCTA_CTX& c, bool edge
) {
    constexpr int ILP = 4;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t lane_step = uint32_t(blockDim.x);
    const uint32_t group_step = lane_step * 4u;
    const BucketPhysicalBlock& sum_db = edge ? c.xb : c.db;
    for (uint32_t base = uint32_t(threadIdx.x); base < cols; base += group_step) {
        uint32_t lr[ILP]{};
        uint8_t valid[ILP]{};
        Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            lr[j] = base + uint32_t(j) * lane_step;
            valid[j] = uint8_t(lr[j] < cols);
            if (valid[j]) {
                x[j] = ip_base[lr[j]];
                old[j] = dp_base[lr[j]];
#if P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
            }
        }
        if (valid[3]) {
            P10DC_ORBITCTA_PLAN_SUM_QUAD(
                c, sum_db, lr[0], lr[1], lr[2], lr[3],
                extra[0], extra[1], extra[2], extra[3]);
        } else {
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
            if (valid[0] && valid[1])
                P10DC_ORBITCTA_PLAN_SUM_PAIR(c, sum_db, lr[0], lr[1], extra[0], extra[1]);
            else if (valid[0])
                extra[0] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[0]);
            if (valid[2] && valid[3])
                P10DC_ORBITCTA_PLAN_SUM_PAIR(c, sum_db, lr[2], lr[3], extra[2], extra[3]);
            else if (valid[2])
                extra[2] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[2]);
#else
#pragma unroll
            for (int j = 0; j < ILP; ++j)
                if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
#endif
        }
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
