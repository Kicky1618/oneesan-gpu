#pragma once

#ifndef P10DC_ORBITCTA_CTX
#error "pipe2 producer-warp columns require P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_COL_ILP
#define P10DC_ORBITCTA_COL_ILP 1
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT
#define P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT 0
#endif
static_assert(P10DC_ORBITCTA_COL_ILP == 1 || P10DC_ORBITCTA_COL_ILP == 2 ||
              P10DC_ORBITCTA_COL_ILP == 4,
              "pipe2 producer-warp columns require ILP 1,2,4");
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT >= 0 &&
              P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT <= 4,
              "pipe2 producer worker weight must be 0..4");
#ifdef P10DC_ORBITCTA_PLAN_SUM_QUAD
static_assert(P10DC_ORBITCTA_COL_ILP == 4,
              "producer-warp quad plan sum requires ILP=4");
#endif

// Warp 0 resolves the next context before entering the current column loop.
// weight=0 is the existing producer-only mode: warp 0 owns no current columns
// and the remaining warps cover the block compactly. weight>0 gives warp 0 one
// virtual warp-slot after prepare while each worker warp owns WEIGHT slots:
//
//   producer share = 1 / (1 + (nwarps-1)*WEIGHT).
//
// For 256 threads this is 1/8, 1/15, 1/22, 1/29 for weights 1..4, bridging
// standard PIPE2 and producer-only instead of forcing a 12.5% -> 0% jump.
// Every virtual slot is exactly 32 lanes, so each column has a unique logical
// worker and pair/quad gather lane contiguity is preserved.
__device__ __forceinline__ void p10dc_orbitcta_flat_pipe2_producer_partition(
    uint32_t& first_slot, uint32_t& slot_count, uint32_t& logical_workers
) {
    const uint32_t tid = uint32_t(threadIdx.x);
    const uint32_t warp = tid >> 5;
    const uint32_t nwarps = uint32_t(blockDim.x) >> 5;
    constexpr uint32_t WEIGHT =
        uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT);
    if constexpr (WEIGHT == 0u) {
        if (warp == 0u) {
            first_slot = 0u;
            slot_count = 0u;
            logical_workers = (nwarps - 1u) << 5;
            return;
        }
        first_slot = warp - 1u;
        slot_count = 1u;
        logical_workers = (nwarps - 1u) << 5;
    } else {
        if (warp == 0u) {
            first_slot = 0u;
            slot_count = 1u;
        } else {
            first_slot = 1u + (warp - 1u) * WEIGHT;
            slot_count = WEIGHT;
        }
        logical_workers = (1u + (nwarps - 1u) * WEIGHT) << 5;
    }
}

__device__ __forceinline__ void p10dc_orbitcta_flat_forward_columns_pipe2_producer_warp(
    P10DC_ORBITCTA_CTX& c
) {
    if (blockDim.x <= 32 || (uint32_t(blockDim.x) & 31u) != 0u) {
        p10dc_orbitcta_flat_forward_columns(c);
        return;
    }
    constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    uint32_t first_slot = 0u, slot_count = 0u, workers = 0u;
    p10dc_orbitcta_flat_pipe2_producer_partition(first_slot, slot_count, workers);
    if (!slot_count) return;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t group_step = workers * uint32_t(ILP);
    for (uint32_t vs = 0; vs < slot_count; ++vs) {
        const uint32_t worker = (first_slot + vs) * 32u + lane;
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
#ifdef P10DC_ORBITCTA_PLAN_SUM_QUAD
            if (valid[0] && valid[1] && valid[2] && valid[3]) {
                P10DC_ORBITCTA_PLAN_SUM_QUAD(
                    c, c.db, lr[0], lr[1], lr[2], lr[3],
                    extra[0], extra[1], extra[2], extra[3]);
            } else {
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
#pragma unroll
                for (int j = 0; j < 4; j += 2) {
                    if (valid[j] && valid[j + 1]) {
                        P10DC_ORBITCTA_PLAN_SUM_PAIR(
                            c, c.db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                    } else {
                        if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
                        if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j + 1]);
                    }
                }
#else
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
#endif
            }
#elif defined(P10DC_ORBITCTA_PLAN_SUM_PAIR)
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
}

__device__ __forceinline__ void p10dc_orbitcta_flat_reverse_columns_pipe2_producer_warp(
    P10DC_ORBITCTA_CTX& c, bool edge
) {
    if (blockDim.x <= 32 || (uint32_t(blockDim.x) & 31u) != 0u) {
        p10dc_orbitcta_flat_reverse_columns(c, edge);
        return;
    }
    constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    uint32_t first_slot = 0u, slot_count = 0u, workers = 0u;
    p10dc_orbitcta_flat_pipe2_producer_partition(first_slot, slot_count, workers);
    if (!slot_count) return;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t group_step = workers * uint32_t(ILP);
    const BucketPhysicalBlock& sum_db = edge ? c.xb : c.db;
    for (uint32_t vs = 0; vs < slot_count; ++vs) {
        const uint32_t worker = (first_slot + vs) * 32u + lane;
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
#ifdef P10DC_ORBITCTA_PLAN_SUM_QUAD
            if (valid[0] && valid[1] && valid[2] && valid[3]) {
                P10DC_ORBITCTA_PLAN_SUM_QUAD(
                    c, sum_db, lr[0], lr[1], lr[2], lr[3],
                    extra[0], extra[1], extra[2], extra[3]);
            } else {
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
#pragma unroll
                for (int j = 0; j < 4; j += 2) {
                    if (valid[j] && valid[j + 1]) {
                        P10DC_ORBITCTA_PLAN_SUM_PAIR(
                            c, sum_db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                    } else {
                        if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
                        if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j + 1]);
                    }
                }
#else
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
#endif
            }
#elif defined(P10DC_ORBITCTA_PLAN_SUM_PAIR)
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
}
