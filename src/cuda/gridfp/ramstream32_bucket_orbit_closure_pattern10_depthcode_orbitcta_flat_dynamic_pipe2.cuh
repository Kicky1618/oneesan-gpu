#pragma once

// Included after flat_dynamic.cuh while the orbit-CTA hook macros are active.
#ifndef P10DC_ORBITCTA_CTX
#error "dynamic pipe2 requires P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH
#define P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH 1
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP
#define P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP 0
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
#define P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP 0
#endif
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 1 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 2 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 4 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 8 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 16,
              "dynamic pipe2 batch must be 1,2,4,8,16");
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP == 0 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP == 1,
              "dynamic pipe2 producer warp must be 0/1");
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP == 0 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP == 1,
              "dynamic pipe2 producer prectx warpcoop must be 0/1");
static_assert(!P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP,
              "producer prectx warpcoop requires producer warp");
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_warp.cuh"
#endif
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_prectx_warpcoop.cuh"
#endif

// The ordinary dynamic scheduler serializes
//   lane0 prepare -> CTA barrier -> columns -> CTA barrier
// for every orbit. Keep two resolved contexts instead. While the CTA consumes
// current, the producer prepares next. One CTA barrier then retires current and
// publishes next. With producer-prectx warpcoop, lanes 0..8 share compact row
// expansion while the worker warps consume current.
//
// IMPORTANT: "has an orbit id" and "the resolved orbit is executable" are
// different states. prepare_* may deliberately leave c.valid=0 for an orbit
// that must be skipped. The ordinary dynamic scheduler still advances past that
// orbit. Pipe2 therefore keeps a separate has_item bit for each context; using
// c.valid as the end-of-queue sentinel would abandon the rest of an acquired
// lease and is not exact.
//
// cp.async pair/quad scratch already occupies the complete region returned by
// p10dc_direct_warpctx_smem_bytes(). Context #1 lives after that region. The
// flat graph launcher must add sizeof(P10DC_ORBITCTA_CTX) after the aligned
// ctx1 offset; see p10dc_orbitcta_flat_high_smem_bytes().
__host__ __device__ static inline size_t
p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes_for_threads(int threads) {
    size_t n = 0;
#if P10DC_RANKFORMULA_CPASYNC_PAIR
    n = p10dc_direct_pair_scratch_offset_bytes(threads) +
        size_t(threads) * P10DC_RANKFORMULA_CPASYNC_VALUES_PER_THREAD * sizeof(Count);
#else
    n = sizeof(P10DC_ORBITCTA_CTX);
#endif
    return (n + alignof(P10DC_ORBITCTA_CTX) - 1u) &
           ~(size_t(alignof(P10DC_ORBITCTA_CTX)) - 1u);
}

__device__ __forceinline__ size_t p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes() {
    return p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes_for_threads(int(blockDim.x));
}

__device__ __forceinline__ P10DC_ORBITCTA_CTX&
p10dc_orbitcta_flat_dynamic_pipe2_context(
    unsigned long long* storage, uint32_t which
) {
    if (which == 0u) return *reinterpret_cast<P10DC_ORBITCTA_CTX*>(storage);
    auto* raw = reinterpret_cast<unsigned char*>(storage);
    return *reinterpret_cast<P10DC_ORBITCTA_CTX*>(
        raw + p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes());
}

__device__ __forceinline__ uint32_t p10dc_orbitcta_flat_dynamic_pipe2_next_k(
    uint32_t total,
    uint32_t lease_batch,
    uint32_t& lease_base_lane0,
    uint32_t& lease_pos_lane0
) {
    if (lease_pos_lane0 < lease_batch) {
        const uint32_t k = lease_base_lane0 + lease_pos_lane0;
        ++lease_pos_lane0;
        return k < total ? k : 0xffffffffu;
    }
    lease_base_lane0 = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, lease_batch);
    lease_pos_lane0 = 0u;
    if (lease_base_lane0 >= total) return 0xffffffffu;
    lease_pos_lane0 = 1u;
    return lease_base_lane0;
}

__device__ __forceinline__ void p10dc_orbitcta_flat_dynamic_pipe2_forward_columns(
    P10DC_ORBITCTA_CTX& c
) {
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP
    p10dc_orbitcta_flat_forward_columns_pipe2_producer_warp(c);
#else
    p10dc_orbitcta_flat_forward_columns(c);
#endif
}

__device__ __forceinline__ void p10dc_orbitcta_flat_dynamic_pipe2_reverse_columns(
    P10DC_ORBITCTA_CTX& c, bool edge
) {
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP
    p10dc_orbitcta_flat_reverse_columns_pipe2_producer_warp(c, edge);
#else
    p10dc_orbitcta_flat_reverse_columns(c, edge);
#endif
}

__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t base_off = pi * D_BKF_HIGH_PITCH;
    extern __shared__ unsigned long long storage[];
    __shared__ uint32_t has_item[2];
    P10DC_ORBITCTA_CTX& c0 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 0u);
    P10DC_ORBITCTA_CTX& c1 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 1u);

    uint32_t lease_base_lane0 = 0u;
    uint32_t lease_pos_lane0 = 0u;
    uint32_t lease_batch_lane0 = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
    uint32_t first_k_lane0 = 0xffffffffu;
    if (threadIdx.x == 0) {
        const uint32_t nn0 = D_BKF_HIGH_NN_OFF[base_off];
        const uint32_t nn1 = D_BKF_HIGH_NN_OFF[base_off + nblocks];
        const uint32_t nr0 = D_BKF_HIGH_NRNL_OFF[base_off];
        const uint32_t nr1 = D_BKF_HIGH_NRNL_OFF[base_off + nblocks];
        c0.n0 = c1.n0 = nn1 - nn0;
        c0.n1 = c1.n1 = nr0;
        c0.total = c1.total = c0.n0 + (nr1 - nr0);
        c0.valid = c1.valid = 0;
        has_item[0] = has_item[1] = 0u;
        lease_batch_lane0 = p10dc_orbitcta_flat_dynamic_effective_batch(c0.total);
        lease_base_lane0 = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, lease_batch_lane0);
        if (lease_base_lane0 < c0.total) {
            has_item[0] = 1u;
            lease_pos_lane0 = 1u;
            first_k_lane0 = lease_base_lane0;
#if !P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
            p10dc_orbitcta_flat_dynamic_prepare_forward(
                c0, first_k_lane0, base_off, nblocks, p);
#endif
        }
    }
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
    if (threadIdx.x < 32u)
        p10dc_orbitcta_flat_dynamic_pipe2_prepare_forward_producer_prectx_warpcoop(
            c0, first_k_lane0, base_off, nblocks, p);
#endif
    __syncthreads();

    const uint32_t total = c0.total;
    uint32_t cur = 0u;
    for (;;) {
        P10DC_ORBITCTA_CTX& current =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur);
        P10DC_ORBITCTA_CTX& next =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur ^ 1u);
        if (!has_item[cur]) break;

        uint32_t next_k_lane0 = 0xffffffffu;
        if (threadIdx.x == 0) {
            next.valid = 0;
            has_item[cur ^ 1u] = 0u;
            next_k_lane0 = p10dc_orbitcta_flat_dynamic_pipe2_next_k(
                total, lease_batch_lane0, lease_base_lane0, lease_pos_lane0);
            if (next_k_lane0 != 0xffffffffu) {
                has_item[cur ^ 1u] = 1u;
#if !P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
                p10dc_orbitcta_flat_dynamic_prepare_forward(
                    next, next_k_lane0, base_off, nblocks, p);
#endif
            }
        }
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
        if (threadIdx.x < 32u)
            p10dc_orbitcta_flat_dynamic_pipe2_prepare_forward_producer_prectx_warpcoop(
                next, next_k_lane0, base_off, nblocks, p);
#endif

        if (current.valid == 1)
            p10dc_orbitcta_flat_dynamic_pipe2_forward_columns(current);
        __syncthreads();
        cur ^= 1u;
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    const uint32_t base_off = pi * D_RS54_PITCH;
    const bool edge = p == TARGET_W - 1;
    extern __shared__ unsigned long long storage[];
    __shared__ uint32_t has_item[2];
    P10DC_ORBITCTA_CTX& c0 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 0u);
    P10DC_ORBITCTA_CTX& c1 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 1u);

    uint32_t lease_base_lane0 = 0u;
    uint32_t lease_pos_lane0 = 0u;
    uint32_t lease_batch_lane0 = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
    uint32_t first_k_lane0 = 0xffffffffu;
    if (threadIdx.x == 0) {
        const uint32_t nn0 = D_RS54_HIGH_NN_OFF[base_off];
        const uint32_t nn1 = D_RS54_HIGH_NN_OFF[base_off + nblocks];
        const uint32_t nr0 = D_RS54_HIGH_NR_OFF[base_off];
        const uint32_t nr1 = D_RS54_HIGH_NR_OFF[base_off + nblocks];
        const uint32_t nl0 = D_RS54_HIGH_NL_OFF[base_off];
        const uint32_t nl1 = D_RS54_HIGH_NL_OFF[base_off + nblocks];
        c0.n0 = c1.n0 = nn1 - nn0;
        c0.n1 = c1.n1 = nr1 - nr0;
        c0.total = c1.total = c0.n0 + c0.n1 + (nl1 - nl0);
        c0.valid = c1.valid = 0;
        has_item[0] = has_item[1] = 0u;
        lease_batch_lane0 = p10dc_orbitcta_flat_dynamic_effective_batch(c0.total);
        lease_base_lane0 = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, lease_batch_lane0);
        if (lease_base_lane0 < c0.total) {
            has_item[0] = 1u;
            lease_pos_lane0 = 1u;
            first_k_lane0 = lease_base_lane0;
#if !P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
            p10dc_orbitcta_flat_dynamic_prepare_reverse(
                c0, first_k_lane0, base_off, nblocks, p, edge);
#endif
        }
    }
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
    if (threadIdx.x < 32u)
        p10dc_orbitcta_flat_dynamic_pipe2_prepare_reverse_producer_prectx_warpcoop(
            c0, first_k_lane0, base_off, nblocks, p, edge);
#endif
    __syncthreads();

    const uint32_t total = c0.total;
    uint32_t cur = 0u;
    for (;;) {
        P10DC_ORBITCTA_CTX& current =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur);
        P10DC_ORBITCTA_CTX& next =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur ^ 1u);
        if (!has_item[cur]) break;

        uint32_t next_k_lane0 = 0xffffffffu;
        if (threadIdx.x == 0) {
            next.valid = 0;
            has_item[cur ^ 1u] = 0u;
            next_k_lane0 = p10dc_orbitcta_flat_dynamic_pipe2_next_k(
                total, lease_batch_lane0, lease_base_lane0, lease_pos_lane0);
            if (next_k_lane0 != 0xffffffffu) {
                has_item[cur ^ 1u] = 1u;
#if !P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
                p10dc_orbitcta_flat_dynamic_prepare_reverse(
                    next, next_k_lane0, base_off, nblocks, p, edge);
#endif
            }
        }
#if P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP
        if (threadIdx.x < 32u)
            p10dc_orbitcta_flat_dynamic_pipe2_prepare_reverse_producer_prectx_warpcoop(
                next, next_k_lane0, base_off, nblocks, p, edge);
#endif

        if (current.valid == 1)
            p10dc_orbitcta_flat_dynamic_pipe2_reverse_columns(current, edge);
        __syncthreads();
        cur ^= 1u;
    }
}
