#pragma once

// Included after flat_dynamic.cuh while the orbit-CTA hook macros are active.
#ifndef P10DC_ORBITCTA_CTX
#error "dynamic pipe2 requires P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH
#define P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH 1
#endif
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 1 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 2 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 4 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 8 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 16,
              "dynamic pipe2 batch must be 1,2,4,8,16");

// The ordinary dynamic scheduler serializes
//   lane0 prepare -> CTA barrier -> columns -> CTA barrier
// for every orbit. Keep two resolved contexts instead. While the CTA consumes
// current, lane0 prepares next. Warp 0 executes that lane0 branch first, but the
// remaining resident warps can already issue current-orbit memory requests; when
// lane0 finishes it rejoins the ordinary column executor, so no column is lost.
// One CTA barrier then retires current and publishes next.
//
// cp.async pair/quad scratch already occupies the complete region returned by
// p10dc_direct_warpctx_smem_bytes(). Put context #1 *after* that region instead
// of inside its intentionally padded warp-context prefix.
__device__ __forceinline__ size_t p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes() {
    size_t n = 0;
#if P10DC_RANKFORMULA_CPASYNC_PAIR
    n = p10dc_direct_pair_scratch_offset_bytes(int(blockDim.x)) +
        size_t(blockDim.x) * P10DC_RANKFORMULA_CPASYNC_VALUES_PER_THREAD * sizeof(Count);
#else
    n = sizeof(P10DC_ORBITCTA_CTX);
#endif
    return (n + alignof(P10DC_ORBITCTA_CTX) - 1u) &
           ~(size_t(alignof(P10DC_ORBITCTA_CTX)) - 1u);
}

__device__ __forceinline__ P10DC_ORBITCTA_CTX*
p10dc_orbitcta_flat_dynamic_pipe2_contexts(unsigned long long* storage) {
    auto* c0 = reinterpret_cast<P10DC_ORBITCTA_CTX*>(storage);
    auto* raw = reinterpret_cast<unsigned char*>(storage);
    auto* c1 = reinterpret_cast<P10DC_ORBITCTA_CTX*>(
        raw + p10dc_orbitcta_flat_dynamic_pipe2_ctx1_offset_bytes());
    // The caller addresses c0/c1 individually; returning c0 only documents the
    // base ABI and keeps accidental array indexing across the scratch gap out.
    (void)c1;
    return c0;
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
    uint32_t& lease_base_lane0,
    uint32_t& lease_pos_lane0
) {
    constexpr uint32_t BATCH = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
    if (lease_pos_lane0 < BATCH) {
        const uint32_t k = lease_base_lane0 + lease_pos_lane0;
        ++lease_pos_lane0;
        // A partial final lease implies the global queue has already crossed
        // total; no subsequent lease can contain useful work.
        return k < total ? k : 0xffffffffu;
    }
    lease_base_lane0 = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, BATCH);
    lease_pos_lane0 = 0u;
    if (lease_base_lane0 >= total) return 0xffffffffu;
    lease_pos_lane0 = 1u;
    return lease_base_lane0;
}

__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t base_off = pi * D_BKF_HIGH_PITCH;
    extern __shared__ unsigned long long storage[];
    P10DC_ORBITCTA_CTX& c0 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 0u);
    P10DC_ORBITCTA_CTX& c1 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 1u);

    uint32_t lease_base_lane0 = 0u;
    uint32_t lease_pos_lane0 = 0u;
    if (threadIdx.x == 0) {
        const uint32_t nn0 = D_BKF_HIGH_NN_OFF[base_off];
        const uint32_t nn1 = D_BKF_HIGH_NN_OFF[base_off + nblocks];
        const uint32_t nr0 = D_BKF_HIGH_NRNL_OFF[base_off];
        const uint32_t nr1 = D_BKF_HIGH_NRNL_OFF[base_off + nblocks];
        c0.n0 = c1.n0 = nn1 - nn0;
        c0.n1 = c1.n1 = nr0;
        c0.total = c1.total = c0.n0 + (nr1 - nr0);
        c0.valid = c1.valid = 0;
        constexpr uint32_t BATCH = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
        lease_base_lane0 = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, BATCH);
        if (lease_base_lane0 < c0.total) {
            lease_pos_lane0 = 1u;
            p10dc_orbitcta_flat_dynamic_prepare_forward(
                c0, lease_base_lane0, base_off, nblocks, p);
        }
    }
    __syncthreads();

    const uint32_t total = c0.total;
    uint32_t cur = 0u;
    for (;;) {
        P10DC_ORBITCTA_CTX& current =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur);
        P10DC_ORBITCTA_CTX& next =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur ^ 1u);
        if (!current.valid) break;

        if (threadIdx.x == 0) {
            next.valid = 0;
            const uint32_t next_k = p10dc_orbitcta_flat_dynamic_pipe2_next_k(
                total, lease_base_lane0, lease_pos_lane0);
            if (next_k != 0xffffffffu)
                p10dc_orbitcta_flat_dynamic_prepare_forward(
                    next, next_k, base_off, nblocks, p);
        }

        // No barrier here: resident warps are free to enter the current column
        // loop while warp 0 executes the lane-0 next-orbit preparation branch.
        p10dc_orbitcta_flat_forward_columns(current);
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
    P10DC_ORBITCTA_CTX& c0 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 0u);
    P10DC_ORBITCTA_CTX& c1 = p10dc_orbitcta_flat_dynamic_pipe2_context(storage, 1u);

    uint32_t lease_base_lane0 = 0u;
    uint32_t lease_pos_lane0 = 0u;
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
        constexpr uint32_t BATCH = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
        lease_base_lane0 = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, BATCH);
        if (lease_base_lane0 < c0.total) {
            lease_pos_lane0 = 1u;
            p10dc_orbitcta_flat_dynamic_prepare_reverse(
                c0, lease_base_lane0, base_off, nblocks, p, edge);
        }
    }
    __syncthreads();

    const uint32_t total = c0.total;
    uint32_t cur = 0u;
    for (;;) {
        P10DC_ORBITCTA_CTX& current =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur);
        P10DC_ORBITCTA_CTX& next =
            p10dc_orbitcta_flat_dynamic_pipe2_context(storage, cur ^ 1u);
        if (!current.valid) break;

        if (threadIdx.x == 0) {
            next.valid = 0;
            const uint32_t next_k = p10dc_orbitcta_flat_dynamic_pipe2_next_k(
                total, lease_base_lane0, lease_pos_lane0);
            if (next_k != 0xffffffffu)
                p10dc_orbitcta_flat_dynamic_prepare_reverse(
                    next, next_k, base_off, nblocks, p, edge);
        }

        p10dc_orbitcta_flat_reverse_columns(current, edge);
        __syncthreads();
        cur ^= 1u;
    }
}
