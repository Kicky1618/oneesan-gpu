#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"

#ifndef P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_RANKFORMULA_PRECTX_FORWARD 0
#endif
static_assert(P10DC_RANKFORMULA_PRECTX_FORWARD == 0 ||
              P10DC_RANKFORMULA_PRECTX_FORWARD == 1,
              "P10DC_RANKFORMULA_PRECTX_FORWARD must be 0 or 1");

#if P10DC_RANKFORMULA_PRECTX_FORWARD
__constant__ P10DCDirectHighResolvedCtx* D_P10DC_PRECTX_FWD_NN;
__constant__ P10DCDirectHighResolvedCtx* D_P10DC_PRECTX_FWD_NRNL;

__device__ __forceinline__ P10DCDirectHighResolvedCtx
p10dc_make_forward_prectx(uint32_t bid, int p, BucketOrbitOp op, bool nn) {
    P10DCDirectHighResolvedCtx c{};
    const uint32_t sid = nn ? 0u : 3u;
    const uint32_t sl = bkf_orbit_src(op);
    const uint32_t jl = bkf_orbit_partner(op);
    const uint32_t dl = bkf_orbit_drop(op);
    const uint32_t ss = bkf_loc_owner(sl);
    const uint32_t js = bkf_loc_owner(jl);
    const uint32_t ds = bkf_loc_owner(dl);
    c.xb = bkf_high_main(ss, bid);
    if (!(c.xb.valid && c.xb.rows && c.xb.cols)) return c;
    uint32_t jbid = bid;
    if (p == LOW_LUT_K + 1) {
        const uint32_t center = nn ? uint32_t(R) : uint32_t(N);
        const int he = int(c.xb.hs) + (center == uint32_t(R) ? 1 : 0);
        jbid = uint32_t(3 * he + int(center));
    }
    c.jb = bkf_high_main(js, jbid);
    c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
    const uint32_t payload = p10dc_payload(op, false, true, sid, p, uint32_t(c.xb.hs));
    p10dc_prepare_forward_high_delta_direct_affine(
        c, payload, dl, p, ss, js, ds,
        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
    c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
    c.valid = 1;
    return c;
}

__global__ void p10dc_fill_forward_prectx_kernel(
    P10DCDirectHighResolvedCtx* nn_out,
    P10DCDirectHighResolvedCtx* nrnl_out
) {
    const uint32_t nb = D_BKF_MAIN_NBLOCKS;
    if (!nb) return;
    const uint32_t flat = uint32_t(blockIdx.x);
    const uint32_t pi = flat / nb;
    const uint32_t bid = flat - pi * nb;
    if (pi >= uint32_t(HIGH_LUT_K) || bid >= nb) return;
    const int p = (TARGET_W - 1) - int(pi);
    const uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);

    const uint32_t na = D_BKF_HIGH_NN_OFF[oi];
    const uint32_t nbeg = D_BKF_HIGH_NN_OFF[oi + 1u];
    for (uint32_t q = na + threadIdx.x; q < nbeg; q += blockDim.x)
        nn_out[q] = p10dc_make_forward_prectx(bid, p, D_BKF_HIGH_NN[q], true);

    const uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi];
    const uint32_t rb = D_BKF_HIGH_NRNL_OFF[oi + 1u];
    for (uint32_t q = ra + threadIdx.x; q < rb; q += blockDim.x)
        nrnl_out[q] = p10dc_make_forward_prectx(bid, p, D_BKF_HIGH_NRNL[q], false);
}

__device__ __forceinline__ P10DCDirectHighResolvedCtx
p10dc_load_forward_prectx(uint32_t qi, bool nn) {
    return nn ? D_P10DC_PRECTX_FWD_NN[qi] : D_P10DC_PRECTX_FWD_NRNL[qi];
}

template<class BaseTables>
struct BucketFusedPrecomputedForwardHighCtxTables : BaseTables {
    P10DCDirectHighResolvedCtx* prectx_fwd_nn = nullptr;
    P10DCDirectHighResolvedCtx* prectx_fwd_nrnl = nullptr;
    size_t prectx_fwd_nn_count = 0;
    size_t prectx_fwd_nrnl_count = 0;
    uint32_t prectx_main_blocks = 0;

    void install_metadata(
        const StorageLayout& layout,
        const BucketOrbitStreamsHost& o,
        const BucketFusedHost& f
    ) {
        BaseTables::install_metadata(layout, o, f);
        prectx_fwd_nn_count = o.high_nn.size();
        prectx_fwd_nrnl_count = o.high_nrnl.size();
        prectx_main_blocks = uint32_t(layout.main_blocks.size());
    }

    void bind_owner(
        uint32_t fixed,
        const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BaseTables::bind_owner(fixed, buckets, slot);
        if (prectx_fwd_nn_count) {
            ck(cudaMalloc(&prectx_fwd_nn,
                          prectx_fwd_nn_count * sizeof(P10DCDirectHighResolvedCtx)),
               "p10dc prectx forward NN alloc");
        }
        if (prectx_fwd_nrnl_count) {
            ck(cudaMalloc(&prectx_fwd_nrnl,
                          prectx_fwd_nrnl_count * sizeof(P10DCDirectHighResolvedCtx)),
               "p10dc prectx forward NRNL alloc");
        }
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_FWD_NN,
                              &prectx_fwd_nn, sizeof(prectx_fwd_nn)),
           "p10dc prectx forward NN ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_FWD_NRNL,
                              &prectx_fwd_nrnl, sizeof(prectx_fwd_nrnl)),
           "p10dc prectx forward NRNL ptr");

        const uint32_t contexts = uint32_t(HIGH_LUT_K) * prectx_main_blocks;
        if (contexts) {
            p10dc_fill_forward_prectx_kernel<<<contexts, 128>>>(
                prectx_fwd_nn, prectx_fwd_nrnl);
            ck(cudaGetLastError(), "p10dc prectx forward fill launch");
            ck(cudaDeviceSynchronize(), "p10dc prectx forward fill sync");
        }
        std::cerr << "p10dc_prectx_forward fixed_owner=" << fixed
                  << " nn_entries=" << prectx_fwd_nn_count
                  << " nrnl_entries=" << prectx_fwd_nrnl_count
                  << " context_bytes=" << sizeof(P10DCDirectHighResolvedCtx)
                  << " total_mib="
                  << double((prectx_fwd_nn_count + prectx_fwd_nrnl_count) *
                            sizeof(P10DCDirectHighResolvedCtx)) / double(1 << 20)
                  << " build_once=1 reuse_across_grid_x=1 reuse_across_moduli=1\n";
    }

    void release() {
        if (prectx_fwd_nrnl) cudaFree(prectx_fwd_nrnl);
        if (prectx_fwd_nn) cudaFree(prectx_fwd_nn);
        prectx_fwd_nrnl = nullptr;
        prectx_fwd_nn = nullptr;
        prectx_fwd_nrnl_count = 0;
        prectx_fwd_nn_count = 0;
        prectx_main_blocks = 0;
        BaseTables::release();
    }
};
#endif
