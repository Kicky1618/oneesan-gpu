#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#include "ramstream32_bucket_reverse_split54.cuh"

#ifndef P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_RANKFORMULA_PRECTX_FORWARD 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_REVERSE
#define P10DC_RANKFORMULA_PRECTX_REVERSE 0
#endif
static_assert(P10DC_RANKFORMULA_PRECTX_FORWARD == 0 ||
              P10DC_RANKFORMULA_PRECTX_FORWARD == 1,
              "P10DC_RANKFORMULA_PRECTX_FORWARD must be 0 or 1");
static_assert(P10DC_RANKFORMULA_PRECTX_REVERSE == 0 ||
              P10DC_RANKFORMULA_PRECTX_REVERSE == 1,
              "P10DC_RANKFORMULA_PRECTX_REVERSE must be 0 or 1");

#if P10DC_RANKFORMULA_PRECTX_FORWARD || P10DC_RANKFORMULA_PRECTX_REVERSE
struct P10DCHighClosurePreCtx {
    Count* local_base[BKCZ_MAX_LOCAL]{};
    Count* cross_base = nullptr;
    uint32_t cross_depth = 0;
    uint8_t local_n = 0;
    uint8_t pad[3]{};
};
static_assert(sizeof(P10DCHighClosurePreCtx) < sizeof(P10DCDirectHighResolvedCtx),
              "HIGH closure prectx must stay smaller than the runtime context");

__device__ __forceinline__ P10DCHighClosurePreCtx
p10dc_pack_high_closure_prectx(const P10DCDirectHighResolvedCtx& c) {
    P10DCHighClosurePreCtx z{};
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) z.local_base[i] = c.local_base[i];
    z.cross_base = c.cross_base;
    z.cross_depth = c.cross_depth;
    z.local_n = c.local_n;
    return z;
}
#endif

#if P10DC_RANKFORMULA_PRECTX_FORWARD
__constant__ P10DCHighClosurePreCtx* D_P10DC_PRECTX_FWD_NN;
__constant__ P10DCHighClosurePreCtx* D_P10DC_PRECTX_FWD_NRNL;

__device__ __forceinline__ P10DCHighClosurePreCtx
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
    if (!(c.xb.valid && c.xb.rows && c.xb.cols)) return {};
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
    return p10dc_pack_high_closure_prectx(c);
}

__global__ void p10dc_fill_forward_prectx_kernel(
    P10DCHighClosurePreCtx* nn_out,
    P10DCHighClosurePreCtx* nrnl_out
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

__device__ __forceinline__ void p10dc_apply_forward_prectx(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, bool nn
) {
    const P10DCHighClosurePreCtx z =
        nn ? D_P10DC_PRECTX_FWD_NN[qi] : D_P10DC_PRECTX_FWD_NRNL[qi];
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) c.local_base[i] = z.local_base[i];
    c.cross_base = z.cross_base;
    c.cross_depth = z.cross_depth;
    c.local_n = z.local_n;
}
#endif

#if P10DC_RANKFORMULA_PRECTX_REVERSE
__constant__ P10DCHighClosurePreCtx* D_P10DC_PRECTX_REV_NN;
__constant__ P10DCHighClosurePreCtx* D_P10DC_PRECTX_REV_NR;
__constant__ P10DCHighClosurePreCtx* D_P10DC_PRECTX_REV_NL;

__device__ __forceinline__ P10DCHighClosurePreCtx
p10dc_make_reverse_prectx(uint32_t bid, int p, BucketOrbitOp op, uint32_t kind, uint32_t sid) {
    P10DCDirectHighResolvedCtx c{};
    const bool edge = p == TARGET_W - 1;
    const uint32_t sl = bkf_orbit_src(op);
    const uint32_t jl = bkf_orbit_partner(op);
    const uint32_t dl = bkf_orbit_drop(op);
    const uint32_t ss = bkf_loc_owner(sl);
    const uint32_t js = bkf_loc_owner(jl);
    const uint32_t ds = bkf_loc_owner(dl);
    c.xb = bkf_high_main(ss, bid);
    if (!(c.xb.valid && c.xb.rows && c.xb.cols)) return {};
    c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
    c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
    const uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(c.xb.hs));
    p10dc_prepare_reverse_high_delta_direct_affine(
        c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
        ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
    return p10dc_pack_high_closure_prectx(c);
}

__global__ void p10dc_fill_reverse_prectx_kernel(
    P10DCHighClosurePreCtx* nn_out,
    P10DCHighClosurePreCtx* nr_out,
    P10DCHighClosurePreCtx* nl_out
) {
    const uint32_t nb = D_BKF_MAIN_NBLOCKS;
    if (!nb) return;
    const uint32_t flat = uint32_t(blockIdx.x);
    const uint32_t pi = flat / nb;
    const uint32_t bid = flat - pi * nb;
    if (pi >= uint32_t(HIGH_LUT_K) || bid >= nb) return;
    const int p = (LOW_LUT_K + 1) + int(pi);
    const uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);

    const uint32_t na = D_RS54_HIGH_NN_OFF[oi];
    const uint32_t nbeg = D_RS54_HIGH_NN_OFF[oi + 1u];
    for (uint32_t q = na + threadIdx.x; q < nbeg; q += blockDim.x)
        nn_out[q] = p10dc_make_reverse_prectx(bid, p, D_RS54_HIGH_NN[q], CPU_ORBIT_NN, 0u);

    const uint32_t ra = D_RS54_HIGH_NR_OFF[oi];
    const uint32_t rb = D_RS54_HIGH_NR_OFF[oi + 1u];
    for (uint32_t q = ra + threadIdx.x; q < rb; q += blockDim.x)
        nr_out[q] = p10dc_make_reverse_prectx(bid, p, D_RS54_HIGH_NR[q], CPU_ORBIT_NR, 1u);

    const uint32_t la = D_RS54_HIGH_NL_OFF[oi];
    const uint32_t lb = D_RS54_HIGH_NL_OFF[oi + 1u];
    for (uint32_t q = la + threadIdx.x; q < lb; q += blockDim.x)
        nl_out[q] = p10dc_make_reverse_prectx(bid, p, D_RS54_HIGH_NL[q], CPU_ORBIT_NL, 2u);
}

__device__ __forceinline__ void p10dc_apply_reverse_prectx(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, uint32_t kind
) {
    P10DCHighClosurePreCtx z{};
    if (kind == CPU_ORBIT_NN) z = D_P10DC_PRECTX_REV_NN[qi];
    else if (kind == CPU_ORBIT_NR) z = D_P10DC_PRECTX_REV_NR[qi];
    else z = D_P10DC_PRECTX_REV_NL[qi];
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) c.local_base[i] = z.local_base[i];
    c.cross_base = z.cross_base;
    c.cross_depth = z.cross_depth;
    c.local_n = z.local_n;
}
#endif

template<class BaseTables>
struct BucketFusedPrecomputedHighCtxTables : BaseTables {
#if P10DC_RANKFORMULA_PRECTX_FORWARD
    P10DCHighClosurePreCtx* prectx_fwd_nn = nullptr;
    P10DCHighClosurePreCtx* prectx_fwd_nrnl = nullptr;
    size_t prectx_fwd_nn_count = 0;
    size_t prectx_fwd_nrnl_count = 0;
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
    P10DCHighClosurePreCtx* prectx_rev_nn = nullptr;
    P10DCHighClosurePreCtx* prectx_rev_nr = nullptr;
    P10DCHighClosurePreCtx* prectx_rev_nl = nullptr;
    size_t prectx_rev_nn_count = 0;
    size_t prectx_rev_nr_count = 0;
    size_t prectx_rev_nl_count = 0;
#endif
    uint32_t prectx_main_blocks = 0;

    void install_metadata(
        const StorageLayout& layout,
        const BucketOrbitStreamsHost& o,
        const BucketFusedHost& f
    ) {
        BaseTables::install_metadata(layout, o, f);
#if P10DC_RANKFORMULA_PRECTX_FORWARD
        prectx_fwd_nn_count = o.high_nn.size();
        prectx_fwd_nrnl_count = o.high_nrnl.size();
#endif
        prectx_main_blocks = uint32_t(layout.main_blocks.size());
    }

#if P10DC_RANKFORMULA_PRECTX_REVERSE
    static size_t reverse_total_from_offsets(uint32_t* off, uint32_t pitch) {
        if (!off || !pitch || !HIGH_LUT_K) return 0;
        uint32_t total = 0;
        const size_t last = size_t(HIGH_LUT_K) * pitch - 1u;
        ck(cudaMemcpy(&total, off + last, sizeof(total), cudaMemcpyDeviceToHost),
           "p10dc reverse prectx total");
        return size_t(total);
    }
#endif

    void bind_owner(
        uint32_t fixed,
        const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BaseTables::bind_owner(fixed, buckets, slot);
#if P10DC_RANKFORMULA_PRECTX_FORWARD
        if (prectx_fwd_nn_count)
            ck(cudaMalloc(&prectx_fwd_nn, prectx_fwd_nn_count * sizeof(P10DCHighClosurePreCtx)),
               "p10dc prectx forward NN alloc");
        if (prectx_fwd_nrnl_count)
            ck(cudaMalloc(&prectx_fwd_nrnl, prectx_fwd_nrnl_count * sizeof(P10DCHighClosurePreCtx)),
               "p10dc prectx forward NRNL alloc");
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_FWD_NN, &prectx_fwd_nn, sizeof(prectx_fwd_nn)),
           "p10dc prectx forward NN ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_FWD_NRNL, &prectx_fwd_nrnl, sizeof(prectx_fwd_nrnl)),
           "p10dc prectx forward NRNL ptr");
        const uint32_t fctx = uint32_t(HIGH_LUT_K) * prectx_main_blocks;
        if (fctx) {
            p10dc_fill_forward_prectx_kernel<<<fctx, 128>>>(prectx_fwd_nn, prectx_fwd_nrnl);
            ck(cudaGetLastError(), "p10dc prectx forward fill launch");
            ck(cudaDeviceSynchronize(), "p10dc prectx forward fill sync");
        }
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
        uint32_t pitch = 0;
        uint32_t *nn_off = nullptr, *nr_off = nullptr, *nl_off = nullptr;
        ck(cudaMemcpyFromSymbol(&pitch, D_RS54_PITCH, sizeof(pitch)), "p10dc reverse prectx pitch");
        ck(cudaMemcpyFromSymbol(&nn_off, D_RS54_HIGH_NN_OFF, sizeof(nn_off)), "p10dc reverse prectx NN off ptr");
        ck(cudaMemcpyFromSymbol(&nr_off, D_RS54_HIGH_NR_OFF, sizeof(nr_off)), "p10dc reverse prectx NR off ptr");
        ck(cudaMemcpyFromSymbol(&nl_off, D_RS54_HIGH_NL_OFF, sizeof(nl_off)), "p10dc reverse prectx NL off ptr");
        prectx_rev_nn_count = reverse_total_from_offsets(nn_off, pitch);
        prectx_rev_nr_count = reverse_total_from_offsets(nr_off, pitch);
        prectx_rev_nl_count = reverse_total_from_offsets(nl_off, pitch);
        if (prectx_rev_nn_count)
            ck(cudaMalloc(&prectx_rev_nn, prectx_rev_nn_count * sizeof(P10DCHighClosurePreCtx)), "p10dc prectx reverse NN alloc");
        if (prectx_rev_nr_count)
            ck(cudaMalloc(&prectx_rev_nr, prectx_rev_nr_count * sizeof(P10DCHighClosurePreCtx)), "p10dc prectx reverse NR alloc");
        if (prectx_rev_nl_count)
            ck(cudaMalloc(&prectx_rev_nl, prectx_rev_nl_count * sizeof(P10DCHighClosurePreCtx)), "p10dc prectx reverse NL alloc");
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_REV_NN, &prectx_rev_nn, sizeof(prectx_rev_nn)), "p10dc prectx reverse NN ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_REV_NR, &prectx_rev_nr, sizeof(prectx_rev_nr)), "p10dc prectx reverse NR ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_PRECTX_REV_NL, &prectx_rev_nl, sizeof(prectx_rev_nl)), "p10dc prectx reverse NL ptr");
        const uint32_t rctx = uint32_t(HIGH_LUT_K) * prectx_main_blocks;
        if (rctx) {
            p10dc_fill_reverse_prectx_kernel<<<rctx, 128>>>(prectx_rev_nn, prectx_rev_nr, prectx_rev_nl);
            ck(cudaGetLastError(), "p10dc prectx reverse fill launch");
            ck(cudaDeviceSynchronize(), "p10dc prectx reverse fill sync");
        }
#endif
        std::cerr << "p10dc_prectx_high fixed_owner=" << fixed
                  << " closure_context_bytes=" << sizeof(P10DCHighClosurePreCtx)
                  << " runtime_context_bytes=" << sizeof(P10DCDirectHighResolvedCtx)
#if P10DC_RANKFORMULA_PRECTX_FORWARD
                  << " fwd_nn=" << prectx_fwd_nn_count
                  << " fwd_nrnl=" << prectx_fwd_nrnl_count
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
                  << " rev_nn=" << prectx_rev_nn_count
                  << " rev_nr=" << prectx_rev_nr_count
                  << " rev_nl=" << prectx_rev_nl_count
#endif
                  << " forward=" << P10DC_RANKFORMULA_PRECTX_FORWARD
                  << " reverse=" << P10DC_RANKFORMULA_PRECTX_REVERSE
                  << " build_once=1 reuse_across_grid_x=1 reuse_across_moduli=1"
                  << " scheduler_metadata_preserved=1 io_rows_runtime_resolve=1\n";
    }

    void release() {
#if P10DC_RANKFORMULA_PRECTX_REVERSE
        if (prectx_rev_nl) cudaFree(prectx_rev_nl);
        if (prectx_rev_nr) cudaFree(prectx_rev_nr);
        if (prectx_rev_nn) cudaFree(prectx_rev_nn);
        prectx_rev_nl = prectx_rev_nr = prectx_rev_nn = nullptr;
        prectx_rev_nl_count = prectx_rev_nr_count = prectx_rev_nn_count = 0;
#endif
#if P10DC_RANKFORMULA_PRECTX_FORWARD
        if (prectx_fwd_nrnl) cudaFree(prectx_fwd_nrnl);
        if (prectx_fwd_nn) cudaFree(prectx_fwd_nn);
        prectx_fwd_nrnl = prectx_fwd_nn = nullptr;
        prectx_fwd_nrnl_count = prectx_fwd_nn_count = 0;
#endif
        prectx_main_blocks = 0;
        BaseTables::release();
    }
};

// Backward-compatible name used by early forward-only wiring.
template<class BaseTables>
using BucketFusedPrecomputedForwardHighCtxTables = BucketFusedPrecomputedHighCtxTables<BaseTables>;
