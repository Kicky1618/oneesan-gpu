#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#include "ramstream32_bucket_reverse_split54.cuh"

#ifndef P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_RANKFORMULA_PRECTX_FORWARD 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_REVERSE
#define P10DC_RANKFORMULA_PRECTX_REVERSE 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_COMPACT
#define P10DC_RANKFORMULA_PRECTX_COMPACT 0
#endif
static_assert(P10DC_RANKFORMULA_PRECTX_COMPACT == 1,
              "compact prectx header requires P10DC_RANKFORMULA_PRECTX_COMPACT=1");
static_assert(P10DC_RANKFORMULA_PRECTX_FORWARD || P10DC_RANKFORMULA_PRECTX_REVERSE,
              "compact prectx requires forward and/or reverse precomputation");
static_assert(BUCKET_LOCATOR_BITS == 18,
              "compact HIGH prectx assumes 18-bit owner-local bucket locators");
static_assert(P10DC_HIGH_ROW_AFFINE_BLOCKS <= 64,
              "compact HIGH prectx reserves 6 bits for main block id");

static constexpr uint32_t P10DC_HIGH_ROW_REF_BID_SHIFT = BUCKET_LOCATOR_BITS;
static constexpr uint32_t P10DC_HIGH_ROW_REF_LOC_MASK =
    (uint32_t(1) << BUCKET_LOCATOR_BITS) - 1u;
static constexpr uint32_t P10DC_HIGH_ROW_REF_INVALID = 0xffffffffu;

__device__ __forceinline__ uint32_t p10dc_high_row_ref_pack(
    uint32_t loc, uint32_t bid
) {
    if (loc > P10DC_HIGH_ROW_REF_LOC_MASK || bid >= 64u)
        return P10DC_HIGH_ROW_REF_INVALID;
    return loc | (bid << P10DC_HIGH_ROW_REF_BID_SHIFT);
}

__device__ __forceinline__ Count* p10dc_high_row_ref_resolve_unchecked(
    uint32_t ref, uint32_t hs
) {
    const uint32_t loc = ref & P10DC_HIGH_ROW_REF_LOC_MASK;
    const uint32_t bid = ref >> P10DC_HIGH_ROW_REF_BID_SHIFT;
    const uint32_t owner = bkf_loc_owner(loc);
    const uint32_t rank = bkf_loc_rank(loc);
    const P10DCHighRowAffine a =
        D_P10DC_HIGH_ROW_AFFINE[size_t(owner) * D_BKF_MAIN_NBLOCKS + bid];
    return a.base + Code(rank) * D_P10DC_HIGH_ROW_STRIDE[hs];
}

// Pointer-valued prectx stores BKCZ_MAX_LOCAL+1 CUDA pointers.  The compact
// form stores exact affine row coordinates instead; row pointers are restored
// once per orbit before the column loop.
struct P10DCHighClosureCompactPreCtx {
    uint32_t local_ref[BKCZ_MAX_LOCAL]{};
    uint32_t cross_ref = P10DC_HIGH_ROW_REF_INVALID;
    uint8_t local_n = 0;
    uint8_t cross_depth = 0;
    uint8_t fixed_hs = 0;
    uint8_t pad = 0;
};
static_assert(sizeof(P10DCHighClosureCompactPreCtx) ==
                  sizeof(uint32_t) * (BKCZ_MAX_LOCAL + 2u),
              "compact HIGH prectx layout changed unexpectedly");
static_assert(sizeof(P10DCHighClosureCompactPreCtx) < sizeof(P10DCDirectHighResolvedCtx),
              "compact HIGH prectx must be smaller than runtime context");

__device__ __forceinline__ bool p10dc_compact_add_high_ref(
    P10DCHighClosureCompactPreCtx& c,
    uint32_t key, uint32_t center, int fixed_hs
) {
    if (fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || center > uint32_t(::L)) return false;
    const int he = fixed_hs - bkci_delta(center);
    if (he < 0 || he > HIGH_LUT_K + 1) return false;
    const uint32_t z = D_BKF_HIGH_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return false;
    const uint32_t loc = bkf_direct_locator(z);
    const uint32_t owner = bkf_loc_owner(loc);
    const uint32_t bid = uint32_t(3 * he + int(center));
    Count* check = nullptr;
    if (!p10dc_high_row_affine_resolve(
            owner, bid, bkf_loc_rank(loc), uint32_t(fixed_hs), check))
        return false;
    const uint32_t n = uint32_t(c.local_n);
    if (n >= BKCZ_MAX_LOCAL) return false;
    const uint32_t ref = p10dc_high_row_ref_pack(loc, bid);
    if (ref == P10DC_HIGH_ROW_REF_INVALID) return false;
    c.local_ref[n] = ref;
    c.local_n = uint8_t(n + 1u);
    return true;
}

__device__ __forceinline__ void p10dc_compact_set_high_cross_ref(
    P10DCHighClosureCompactPreCtx& c,
    uint32_t key, uint32_t center, int fixed_hs, uint32_t depth
) {
    if (!depth || fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || center > uint32_t(::L)) return;
    const int he = fixed_hs - bkci_delta(center);
    if (he < 0 || he > HIGH_LUT_K + 1) return;
    const uint32_t z = D_BKF_HIGH_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return;
    const uint32_t loc = bkf_direct_locator(z);
    const uint32_t owner = bkf_loc_owner(loc);
    const uint32_t bid = uint32_t(3 * he + int(center));
    Count* check = nullptr;
    if (!p10dc_high_row_affine_resolve(
            owner, bid, bkf_loc_rank(loc), uint32_t(fixed_hs), check))
        return;
    const uint32_t ref = p10dc_high_row_ref_pack(loc, bid);
    if (ref == P10DC_HIGH_ROW_REF_INVALID) return;
    c.cross_ref = ref;
    c.cross_depth = uint8_t(depth);
}

__device__ __forceinline__ P10DCHighClosureCompactPreCtx
p10dc_build_high_compact_prectx(
    MateID d, int fixed_hs, int p, uint32_t payload
) {
    P10DCHighClosureCompactPreCtx c{};
    c.fixed_hs = uint8_t(fixed_hs);
    if (!p10dc_payload_valid(payload) || mpair(d, p) != NN) return c;
    constexpr uint64_t MASK = (uint64_t(1) << (2 * HIGH_LUT_K)) - 1ull;
    const uint32_t factor = uint32_t((d >> 2) & MASK);
    const uint32_t base = bkcz_ternary_key<HIGH_LUT_K>(factor);
    const uint32_t dest_center = uint32_t(mget(d, 0));

    {
        const int32_t delta = p10dc_high_pair_delta(RL, p);
        const uint32_t center = p10dc_high_center_after_pair(dest_center, RL, p);
        p10dc_compact_add_high_ref(
            c, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }

    uint16_t lm = p10dc_payload_lm(payload), rm = p10dc_payload_rm(payload);
    while (lm) {
        const int i = __ffs(int(lm)) - 1;
        lm = uint16_t(lm & (lm - 1));
        const int q = p - 2 - i;
        int32_t delta = p10dc_high_pair_delta(LL, p);
        uint32_t center = p10dc_high_center_after_pair(dest_center, LL, p);
        if (q > 0) delta -= int32_t(p10dc_pow3(uint32_t(q - 1)));
        else center = uint32_t(R);
        p10dc_compact_add_high_ref(
            c, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }
    while (rm) {
        const int i = __ffs(int(rm)) - 1;
        rm = uint16_t(rm & (rm - 1));
        const int q = p + 1 + i;
        const int32_t delta = p10dc_high_pair_delta(RR, p) +
                              int32_t(p10dc_pow3(uint32_t(q - 1)));
        const uint32_t center = p10dc_high_center_after_pair(dest_center, RR, p);
        p10dc_compact_add_high_ref(
            c, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }

    const uint32_t depth = uint32_t(p10dc_payload_depth(payload));
    if (depth) {
        const int32_t delta = p10dc_high_pair_delta(LL, p);
        const uint32_t center = p10dc_high_center_after_pair(dest_center, LL, p);
        p10dc_compact_set_high_cross_ref(
            c, uint32_t(int32_t(base) + delta), center, fixed_hs + 2, depth);
    }
    return c;
}

__device__ __forceinline__ P10DCHighClosureCompactPreCtx
p10dc_make_forward_compact_prectx(uint32_t bid, int p, BucketOrbitOp op, bool nn) {
    const uint32_t sid = nn ? 0u : 3u;
    const uint32_t sl = bkf_orbit_src(op);
    const uint32_t dl = bkf_orbit_drop(op);
    const uint32_t ss = bkf_loc_owner(sl);
    const BucketPhysicalBlock xb = bkf_high_main(ss, bid);
    if (!(xb.valid && xb.rows && xb.cols)) return {};
    const uint32_t ds = bkf_loc_owner(dl);
    const BucketPhysicalBlock db = bkf_high_block(ds, uint32_t(xb.hs));
    const uint32_t payload = p10dc_payload(op, false, true, sid, p, uint32_t(xb.hs));
    const uint32_t dc = bkci_high_code(dl, db.he);
    const int rel = p - LOW_LUT_K;
    const MateID d = minsert(MateID(dc), rel, N);
    return p10dc_build_high_compact_prectx(d, db.hs, rel, payload);
}

__device__ __forceinline__ P10DCHighClosureCompactPreCtx
p10dc_make_reverse_compact_prectx(
    uint32_t bid, int p, BucketOrbitOp op, uint32_t kind, uint32_t sid
) {
    const bool edge = p == TARGET_W - 1;
    const uint32_t sl = bkf_orbit_src(op);
    const uint32_t dl = bkf_orbit_drop(op);
    const uint32_t ss = bkf_loc_owner(sl);
    const BucketPhysicalBlock xb = bkf_high_main(ss, bid);
    if (!(xb.valid && xb.rows && xb.cols)) return {};
    const uint32_t ds = bkf_loc_owner(dl);
    const BucketPhysicalBlock db = bkf_high_block(ds, uint32_t(xb.hs));
    const BucketPhysicalBlock plan_db = edge ? xb : db;
    const uint32_t loc = edge ? sl : dl;
    const uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(xb.hs));
    const uint32_t dc = bkci_high_code(loc, plan_db.he);
    const int rel = p - LOW_LUT_K;
    const MateID d = edge
        ? (MateID(plan_db.c) | (MateID(dc) << 2))
        : blocked_exclude_reverse(MateID(dc), HIGH_LUT_K + 1, rel);
    (void)kind;
    return p10dc_build_high_compact_prectx(d, plan_db.hs, rel, payload);
}

#if P10DC_RANKFORMULA_PRECTX_FORWARD
__constant__ P10DCHighClosureCompactPreCtx* D_P10DC_COMPACT_PRECTX_FWD_NN;
__constant__ P10DCHighClosureCompactPreCtx* D_P10DC_COMPACT_PRECTX_FWD_NRNL;

__global__ void p10dc_fill_forward_compact_prectx_kernel(
    P10DCHighClosureCompactPreCtx* nn_out,
    P10DCHighClosureCompactPreCtx* nrnl_out
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
        nn_out[q] = p10dc_make_forward_compact_prectx(bid, p, D_BKF_HIGH_NN[q], true);
    const uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi];
    const uint32_t rb = D_BKF_HIGH_NRNL_OFF[oi + 1u];
    for (uint32_t q = ra + threadIdx.x; q < rb; q += blockDim.x)
        nrnl_out[q] = p10dc_make_forward_compact_prectx(bid, p, D_BKF_HIGH_NRNL[q], false);
}

__device__ __forceinline__ void p10dc_apply_forward_prectx(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, bool nn
) {
    const P10DCHighClosureCompactPreCtx z =
        nn ? D_P10DC_COMPACT_PRECTX_FWD_NN[qi] : D_P10DC_COMPACT_PRECTX_FWD_NRNL[qi];
    c.local_n = z.local_n;
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i)
        if (i < uint32_t(z.local_n))
            c.local_base[i] = p10dc_high_row_ref_resolve_unchecked(z.local_ref[i], z.fixed_hs);
    c.cross_depth = uint32_t(z.cross_depth);
    c.cross_base = z.cross_depth
        ? p10dc_high_row_ref_resolve_unchecked(z.cross_ref, uint32_t(z.fixed_hs) + 2u)
        : nullptr;
}
#endif

#if P10DC_RANKFORMULA_PRECTX_REVERSE
__constant__ P10DCHighClosureCompactPreCtx* D_P10DC_COMPACT_PRECTX_REV_NN;
__constant__ P10DCHighClosureCompactPreCtx* D_P10DC_COMPACT_PRECTX_REV_NR;
__constant__ P10DCHighClosureCompactPreCtx* D_P10DC_COMPACT_PRECTX_REV_NL;

__global__ void p10dc_fill_reverse_compact_prectx_kernel(
    P10DCHighClosureCompactPreCtx* nn_out,
    P10DCHighClosureCompactPreCtx* nr_out,
    P10DCHighClosureCompactPreCtx* nl_out
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
        nn_out[q] = p10dc_make_reverse_compact_prectx(
            bid, p, D_RS54_HIGH_NN[q], CPU_ORBIT_NN, 0u);
    const uint32_t ra = D_RS54_HIGH_NR_OFF[oi];
    const uint32_t rb = D_RS54_HIGH_NR_OFF[oi + 1u];
    for (uint32_t q = ra + threadIdx.x; q < rb; q += blockDim.x)
        nr_out[q] = p10dc_make_reverse_compact_prectx(
            bid, p, D_RS54_HIGH_NR[q], CPU_ORBIT_NR, 1u);
    const uint32_t la = D_RS54_HIGH_NL_OFF[oi];
    const uint32_t lb = D_RS54_HIGH_NL_OFF[oi + 1u];
    for (uint32_t q = la + threadIdx.x; q < lb; q += blockDim.x)
        nl_out[q] = p10dc_make_reverse_compact_prectx(
            bid, p, D_RS54_HIGH_NL[q], CPU_ORBIT_NL, 2u);
}

__device__ __forceinline__ void p10dc_apply_reverse_prectx(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, uint32_t kind
) {
    P10DCHighClosureCompactPreCtx z{};
    if (kind == CPU_ORBIT_NN) z = D_P10DC_COMPACT_PRECTX_REV_NN[qi];
    else if (kind == CPU_ORBIT_NR) z = D_P10DC_COMPACT_PRECTX_REV_NR[qi];
    else z = D_P10DC_COMPACT_PRECTX_REV_NL[qi];
    c.local_n = z.local_n;
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i)
        if (i < uint32_t(z.local_n))
            c.local_base[i] = p10dc_high_row_ref_resolve_unchecked(z.local_ref[i], z.fixed_hs);
    c.cross_depth = uint32_t(z.cross_depth);
    c.cross_base = z.cross_depth
        ? p10dc_high_row_ref_resolve_unchecked(z.cross_ref, uint32_t(z.fixed_hs) + 2u)
        : nullptr;
}
#endif

template<class BaseTables>
struct BucketFusedCompactPrecomputedHighCtxTables : BaseTables {
#if P10DC_RANKFORMULA_PRECTX_FORWARD
    P10DCHighClosureCompactPreCtx* prectx_fwd_nn = nullptr;
    P10DCHighClosureCompactPreCtx* prectx_fwd_nrnl = nullptr;
    size_t prectx_fwd_nn_count = 0;
    size_t prectx_fwd_nrnl_count = 0;
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
    P10DCHighClosureCompactPreCtx* prectx_rev_nn = nullptr;
    P10DCHighClosureCompactPreCtx* prectx_rev_nr = nullptr;
    P10DCHighClosureCompactPreCtx* prectx_rev_nl = nullptr;
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
           "p10dc compact reverse prectx total");
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
            ck(cudaMalloc(&prectx_fwd_nn,
                          prectx_fwd_nn_count * sizeof(P10DCHighClosureCompactPreCtx)),
               "p10dc compact prectx forward NN alloc");
        if (prectx_fwd_nrnl_count)
            ck(cudaMalloc(&prectx_fwd_nrnl,
                          prectx_fwd_nrnl_count * sizeof(P10DCHighClosureCompactPreCtx)),
               "p10dc compact prectx forward NRNL alloc");
        ck(cudaMemcpyToSymbol(D_P10DC_COMPACT_PRECTX_FWD_NN,
                              &prectx_fwd_nn, sizeof(prectx_fwd_nn)),
           "p10dc compact prectx forward NN ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_COMPACT_PRECTX_FWD_NRNL,
                              &prectx_fwd_nrnl, sizeof(prectx_fwd_nrnl)),
           "p10dc compact prectx forward NRNL ptr");
        const uint32_t fctx = uint32_t(HIGH_LUT_K) * prectx_main_blocks;
        if (fctx) {
            p10dc_fill_forward_compact_prectx_kernel<<<fctx, 128>>>(
                prectx_fwd_nn, prectx_fwd_nrnl);
            ck(cudaGetLastError(), "p10dc compact prectx forward fill launch");
            ck(cudaDeviceSynchronize(), "p10dc compact prectx forward fill sync");
        }
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
        uint32_t pitch = 0;
        uint32_t *nn_off = nullptr, *nr_off = nullptr, *nl_off = nullptr;
        ck(cudaMemcpyFromSymbol(&pitch, D_RS54_PITCH, sizeof(pitch)),
           "p10dc compact reverse prectx pitch");
        ck(cudaMemcpyFromSymbol(&nn_off, D_RS54_HIGH_NN_OFF, sizeof(nn_off)),
           "p10dc compact reverse prectx NN off ptr");
        ck(cudaMemcpyFromSymbol(&nr_off, D_RS54_HIGH_NR_OFF, sizeof(nr_off)),
           "p10dc compact reverse prectx NR off ptr");
        ck(cudaMemcpyFromSymbol(&nl_off, D_RS54_HIGH_NL_OFF, sizeof(nl_off)),
           "p10dc compact reverse prectx NL off ptr");
        prectx_rev_nn_count = reverse_total_from_offsets(nn_off, pitch);
        prectx_rev_nr_count = reverse_total_from_offsets(nr_off, pitch);
        prectx_rev_nl_count = reverse_total_from_offsets(nl_off, pitch);
        if (prectx_rev_nn_count)
            ck(cudaMalloc(&prectx_rev_nn,
                          prectx_rev_nn_count * sizeof(P10DCHighClosureCompactPreCtx)),
               "p10dc compact prectx reverse NN alloc");
        if (prectx_rev_nr_count)
            ck(cudaMalloc(&prectx_rev_nr,
                          prectx_rev_nr_count * sizeof(P10DCHighClosureCompactPreCtx)),
               "p10dc compact prectx reverse NR alloc");
        if (prectx_rev_nl_count)
            ck(cudaMalloc(&prectx_rev_nl,
                          prectx_rev_nl_count * sizeof(P10DCHighClosureCompactPreCtx)),
               "p10dc compact prectx reverse NL alloc");
        ck(cudaMemcpyToSymbol(D_P10DC_COMPACT_PRECTX_REV_NN,
                              &prectx_rev_nn, sizeof(prectx_rev_nn)),
           "p10dc compact prectx reverse NN ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_COMPACT_PRECTX_REV_NR,
                              &prectx_rev_nr, sizeof(prectx_rev_nr)),
           "p10dc compact prectx reverse NR ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_COMPACT_PRECTX_REV_NL,
                              &prectx_rev_nl, sizeof(prectx_rev_nl)),
           "p10dc compact prectx reverse NL ptr");
        const uint32_t rctx = uint32_t(HIGH_LUT_K) * prectx_main_blocks;
        if (rctx) {
            p10dc_fill_reverse_compact_prectx_kernel<<<rctx, 128>>>(
                prectx_rev_nn, prectx_rev_nr, prectx_rev_nl);
            ck(cudaGetLastError(), "p10dc compact prectx reverse fill launch");
            ck(cudaDeviceSynchronize(), "p10dc compact prectx reverse fill sync");
        }
#endif
        std::cerr << "p10dc_compact_prectx_high fixed_owner=" << fixed
                  << " closure_context_bytes=" << sizeof(P10DCHighClosureCompactPreCtx)
                  << " pointer_context_bytes="
                  << (sizeof(uint64_t) * (BKCZ_MAX_LOCAL + 1u) + sizeof(uint32_t) + 4u)
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
                  << " row_ref_bits=24 row_ref_storage_bytes=4"
                  << " restore_once_per_orbit=1 column_loop_extra_ops=0\n";
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
