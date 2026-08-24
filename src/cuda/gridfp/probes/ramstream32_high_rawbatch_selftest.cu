#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_rawbatch.cuh"

static uint32_t host_high_flip_low(uint32_t lc, uint32_t depth) {
    int s = int(depth);
    for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
        MateValue v = MateValue((lc >> (2 * pos)) & 3u);
        if (v == ::L) ++s;
        else if (v == R && --s == 0) {
            uint32_t z = 3u << (2 * pos);
            return (lc & ~z) | (uint32_t(::L) << (2 * pos));
        }
    }
    return 0xffffffffu;
}

static inline uint32_t hd_kind(uint32_t x) { return x >> HIGHDESC_KIND_SHIFT; }
static inline uint32_t hd_block(uint32_t x) {
    return (x >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK;
}
static inline uint32_t hd_rank(uint32_t x) { return x & HIGHDESC_RANK_MASK; }
static inline uint32_t hd_depth(uint32_t x) {
    return (x >> HIGHDESC_DEPTH_SHIFT) & HIGHDESC_DEPTH_MASK;
}

static MateID unrank_main_physical(
    uint32_t mask, uint32_t bid, uint32_t elem,
    const StorageFactorHost& storage, const StorageLayout& logical
) {
    constexpr int L = LOW_LUT_K;
    constexpr int S = StorageFactorHost::S;
    const StorageBlock& b = logical.main_blocks[bid];
    uint32_t w = lowmask_major_width(mask, b.hs);
    if (!w) std::exit(220);
    uint32_t hr = elem / w;
    uint32_t lr = elem - hr * w;
    if (hr >= b.rows) std::exit(221);
    uint32_t hc = storage.high_all_codes[storage.high_all_off[b.he] + hr];
    uint32_t la = G_FACTOR.low_mask_off[size_t(mask) * S + b.hs];
    uint32_t lc = G_FACTOR.low_mask_codes[la + lr];
    return MateID(lc) | (MateID(b.c) << (2 * L))
         | (MateID(hc) << (2 * (L + 1)));
}

int main() {
    static_assert(TARGET_W <= 12, "rawbatch selftest is reduced-width only");
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    LowMaskMajorLayout mm = build_lowmask_major_layout(storage, logical);
    HighDescHost desc = build_high_descriptors(storage, logical);
    HighOrbitHost orbit = build_high_orbit(storage, logical);
    WindowPlan wp = make_direct2d_window(true);

    // One huge host-only batch is enough to validate task coverage and all
    // absolute physical target formulas at reduced width.
    HighRawBatchTasks bt = build_high_raw_tasks(mm, logical, 0, mm.masks);
    std::vector<uint8_t> seen(size_t(mm.main_size), 0);
    uint64_t owner_checks = 0, closure_checks = 0;

    for (const HighRawTask& t : bt.tasks) {
        uint32_t mask = t.mask;
        uint32_t bid = t.bid;
        const StorageBlock& sb = logical.main_blocks[bid];
        uint32_t sw = lowmask_major_width(mask, sb.hs);
        if (!sw) return 222;
        Code src_abs_base = lowmask_major_main_block_base(mm, mask, bid);

        for (uint32_t k = 0; k < t.count; ++k) {
            uint32_t elem = t.elem0 + k;
            Code src_abs = src_abs_base + elem;
            if (src_abs >= mm.main_size || seen[size_t(src_abs)]) {
                std::cerr << "raw task overlap/oob mask=" << mask << " bid=" << bid
                          << " elem=" << elem << '\n';
                return 223;
            }
            seen[size_t(src_abs)] = 1;

            uint32_t hr = elem / sw;
            uint32_t lr = elem - hr * sw;
            MateID m = unrank_main_physical(mask, bid, elem, storage, logical);
            if (lowmask_major_rank_main_host(m, storage, logical, mm) != src_abs) {
                std::cerr << "raw source rank mismatch\n";
                return 224;
            }

            for (int p = wp.p_hi; p >= wp.p_lo; --p) {
                uint32_t pi = uint32_t((TARGET_W - 1) - p);
                uint64_t ow = orbit.rec[size_t(pi) * orbit.main_total
                                        + orbit.main_base[bid] + hr];
                uint32_t ok = high_orbit_kind(ow);
                if (ok >= HIGH_ORBIT_NN && ok <= HIGH_ORBIT_NL) {
                    uint32_t jbid = high_orbit_jblock(ow);
                    uint32_t dbid = high_orbit_dblock(ow);
                    uint32_t jw = lowmask_major_width(mask, logical.main_blocks[jbid].hs);
                    uint32_t dw = lowmask_major_width(mask, logical.block_blocks[dbid].hs);
                    Code jabs = lowmask_major_main_block_base(mm, mask, jbid)
                              + Code(high_orbit_jhr(ow)) * jw + lr;
                    Code dabs = lowmask_major_block_block_base(mm, mask, dbid)
                              + Code(high_orbit_dhr(ow)) * dw + lr;

                    MateValue b = mget(m, p - 1);
                    MateValuePair pair = ok == HIGH_ORBIT_NN ? LR
                                       : ok == HIGH_ORBIT_NR ? RN : LN;
                    if (mget(m, p) != N
                        || (ok == HIGH_ORBIT_NN && b != N)
                        || (ok == HIGH_ORBIT_NR && b != R)
                        || (ok == HIGH_ORBIT_NL && b != ::L)) return 225;
                    MateID jm = msetpair(m, p, pair);
                    MateID dm = mshrink(m, p);
                    Code jref = lowmask_major_rank_main_host(jm, storage, logical, mm);
                    Code dref = lowmask_major_rank_block_host(dm, storage, logical, mm);
                    if (jabs != jref || dabs != dref) {
                        std::cerr << "raw orbit target mismatch mask=" << mask
                                  << " p=" << p << " src=" << src_abs
                                  << " j=" << jabs << '/' << jref
                                  << " d=" << dabs << '/' << dref << '\n';
                        return 226;
                    }
                    ++owner_checks;
                } else if (ok == HIGH_ORBIT_CLOSURE) {
                    uint32_t dw = desc.main_desc[size_t(pi) * desc.main_total
                                                 + desc.main_base[bid] + hr];
                    uint32_t kind = hd_kind(dw);
                    uint32_t dbid = hd_block(dw);
                    uint32_t width = lowmask_major_width(mask, logical.block_blocks[dbid].hs);
                    Code dabs = 0;
                    if (kind == HIGHDESC_BLOCK) {
                        dabs = lowmask_major_block_block_base(mm, mask, dbid)
                             + Code(hd_rank(dw)) * width + lr;
                    } else if (kind == HIGHDESC_CROSS) {
                        constexpr uint32_t LM = (1u << (2 * LOW_LUT_K)) - 1u;
                        uint32_t lc = uint32_t(m) & LM;
                        uint32_t lc2 = host_high_flip_low(lc, hd_depth(dw));
                        if (lc2 == 0xffffffffu) return 227;
                        uint32_t packed = storage.low_packed_rank[lc2];
                        if (packed == 0xffffffffu) return 228;
                        uint32_t lr2 = packed & ((1u << LOW_LUT_K) - 1u);
                        dabs = lowmask_major_block_block_base(mm, mask, dbid)
                             + Code(hd_rank(dw)) * width + lr2;
                    } else {
                        return 229;
                    }
                    auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                    if (!z.valid || !z.blocked) return 230;
                    Code dref = lowmask_major_rank_block_host(z.mate, storage, logical, mm);
                    if (dabs != dref) {
                        std::cerr << "raw closure target mismatch mask=" << mask
                                  << " p=" << p << " src=" << src_abs
                                  << " d=" << dabs << '/' << dref << '\n';
                        return 231;
                    }
                    ++closure_checks;
                }
            }
        }
    }

    for (size_t i = 0; i < seen.size(); ++i) {
        if (!seen[i]) {
            std::cerr << "raw task coverage hole i=" << i << '\n';
            return 232;
        }
    }

    std::cout << "high-rawbatch-selftest OK W=" << TARGET_W
              << " tasks=" << bt.tasks.size()
              << " states=" << mm.main_size
              << " owner_checks=" << owner_checks
              << " closure_checks=" << closure_checks << '\n';
    return 0;
}
