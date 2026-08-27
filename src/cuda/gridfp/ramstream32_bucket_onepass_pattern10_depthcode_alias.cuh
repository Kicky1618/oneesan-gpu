#pragma once

#include "ramstream32_bucket_onepass_pattern10_depth8_alias.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// The orbit word has ten spare bits above the three 18-bit locators.  For a
// fixed (phase, side, p, boundary-height) context there are <=1024 reachable
// (pattern10, cross-depth) pairs at W=28; gridfp_pattern10_depthcode_bound.cpp
// proves the stronger all-legal-factor-state bound (718 at 14+13).
//
// Encode those pairs densely into the existing ten orbit bits.  The device
// decode table stores the already-decoded ordinary masks plus CROSS depth, so
// runtime needs neither the depth sidecar nor Fibonacci pattern unranking/LUT.
static constexpr uint32_t P10DC_CODE_COUNT = 1024;
static constexpr uint32_t P10DC_PAIR_COUNT = 1024 * 16;
static constexpr uint32_t P10DC_MASK_BITS = 14;
static constexpr uint32_t P10DC_MASK_MASK = (1u << P10DC_MASK_BITS) - 1u;
static constexpr uint32_t P10DC_RM_SHIFT = P10DC_MASK_BITS;
static constexpr uint32_t P10DC_DEPTH_SHIFT = 2 * P10DC_MASK_BITS;
static_assert(P10DC_DEPTH_SHIFT + 4 == 32);
static_assert(LOW_LUT_K <= 14 && HIGH_LUT_K <= 14);

static constexpr uint32_t P10DC_LOW_H = HIGH_LUT_K + 2;
static constexpr uint32_t P10DC_HIGH_H = LOW_LUT_K + 2;
static constexpr uint32_t P10DC_LOW_CONTEXTS = LOW_LUT_K * P10DC_LOW_H;
static constexpr uint32_t P10DC_HIGH_CONTEXTS = HIGH_LUT_K * P10DC_HIGH_H;
static constexpr uint32_t P10DC_CONTEXTS = P10DC_LOW_CONTEXTS + P10DC_HIGH_CONTEXTS;

#if defined(__CUDACC__)
#define P10DC_HD __host__ __device__ __forceinline__
#else
#define P10DC_HD inline
#endif
P10DC_HD uint32_t p10dc_low_context(int p, uint32_t h) {
    return uint32_t(p - 1) * P10DC_LOW_H + h;
}
P10DC_HD uint32_t p10dc_high_context(int rel, uint32_t h) {
    return P10DC_LOW_CONTEXTS + uint32_t(rel - 1) * P10DC_HIGH_H + h;
}
P10DC_HD uint16_t p10dc_orbit_code(BucketOrbitOp op) {
    return uint16_t((op >> 54) & 0x3ffu);
}
P10DC_HD uint16_t p10dc_packed_lm(uint32_t x) { return uint16_t(x & P10DC_MASK_MASK); }
P10DC_HD uint16_t p10dc_packed_rm(uint32_t x) { return uint16_t((x >> P10DC_RM_SHIFT) & P10DC_MASK_MASK); }
P10DC_HD uint8_t p10dc_packed_depth(uint32_t x) { return uint8_t(x >> P10DC_DEPTH_SHIFT); }
#undef P10DC_HD

static BucketOrbitOp p10dc_set_code(BucketOrbitOp op, uint16_t code) {
    if (code >= P10DC_CODE_COUNT) {
        std::cerr << "pattern10 depthcode overflow code=" << code << '\n';
        std::exit(583);
    }
    return BucketOrbitOp((op & BKCP10_BASE_MASK) | (uint64_t(code) << 54));
}

struct BucketPattern10DepthCodeHost {
    std::vector<uint32_t> decode;
    uint32_t max_codes = 0;
    uint32_t used_contexts = 0;
    size_t bytes() const { return decode.size() * sizeof(uint32_t); }
};

struct BucketPattern10DepthCodeBuilder {
    std::vector<uint16_t> encode;
    std::vector<uint16_t> next;
    BucketPattern10DepthCodeHost out;

    BucketPattern10DepthCodeBuilder()
        : encode(size_t(P10DC_CONTEXTS) * P10DC_PAIR_COUNT, 0xffffu),
          next(P10DC_CONTEXTS, 0),
          out{std::vector<uint32_t>(size_t(P10DC_CONTEXTS) * P10DC_CODE_COUNT, 0), 0, 0} {}

    uint16_t put(uint32_t ctx, uint16_t pattern, uint8_t depth, int len, int p) {
        if (ctx >= P10DC_CONTEXTS || pattern > 0x3ffu || depth > 15) {
            std::cerr << "pattern10 depthcode invalid ctx=" << ctx
                      << " pattern=" << pattern << " depth=" << unsigned(depth) << '\n';
            std::exit(584);
        }
        if (pattern == oneesan::gridfp::CLOSURE_PATTERN10_NONE) depth = 0;
        const uint16_t pair = uint16_t((pattern << 4) | depth);
        uint16_t& code = encode[size_t(ctx) * P10DC_PAIR_COUNT + pair];
        if (code != 0xffffu) return code;
        code = next[ctx]++;
        if (code >= P10DC_CODE_COUNT) {
            std::cerr << "pattern10 depthcode context overflow ctx=" << ctx
                      << " codes=" << unsigned(next[ctx]) << '\n';
            std::exit(585);
        }
        uint16_t lm = 0, rm = 0;
        oneesan::gridfp::closure_pattern10_decode(pattern, len, p, lm, rm);
        if ((uint32_t(lm) & ~P10DC_MASK_MASK) || (uint32_t(rm) & ~P10DC_MASK_MASK)) {
            std::cerr << "pattern10 depthcode mask overflow ctx=" << ctx << '\n';
            std::exit(586);
        }
        out.decode[size_t(ctx) * P10DC_CODE_COUNT + code] =
            uint32_t(lm) | (uint32_t(rm) << P10DC_RM_SHIFT) |
            (uint32_t(depth) << P10DC_DEPTH_SHIFT);
        return code;
    }

    BucketPattern10DepthCodeHost finish() {
        for (uint16_t n : next) {
            if (n) ++out.used_contexts;
            out.max_codes = std::max(out.max_codes, uint32_t(n));
        }
        if (out.max_codes > P10DC_CODE_COUNT) std::exit(587);
        std::vector<uint16_t>().swap(encode);
        std::vector<uint16_t>().swap(next);
        return std::move(out);
    }
};

struct BucketForwardPattern10DepthCodeHost {
    BucketPattern10DepthCodeHost codebook;
    size_t bytes() const { return codebook.bytes(); }
};
struct BucketReversePattern10DepthCodeHost {
    ReverseSplit54Host split;
    BucketPattern10DepthCodeHost codebook;
    size_t bytes() const { return split.bytes() + codebook.bytes(); }
};

static uint8_t p10dc_low_depth(MateID d, int p, const char* what) {
    MateID source = 0;
    int depth = oneesan::gridfp::low_cross_preimage_partial(d, LOW_LUT_K + 1, p, source);
    if (depth < 0 || depth > 15) {
        std::cerr << "pattern10 depthcode LOW overflow " << what
                  << " p=" << p << " depth=" << depth << '\n';
        std::exit(588);
    }
    return uint8_t(depth);
}
static uint8_t p10dc_high_depth(MateID d, int rel, const char* what) {
    MateID source = 0;
    int depth = oneesan::gridfp::high_cross_preimage_partial(d, HIGH_LUT_K + 1, rel, source);
    if (depth < 0 || depth > 15) {
        std::cerr << "pattern10 depthcode HIGH overflow " << what
                  << " rel=" << rel << " depth=" << depth << '\n';
        std::exit(589);
    }
    return uint8_t(depth);
}

static BucketForwardPattern10DepthCodeHost build_bucket_forward_pattern10_depthcode_zero(
    const StorageLayout& layout, BucketOrbitStreamsHost& bo, BucketFusedHost& bf
) {
    BucketForwardPattern10DepthCodeHost out;
    BucketPattern10DepthCodeBuilder cb;
    const size_t lp = size_t(bo.low_nblocks) + 1, hp = size_t(bo.high_nblocks) + 1;
    uint64_t ops = 0;

    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t bid = 0; bid < bo.low_nblocks; ++bid) {
            const auto& xb = layout.main_blocks[bid];
            if (!xb.valid) continue;
            const auto& db = p == 1 ? xb : layout.block_blocks[xb.he];
            const uint32_t ctx = p10dc_low_context(p, xb.he);
            auto scan = [&](std::vector<BucketOrbitOp>& v, const std::vector<uint32_t>& off,
                            bool active, const char* what) {
                uint32_t a = off[size_t(pi) * lp + bid], b = off[size_t(pi) * lp + bid + 1];
                for (uint32_t q = a; q < b; ++q) {
                    uint16_t id = oneesan::gridfp::CLOSURE_PATTERN10_NONE;
                    uint8_t depth = 0;
                    if (active) {
                        uint32_t loc = p == 1 ? bkf_orbit_src(v[q]) : bkf_orbit_drop(v[q]);
                        uint32_t dc = bkcp10_low_code_host(bf, loc, db.hs);
                        MateID d = p == 1
                            ? (MateID(dc) | (MateID(db.c) << (2 * LOW_LUT_K)))
                            : minsert(MateID(dc), p, N);
                        id = oneesan::gridfp::closure_pattern10_encode(d, LOW_LUT_K + 1, p);
                        depth = p10dc_low_depth(d, p, what);
                    }
                    v[q] = p10dc_set_code(v[q], cb.put(ctx, id, depth, LOW_LUT_K + 1, p));
                    ++ops;
                }
            };
            scan(bo.low_nn, bo.low_nn_off, true, "forward-low-nn");
            scan(bo.low_nr, bo.low_nr_off, p != 1, "forward-low-nr");
            scan(bo.low_nl, bo.low_nl_off, p != 1, "forward-low-nl");
        }
    }

    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        int rel = p - LOW_LUT_K;
        for (uint32_t bid = 0; bid < bo.high_nblocks; ++bid) {
            const auto& xb = layout.main_blocks[bid];
            if (!xb.valid) continue;
            const auto& db = layout.block_blocks[xb.hs];
            const uint32_t ctx = p10dc_high_context(rel, xb.hs);
            auto scan = [&](std::vector<BucketOrbitOp>& v, const std::vector<uint32_t>& off,
                            const char* what) {
                uint32_t a = off[size_t(pi) * hp + bid], b = off[size_t(pi) * hp + bid + 1];
                for (uint32_t q = a; q < b; ++q) {
                    uint32_t loc = bkf_orbit_drop(v[q]);
                    uint32_t dc = bkcp10_high_code_host(bf, loc, db.he);
                    MateID d = minsert(MateID(dc), rel, N);
                    uint16_t id = oneesan::gridfp::closure_pattern10_encode(d, HIGH_LUT_K + 1, rel);
                    uint8_t depth = p10dc_high_depth(d, rel, what);
                    v[q] = p10dc_set_code(v[q], cb.put(ctx, id, depth, HIGH_LUT_K + 1, rel));
                    ++ops;
                }
            };
            scan(bo.high_nn, bo.high_nn_off, "forward-high-nn");
            scan(bo.high_nrnl, bo.high_nrnl_off, "forward-high-nrnl");
        }
    }

    out.codebook = cb.finish();
    std::cerr << "bucket_forward_pattern10_depthcode ops=" << ops
              << " codebook_mib=" << double(out.codebook.bytes()) / double(1 << 20)
              << " used_contexts=" << out.codebook.used_contexts
              << " max_codes=" << out.codebook.max_codes
              << " depth_sidecar_bytes=0 decode_unrank=0 direct_build=1\n";
    bucket_zero_release_forward_closure(bf);
    return out;
}

static BucketReversePattern10DepthCodeHost build_bucket_reverse_pattern10_depthcode_zero_checked(
    const StorageLayout& layout, const BucketOrbitStreamsHost&, BucketFusedHost& bf,
    ReverseBucketAtomicHost& rb, ReverseBucketFusedHost& rf
) {
    BucketReversePattern10DepthCodeHost out;
    out.split = build_reverse_split54(layout, rb, true);
    auto& rs = out.split;
    BucketPattern10DepthCodeBuilder cb;
    const size_t rp = size_t(rs.nblocks) + 1;
    uint64_t ops = 0;

    for (int p = 1; p <= LOW_LUT_K; ++p) {
        uint32_t pi = uint32_t(p - 1);
        for (uint32_t bid = 0; bid < rs.nblocks; ++bid) {
            const auto& xb = layout.main_blocks[bid];
            if (!xb.valid) continue;
            const auto& db = layout.block_blocks[xb.he];
            const uint32_t ctx = p10dc_low_context(p, xb.he);
            auto scan = [&](std::vector<BucketOrbitOp>& v, const std::vector<uint32_t>& off,
                            const char* what) {
                uint32_t a = off[size_t(pi) * rp + bid], b = off[size_t(pi) * rp + bid + 1];
                for (uint32_t q = a; q < b; ++q) {
                    uint32_t loc = bkf_orbit_drop(v[q]);
                    uint32_t dc = bkcp10_low_code_host(bf, loc, db.hs);
                    MateID d = blocked_exclude_reverse(MateID(dc), LOW_LUT_K + 1, p);
                    uint16_t id = oneesan::gridfp::closure_pattern10_encode(d, LOW_LUT_K + 1, p);
                    uint8_t depth = p10dc_low_depth(d, p, what);
                    v[q] = p10dc_set_code(v[q], cb.put(ctx, id, depth, LOW_LUT_K + 1, p));
                    ++ops;
                }
            };
            scan(rs.low.nn, rs.low.nn_off, "reverse-low-nn");
            scan(rs.low.nr, rs.low.nr_off, "reverse-low-nr");
            scan(rs.low.nl, rs.low.nl_off, "reverse-low-nl");
        }
    }

    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
        int rel = p - LOW_LUT_K;
        bool edge = p == TARGET_W - 1;
        for (uint32_t bid = 0; bid < rs.nblocks; ++bid) {
            const auto& xb = layout.main_blocks[bid];
            if (!xb.valid) continue;
            const auto& db = edge ? xb : layout.block_blocks[xb.hs];
            const uint32_t ctx = p10dc_high_context(rel, xb.hs);
            auto scan = [&](std::vector<BucketOrbitOp>& v, const std::vector<uint32_t>& off,
                            bool nn, const char* what) {
                uint32_t a = off[size_t(pi) * rp + bid], b = off[size_t(pi) * rp + bid + 1];
                for (uint32_t q = a; q < b; ++q) {
                    uint16_t id = oneesan::gridfp::CLOSURE_PATTERN10_NONE;
                    uint8_t depth = 0;
                    if (!edge || nn) {
                        uint32_t loc = edge ? bkf_orbit_src(v[q]) : bkf_orbit_drop(v[q]);
                        uint32_t dc = bkcp10_high_code_host(bf, loc, db.he);
                        MateID d = edge
                            ? (MateID(db.c) | (MateID(dc) << 2))
                            : blocked_exclude_reverse(MateID(dc), HIGH_LUT_K + 1, rel);
                        id = oneesan::gridfp::closure_pattern10_encode(d, HIGH_LUT_K + 1, rel);
                        depth = p10dc_high_depth(d, rel, what);
                    }
                    v[q] = p10dc_set_code(v[q], cb.put(ctx, id, depth, HIGH_LUT_K + 1, rel));
                    ++ops;
                }
            };
            scan(rs.high.nn, rs.high.nn_off, true, "reverse-high-nn");
            scan(rs.high.nr, rs.high.nr_off, false, "reverse-high-nr");
            scan(rs.high.nl, rs.high.nl_off, false, "reverse-high-nl");
        }
    }

    out.codebook = cb.finish();
    std::cerr << "bucket_reverse_pattern10_depthcode ops=" << ops
              << " split54_mib=" << double(out.split.bytes()) / double(1 << 20)
              << " codebook_mib=" << double(out.codebook.bytes()) / double(1 << 20)
              << " used_contexts=" << out.codebook.used_contexts
              << " max_codes=" << out.codebook.max_codes
              << " depth_sidecar_bytes=0 decode_unrank=0 direct_build=1\n";
    bucket_zero_release_reverse_closure(rf);
    return out;
}

__constant__ uint32_t* D_P10DC_FORWARD = nullptr;
__constant__ uint32_t* D_P10DC_REVERSE = nullptr;

struct BucketForwardPattern10DepthCodeDeviceTables {
    uint32_t* decode = nullptr;
    void install(const BucketForwardPattern10DepthCodeHost& h) {
        if (!h.codebook.decode.empty()) {
            ck(cudaMalloc(&decode, h.codebook.bytes()), "p10dc forward decode");
            ck(cudaMemcpy(decode, h.codebook.decode.data(), h.codebook.bytes(), cudaMemcpyHostToDevice),
               "p10dc forward decode H2D");
        }
        ck(cudaMemcpyToSymbol(D_P10DC_FORWARD, &decode, sizeof(decode)), "p10dc forward ptr");
    }
    void release() { cudaFree(decode); decode = nullptr; }
};
struct BucketReversePattern10DepthCodeDeviceTables {
    ReverseSplit54DeviceTables split;
    uint32_t* decode = nullptr;
    void install(const BucketReversePattern10DepthCodeHost& h) {
        split.install(h.split);
        if (!h.codebook.decode.empty()) {
            ck(cudaMalloc(&decode, h.codebook.bytes()), "p10dc reverse decode");
            ck(cudaMemcpy(decode, h.codebook.decode.data(), h.codebook.bytes(), cudaMemcpyHostToDevice),
               "p10dc reverse decode H2D");
        }
        ck(cudaMemcpyToSymbol(D_P10DC_REVERSE, &decode, sizeof(decode)), "p10dc reverse ptr");
    }
    void release() { cudaFree(decode); decode = nullptr; split.release(); }
};

__device__ __forceinline__ uint32_t p10dc_forward_decode(BucketOrbitOp op, uint32_t ctx) {
    return D_P10DC_FORWARD[size_t(ctx) * P10DC_CODE_COUNT + p10dc_orbit_code(op)];
}
__device__ __forceinline__ uint32_t p10dc_reverse_decode(BucketOrbitOp op, uint32_t ctx) {
    return D_P10DC_REVERSE[size_t(ctx) * P10DC_CODE_COUNT + p10dc_orbit_code(op)];
}
