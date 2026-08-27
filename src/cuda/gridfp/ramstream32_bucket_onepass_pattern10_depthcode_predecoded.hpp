#pragma once

#include "ramstream32_bucket_onepass_pattern10_depthcode.hpp"

// Predecode each context-local 10-bit orbit code into the exact information
// needed by the closure-plan builder.  For factor width <=14 the left/right
// pattern masks need at most 13 bits each; together with depth4 and an active
// bit this fits in one uint32_t.
static constexpr uint32_t P10DCP_MASK_BITS = 13;
static constexpr uint32_t P10DCP_MASK_MASK = (1u << P10DCP_MASK_BITS) - 1u;
static constexpr uint32_t P10DCP_RM_SHIFT = P10DCP_MASK_BITS;
static constexpr uint32_t P10DCP_DEPTH_SHIFT = 2 * P10DCP_MASK_BITS;
static constexpr uint32_t P10DCP_ACTIVE_SHIFT = P10DCP_DEPTH_SHIFT + 4;
static_assert(P10DCP_ACTIVE_SHIFT < 32);

struct P10DepthCodePredecodedBookHost {
    uint32_t mode = P10DC_STREAM;
    std::vector<uint32_t> base;
    std::vector<uint32_t> decode;
    size_t bytes() const {
        return base.size() * sizeof(uint32_t) + decode.size() * sizeof(uint32_t);
    }
};

struct BucketForwardPattern10DepthCodePredecodedHost {
    size_t bytes() const { return 0; }
};
struct BucketReversePattern10DepthCodePredecodedHost {
    ReverseSplit54Host split;
    P10DepthCodePredecodedBookHost codebook;
    size_t bytes() const { return split.bytes() + codebook.bytes(); }
};

static BucketForwardPattern10DepthCodePredecodedHost
build_bucket_forward_pattern10_depthcode_predecoded_placeholder(
    const StorageLayout&, BucketOrbitStreamsHost&, BucketFusedHost&
) {
    return {};
}

static uint32_t p10dcp_pack(uint16_t pair, bool high, int absolute_p) {
    using namespace oneesan::gridfp;
    const uint16_t pattern = uint16_t(pair >> 4);
    const uint32_t depth = uint32_t(pair & 15u);
    if (pattern == CLOSURE_PATTERN10_NONE) return 0;

    const int p = high ? absolute_p - LOW_LUT_K : absolute_p;
    const int len = high ? HIGH_LUT_K + 1 : LOW_LUT_K + 1;
    uint16_t lm = 0, rm = 0;
    closure_pattern10_decode(pattern, len, p, lm, rm);
    if ((uint32_t(lm) & ~P10DCP_MASK_MASK) ||
        (uint32_t(rm) & ~P10DCP_MASK_MASK)) {
        std::cerr << "pattern10 depthcode predecode mask overflow high=" << high
                  << " p=" << absolute_p << " lm=" << lm << " rm=" << rm << '\n';
        std::exit(599);
    }
    return uint32_t(lm)
         | (uint32_t(rm) << P10DCP_RM_SHIFT)
         | (depth << P10DCP_DEPTH_SHIFT)
         | (1u << P10DCP_ACTIVE_SHIFT);
}

static BucketReversePattern10DepthCodePredecodedHost
build_bucket_reverse_pattern10_depthcode_predecoded_zero_checked(
    const StorageLayout& layout, BucketOrbitStreamsHost& bo, BucketFusedHost& bf,
    ReverseBucketAtomicHost& rb, ReverseBucketFusedHost& rf
) {
    BucketReversePattern10DepthCodeHost src =
        build_bucket_reverse_pattern10_depthcode_zero_checked(layout, bo, bf, rb, rf);

    BucketReversePattern10DepthCodePredecodedHost out;
    out.split = std::move(src.split);
    out.codebook.mode = src.codebook.mode;
    out.codebook.base = std::move(src.codebook.base);
    out.codebook.decode.resize(src.codebook.decode.size());

    std::vector<uint32_t> active_keys;
    active_keys.reserve(out.codebook.base.size());
    for (uint32_t k = 0; k < out.codebook.base.size(); ++k) {
        if (out.codebook.base[k] != P10DC_INVALID_BASE) active_keys.push_back(k);
    }

    for (size_t ci = 0; ci < active_keys.size(); ++ci) {
        const uint32_t k = active_keys[ci];
        const uint32_t begin = out.codebook.base[k];
        const uint32_t end = ci + 1 < active_keys.size()
            ? out.codebook.base[active_keys[ci + 1]]
            : uint32_t(src.codebook.decode.size());
        if (begin > end || end > src.codebook.decode.size()) std::exit(600);

        uint32_t t = k / P10DC_HDIM;
        const int absolute_p = int(t % uint32_t(TARGET_W));
        t /= uint32_t(TARGET_W);
        t /= P10DC_STREAMS;
        const bool high = (t & 1u) != 0;

        for (uint32_t i = begin; i < end; ++i) {
            out.codebook.decode[i] = p10dcp_pack(src.codebook.decode[i], high, absolute_p);
        }
    }

    std::cerr << "pattern10_depthcode_predecoded mode=" << out.codebook.mode
              << " contexts=" << active_keys.size()
              << " entries=" << out.codebook.decode.size()
              << " codebook_mib=" << double(out.codebook.bytes()) / double(1 << 20)
              << " runtime_pattern_decode=0 sidecar_bytes_per_orbit=0\n";
    return out;
}
