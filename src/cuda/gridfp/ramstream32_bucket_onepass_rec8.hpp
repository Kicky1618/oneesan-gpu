#pragma once

#include "ramstream32_bucket_fused.cuh"
#include "ramstream32_bucket_reverse_fused.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// One-pass orbit+closure does not need dst_locator in each destination record:
// the owning orbit already determines the destination block and active-half
// locator.  local and CROSS sources use the same 32-bit encoding; depth==0 is
// an ordinary source and depth>0 is CROSS.  Concatenate both slices into one
// source stream and keep only begin + packed counts.
//
// Record size: 8 B instead of BucketFusedDst's 16 B.  Source-entry count is
// unchanged; the two source arrays become one contiguous array per side.
struct BucketOnePassRec8 {
    uint32_t begin = 0;
    uint32_t counts = 0; // low16=ordinary, high16=CROSS
};
static_assert(sizeof(BucketOnePassRec8) == 8);

struct BucketForwardOnePassRec8Host {
    std::vector<BucketOnePassRec8> low_rec, high_rec;
    std::vector<uint32_t> low_src, high_src;
    size_t bytes() const {
        return (low_rec.size() + high_rec.size()) * sizeof(BucketOnePassRec8)
            + (low_src.size() + high_src.size()) * sizeof(uint32_t);
    }
};
struct BucketReverseOnePassRec8Host {
    std::vector<BucketOnePassRec8> low_rec, high_rec;
    std::vector<uint32_t> low_src, high_src;
    size_t bytes() const {
        return (low_rec.size() + high_rec.size()) * sizeof(BucketOnePassRec8)
            + (low_src.size() + high_src.size()) * sizeof(uint32_t);
    }
};

static void bucket_onepass_rec8_append(
    std::vector<BucketOnePassRec8>& dst,
    std::vector<uint32_t>& src,
    const BucketFusedDst& r,
    const std::vector<uint32_t>& local,
    const std::vector<uint32_t>& cross,
    const char* what
) {
    uint32_t lc = r.counts & 0xffffu, cc = r.counts >> 16;
    if (uint64_t(r.local_begin) + lc > local.size()
        || uint64_t(r.cross_begin) + cc > cross.size()) {
        std::cerr << "one-pass rec8 source slice overflow " << what << '\n';
        std::exit(440);
    }
    if (src.size() > 0xffffffffull - uint64_t(lc) - uint64_t(cc)) {
        std::cerr << "one-pass rec8 source stream exceeds uint32 begin " << what << '\n';
        std::exit(441);
    }
    uint32_t begin = uint32_t(src.size());
    src.insert(src.end(), local.begin() + r.local_begin,
               local.begin() + r.local_begin + lc);
    src.insert(src.end(), cross.begin() + r.cross_begin,
               cross.begin() + r.cross_begin + cc);
    dst.push_back({begin, r.counts});
}

static BucketForwardOnePassRec8Host build_bucket_forward_onepass_rec8(
    const BucketFusedHost& f
) {
    BucketForwardOnePassRec8Host out;
    out.low_rec.reserve(f.low_dst.size());
    out.high_rec.reserve(f.high_dst.size());
    out.low_src.reserve(f.low_local_src.size() + f.low_cross_op.size());
    out.high_src.reserve(f.high_local_src.size() + f.high_cross_op.size());
    for (const auto& r : f.low_dst)
        bucket_onepass_rec8_append(out.low_rec, out.low_src, r,
                                   f.low_local_src, f.low_cross_op, "forward-low");
    for (const auto& r : f.high_dst)
        bucket_onepass_rec8_append(out.high_rec, out.high_src, r,
                                   f.high_local_src, f.high_cross_op, "forward-high");
    if (out.low_src.size() != f.low_local_src.size() + f.low_cross_op.size()
        || out.high_src.size() != f.high_local_src.size() + f.high_cross_op.size()) {
        std::cerr << "one-pass rec8 forward source partition mismatch\n";
        std::exit(442);
    }
    size_t old_bytes = (f.low_dst.size() + f.high_dst.size()) * sizeof(BucketFusedDst)
        + (f.low_local_src.size() + f.low_cross_op.size()
           + f.high_local_src.size() + f.high_cross_op.size()) * sizeof(uint32_t);
    std::cerr << "bucket_forward_onepass_rec8 records="
              << (out.low_rec.size() + out.high_rec.size())
              << " sources=" << (out.low_src.size() + out.high_src.size())
              << " old_mib=" << double(old_bytes) / double(1 << 20)
              << " rec8_mib=" << double(out.bytes()) / double(1 << 20)
              << "\n";
    return out;
}

static BucketReverseOnePassRec8Host build_bucket_reverse_onepass_rec8(
    const ReverseBucketFusedHost& f
) {
    BucketReverseOnePassRec8Host out;
    out.low_rec.reserve(f.low_dst.size());
    out.high_rec.reserve(f.high_dst.size());
    out.low_src.reserve(f.low_local_src.size() + f.low_cross_op.size());
    out.high_src.reserve(f.high_local_src.size() + f.high_cross_op.size());
    for (const auto& r : f.low_dst)
        bucket_onepass_rec8_append(out.low_rec, out.low_src, r,
                                   f.low_local_src, f.low_cross_op, "reverse-low");
    for (const auto& r : f.high_dst)
        bucket_onepass_rec8_append(out.high_rec, out.high_src, r,
                                   f.high_local_src, f.high_cross_op, "reverse-high");
    if (out.low_src.size() != f.low_local_src.size() + f.low_cross_op.size()
        || out.high_src.size() != f.high_local_src.size() + f.high_cross_op.size()) {
        std::cerr << "one-pass rec8 reverse source partition mismatch\n";
        std::exit(443);
    }
    size_t old_bytes = (f.low_dst.size() + f.high_dst.size()) * sizeof(BucketFusedDst)
        + (f.low_local_src.size() + f.low_cross_op.size()
           + f.high_local_src.size() + f.high_cross_op.size()) * sizeof(uint32_t);
    std::cerr << "bucket_reverse_onepass_rec8 records="
              << (out.low_rec.size() + out.high_rec.size())
              << " sources=" << (out.low_src.size() + out.high_src.size())
              << " old_mib=" << double(old_bytes) / double(1 << 20)
              << " rec8_mib=" << double(out.bytes()) / double(1 << 20)
              << "\n";
    return out;
}
