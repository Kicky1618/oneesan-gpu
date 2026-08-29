#pragma once

#include "two_cell_component_device.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_TC_PM_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_PM_HD inline
#endif

namespace oneesan::twocell {

// W<=28 means an A state has at most 27 occupied-sequence slots.  A compact L
// mask therefore uses low bits 0..26, leaving five high bits in one uint32 for
// the distinguished root's occupied ordinal (0..26).
constexpr int kPrimitiveMetaRootShift = 27;
constexpr std::uint32_t kPrimitiveMetaLeftMask =
    (std::uint32_t(1) << kPrimitiveMetaRootShift) - 1u;

ONEESAN_TC_PM_HD std::uint32_t pack_primitive_meta(
    std::uint32_t compact_left,
    int root_ordinal
) {
    return (compact_left & kPrimitiveMetaLeftMask) |
           (std::uint32_t(root_ordinal) << kPrimitiveMetaRootShift);
}

ONEESAN_TC_PM_HD std::uint32_t primitive_meta_left(std::uint32_t meta) {
    return meta & kPrimitiveMetaLeftMask;
}

ONEESAN_TC_PM_HD int primitive_meta_root_ordinal(std::uint32_t meta) {
    return static_cast<int>(meta >> kPrimitiveMetaRootShift);
}

// Compute the distinguished root ordinal in a compact occupied L/R sequence.
// The sequence starts at height one.  The first R that returns to height zero
// is the unmatched/root endpoint used by the packed reflection convention.
ONEESAN_TC_PM_HD int compact_root_ordinal(
    std::uint32_t compact_left,
    int occupied
) {
    int h = 1;
    for (int q = 0; q < occupied; ++q) {
        if ((compact_left >> q) & 1u) ++h;
        else --h;
        if (h == 0) return q;
        if (h < 0) return -1;
    }
    return -1;
}

// Select the zero-based ordinal-th set bit of a 32-bit mask with five popcount
// decisions rather than a loop over physical positions.  Returns -1 if the
// ordinal is outside the mask population.
ONEESAN_TC_PM_HD int select_nth_set32(std::uint32_t mask, int ordinal) {
    if (ordinal < 0 || ordinal >= popcount32(mask)) return -1;
    int pos = 0;

    int c = popcount32(mask & 0x0000ffffu);
    if (ordinal >= c) {
        ordinal -= c;
        mask >>= 16;
        pos += 16;
    }
    c = popcount32(mask & 0x000000ffu);
    if (ordinal >= c) {
        ordinal -= c;
        mask >>= 8;
        pos += 8;
    }
    c = popcount32(mask & 0x0000000fu);
    if (ordinal >= c) {
        ordinal -= c;
        mask >>= 4;
        pos += 4;
    }
    c = popcount32(mask & 0x00000003u);
    if (ordinal >= c) {
        ordinal -= c;
        mask >>= 2;
        pos += 2;
    }
    c = popcount32(mask & 0x00000001u);
    if (ordinal >= c) {
        mask >>= 1;
        pos += 1;
    }
    return pos;
}

ONEESAN_TC_PM_HD std::uint32_t reverse_bits_len_meta(
    std::uint32_t x,
    int len
) {
#if defined(__CUDA_ARCH__)
    return len <= 0 ? 0u : (__brev(x) >> (32 - len));
#else
    std::uint32_t y = 0;
    for (int p = 0; p < len; ++p)
        if ((x >> p) & 1u) y |= std::uint32_t(1) << (len - 1 - p);
    return y;
#endif
}

// Reflect a packed word when its primitive metadata is already known.  This is
// the reverse-sweep hot-path form: primitive rank/address work supplies the LUT
// entry, so reflection needs no root scan or mate reconstruction.
ONEESAN_TC_PM_HD PackedWord reflect_word_with_primitive_meta(
    PackedWord w,
    std::uint32_t meta
) {
    const std::uint32_t active = w.support & low_mask(w.len);
    const int root = select_nth_set32(
        active, primitive_meta_root_ordinal(meta));
    if (root < 0) return PackedWord{};
    const std::uint32_t rbits = active & ~w.left;
    const std::uint32_t matched_r = rbits & ~(std::uint32_t(1) << root);
    return PackedWord{
        reverse_bits_len_meta(active, w.len),
        reverse_bits_len_meta(matched_r, w.len),
        w.len
    };
}

ONEESAN_TC_PM_HD PackedKey reflect_key_with_primitive_meta(
    PackedKey key,
    int W,
    std::uint32_t meta
) {
    return make_state(
        key.type,
        reflect_word_with_primitive_meta(state_word(key, W), meta));
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_PM_HD
