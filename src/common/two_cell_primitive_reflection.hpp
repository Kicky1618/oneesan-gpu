#pragma once

#include "two_cell_primitive_meta.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_TC_PR_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_PR_HD inline
#endif

namespace oneesan::twocell {

constexpr int kPrimitiveMirrorRankBits = 22;
constexpr std::uint32_t kPrimitiveMirrorRankMask =
    (std::uint32_t(1) << kPrimitiveMirrorRankBits) - 1u;
constexpr int kPrimitiveReflectionRootShift = kPrimitiveMirrorRankBits;

ONEESAN_TC_PR_HD std::uint32_t pack_primitive_reflection(
    Rank mirror_rank,
    int root_ordinal
) {
    return (static_cast<std::uint32_t>(mirror_rank) & kPrimitiveMirrorRankMask) |
           (std::uint32_t(root_ordinal) << kPrimitiveReflectionRootShift);
}

ONEESAN_TC_PR_HD Rank primitive_reflection_mirror_rank(std::uint32_t meta) {
    return static_cast<Rank>(meta & kPrimitiveMirrorRankMask);
}

ONEESAN_TC_PR_HD int primitive_reflection_root_ordinal(std::uint32_t meta) {
    return static_cast<int>((meta >> kPrimitiveReflectionRootShift) & 31u);
}

ONEESAN_TC_PR_HD std::uint32_t make_primitive_reflection_meta(
    std::uint32_t compact_left,
    int occupied,
    const RankTables& t
) {
    const int root = compact_root_ordinal(compact_left, occupied);
    if (root < 0) return 0xffffffffu;
    const PackedWord compact{
        low_mask(occupied), compact_left, static_cast<std::uint8_t>(occupied)};
    const PackedWord mirrored = reflect_word_with_root_ordinal(compact, root);
    const Rank mirror_rank = primitive_rank(
        mirrored.support, mirrored.left, occupied, t);
    return pack_primitive_reflection(mirror_rank, root);
}

ONEESAN_TC_PR_HD PackedWord reflect_word_with_reflection_meta(
    PackedWord word,
    std::uint32_t meta
) {
    return reflect_word_with_root_ordinal(
        word, primitive_reflection_root_ordinal(meta));
}

ONEESAN_TC_PR_HD PackedKey reflect_key_with_reflection_meta(
    PackedKey key,
    int W,
    std::uint32_t meta
) {
    return reflect_key_with_root_ordinal(
        key, W, primitive_reflection_root_ordinal(meta));
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_PR_HD
