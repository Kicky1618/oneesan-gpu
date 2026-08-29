#pragma once

#include "../../common/two_cell_turn_closed_device.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_reflect {

constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ std::uint32_t reverse_bits_len_fast(
    std::uint32_t x,
    int len
) {
    return len <= 0 ? 0u : (__brev(x) >> (32 - len));
}

__device__ __forceinline__ int root_position_warp(PackedWord w) {
    const int lane = threadIdx.x & 31;
    const bool in_range = lane < w.len;
    const Symbol c = in_range ? symbol(w, lane) : TC_N;
    const std::uint32_t prefix = low_mask(lane);
    const int before = in_range
        ? 1 + 2 * __popc(w.left & prefix) - __popc(w.support & prefix)
        : 1;
    const int after = before + (c == TC_L ? 1 : (c == TC_R ? -1 : 0));
    const unsigned roots = __ballot_sync(
        kFullWarp, in_range && c == TC_R && after == 0);
    return roots ? (__ffs(static_cast<int>(roots)) - 1) : -1;
}

// All lanes call this with the same packed word.  Reflection is returned in
// every lane, allowing a later component-source lane to reflect its own state
// without shared root/mate tables.
__device__ __forceinline__ PackedWord reflect_word_warp(PackedWord w, int* error) {
    const int root = root_position_warp(w);
    if (root < 0) {
        if ((threadIdx.x & 31) == 0) atomicCAS(error, 0, 551);
        return PackedWord{};
    }
    const std::uint32_t active = w.support & low_mask(w.len);
    const std::uint32_t rbits = active & ~w.left;
    const std::uint32_t matched_r = rbits & ~(std::uint32_t(1) << root);
    return PackedWord{
        reverse_bits_len_fast(active, w.len),
        reverse_bits_len_fast(matched_r, w.len),
        w.len
    };
}

// This variant is for the common component case where every lane owns a
// different state.  Each lane performs O(#L) arithmetic only through popc on
// 32-bit masks; there is no physical-position loop.  The root is obtained by a
// compact bit-parallel prefix test over occupied ordinals.
__device__ __forceinline__ PackedWord reflect_word_lane(PackedWord w, int* error) {
    // W<=28, so a bounded scalar prefix over occupied endpoints is at most 27
    // iterations.  Keep this correctness fallback separate from reflect_word_warp;
    // reverse fusion first uses the warp variant for the common label.  A future
    // source-batch transpose can remove this residual per-state root loop.
    const int root = root_position(w);
    if (root < 0) {
        atomicCAS(error, 0, 552);
        return PackedWord{};
    }
    const std::uint32_t active = w.support & low_mask(w.len);
    const std::uint32_t rbits = active & ~w.left;
    const std::uint32_t matched_r = rbits & ~(std::uint32_t(1) << root);
    return PackedWord{
        reverse_bits_len_fast(active, w.len),
        reverse_bits_len_fast(matched_r, w.len),
        w.len
    };
}

__device__ __forceinline__ PackedKey reflect_key_lane(
    PackedKey k,
    int W,
    int* error
) {
    const PackedWord w = reflect_word_lane(state_word(k, W), error);
    return make_state(k.type, w);
}

} // namespace oneesan::twocell::cuda_reflect
