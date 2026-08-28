#pragma once

#include "../../common/two_cell_component_device.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_face {

constexpr int kMaxComponent = 18;
constexpr int kMaxCandidates = 34; // retained for compatibility probes
constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ int prefix_height(PackedWord w, int boundary) {
    const std::uint32_t mask = low_mask(boundary);
    return 1 + 2 * __popc(w.left & mask) - __popc(w.support & mask);
}

// All lanes already know their pre/post symbol heights.  A match-any group keyed
// by pre-L / post-R height gives every L lane all possible same-level R lanes;
// its first later R is the parenthesis mate.
__device__ __forceinline__ int partner_match_any_cached(
    Symbol c,
    int h_before,
    int h_after,
    int lane,
    int len
) {
    int key = 0x100 + lane;
    if (c == TC_L) key = h_before;
    else if (c == TC_R) key = h_after;
    const unsigned same_height = __match_any_sync(kFullWarp, key);
    const unsigned r_mask = __ballot_sync(
        kFullWarp, lane < len && c == TC_R);
    if (c != TC_L) return -1;
    const unsigned later_r = same_height & r_mask & ~low_mask(lane + 1);
    return later_r ? (__ffs(static_cast<int>(later_r)) - 1) : -1;
}

// Canonical deep source reconstruction.  The output order is exactly the
// serial direct_component_sources() order:
//   0..2 fixed label coordinates,
//   3    no-cut central predecessor,
//   4..  exposed L-strand cuts in physical-L-position order,
//         then enclosing strand, then root strand.
//
// Exhaustive CPU enumeration through W=14 finds that all face-generated
// candidates are already valid and mutually distinct, including against the
// four fixed sources.  Therefore the hot path needs neither valid_word rescans
// nor scalar duplicate elimination.  Each physical lane computes one prefix
// height once; h_after, face tests, matching, and root detection all reuse it.
__device__ __forceinline__ void deep_component_sources_compact(
    PackedWord label,
    int W,
    int i,
    PackedKey* out,
    int* out_size,
    int* max_partner_rounds,
    int* error
) {
    const int lane = threadIdx.x & 31;
    PackedWord collapsed{};
    if (!deep_collapse(label, i, collapsed)) {
        if (lane == 0) atomicCAS(error, 0, 211);
        return;
    }

    if (lane == 0) {
        out[0] = make_state(1, label);
        out[1] = make_state(0, insert_symbol(label, i, TC_N));
        out[2] = make_state(0, insert_symbol(label, i + 1, TC_N));
        *out_size = 3;
        *max_partner_rounds = 0;
    }
    __syncwarp();

    PackedWord central = insert_symbol(collapsed, i, TC_N);
    central = insert_symbol(central, i, TC_N);
    const int j = i + 1;
    PackedWord z = insert_symbol(central, j + 1, TC_N);

    if (lane == 0) {
        PackedKey sparse{};
        if (!inverse_E(z, i, sparse)) {
            atomicCAS(error, 0, 212);
        } else {
            out[3] = sparse;
            *out_size = 4;
        }
    }
    __syncwarp();

    const bool in_range = lane < W;
    const Symbol c = in_range ? symbol(z, lane) : TC_N;
    const int h_before = lane <= W ? prefix_height(z, lane) : 0x3fffffff;
    const int h_after = h_before +
        (c == TC_L ? 1 : (c == TC_R ? -1 : 0));
    int level = lane == 0 ? prefix_height(z, j) : 0;
    level = __shfl_sync(kFullWarp, level, 0);

    const unsigned left_bad = __ballot_sync(
        kFullWarp, lane < j && h_before < level);
    const unsigned right_bad = __ballot_sync(
        kFullWarp, lane >= j + 3 && lane <= W && h_before < level);
    const int face_left = left_bad ? (31 - __clz(left_bad)) + 1 : 0;
    const int face_right = right_bad
        ? (__ffs(static_cast<int>(right_bad)) - 1) - 1
        : W;

    const int partner_q = partner_match_any_cached(
        c, h_before, h_after, lane, W);

    PackedKey mine{};
    bool valid = false;
    if (in_range && c == TC_L && lane >= face_left && h_before == level) {
        const int q = partner_q;
        if (q >= 0 && q < face_right) {
            PackedWord w = z;
            bool full_valid = true;
            if (q < j) {
                w = set_symbol(w, q, TC_L);
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_R);
            } else if (lane > j + 1) {
                w = set_symbol(w, lane, TC_R);
                w = set_symbol(w, j, TC_L);
                w = set_symbol(w, j + 1, TC_L);
            } else if (lane < j && q > j + 1) {
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_L);
            } else {
                full_valid = false;
            }
            if (full_valid && inverse_E(w, i, mine)) valid = true;
        }
    }

    const unsigned valid_mask = __ballot_sync(kFullWarp, valid);
    if (valid) {
        const int rank = __popc(valid_mask & low_mask(lane));
        out[4 + rank] = mine;
    }
    const int lane_count = __popc(valid_mask);

    const int enclosing_lane = face_left > 0 ? face_left - 1 : 0;
    const int enclosing_partner = __shfl_sync(
        kFullWarp, partner_q, enclosing_lane);

    PackedKey enclosing{};
    bool enclosing_valid = false;
    if (lane == 0 && face_left > 0) {
        const int p = face_left - 1;
        if (symbol(z, p) == TC_L &&
            enclosing_partner == face_right && p < j && face_right > j + 1) {
            PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
            enclosing_valid = inverse_E(w, i, enclosing);
        }
    }

    const unsigned root_mask = __ballot_sync(
        kFullWarp, in_range && c == TC_R && h_after == 0);
    PackedKey root_key{};
    bool root_valid = false;
    if (lane == 0) {
        const int root = root_mask ? (__ffs(static_cast<int>(root_mask)) - 1) : -1;
        if (root >= 0 &&
            (level == 0 || (face_left == 0 && face_right < W && root == face_right))) {
            PackedWord w = z;
            if (root < j) {
                w = set_symbol(w, root, TC_L);
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_R);
                root_valid = inverse_E(w, i, root_key);
            } else if (root > j + 1) {
                w = set_symbol(set_symbol(w, j, TC_R), j + 1, TC_L);
                root_valid = inverse_E(w, i, root_key);
            }
        }

        int n = 4 + lane_count;
        if (enclosing_valid) out[n++] = enclosing;
        if (root_valid) out[n++] = root_key;
        if (n > kMaxComponent) {
            atomicCAS(error, 0, 213);
        } else {
            *out_size = n;
        }
        *max_partner_rounds = 0;
    }
    __syncwarp();
}

// Compatibility wrapper for probes written before ballot compaction.  The
// candidate scratch arrays are no longer used.
__device__ __forceinline__ void deep_component_sources(
    PackedWord label,
    int W,
    int i,
    PackedKey* out,
    int* out_size,
    PackedKey*,
    std::uint8_t*,
    int* max_partner_rounds,
    int* error
) {
    deep_component_sources_compact(
        label, W, i, out, out_size, max_partner_rounds, error);
}

} // namespace oneesan::twocell::cuda_face
