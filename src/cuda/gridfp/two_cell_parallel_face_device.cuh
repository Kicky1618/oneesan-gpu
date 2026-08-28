#pragma once

#include "../../common/two_cell_component_device.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_face {

constexpr int kMaxComponent = 18;
constexpr int kMaxCandidates = 34;
constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ int prefix_height(PackedWord w, int boundary) {
    const std::uint32_t mask = low_mask(boundary);
    return 1 + 2 * __popc(w.left & mask) - __popc(w.support & mask);
}

// Parenthesis matching without a per-lane rightward scan.  For an L at p the
// mate is the first later R whose post-R height equals the pre-L height.  Give
// L lanes their pre-symbol height and R lanes their post-symbol height as the
// match-any key; then the first R bit to the right in the equal-key group is
// exactly the mate.  N/unused lanes receive unique non-height keys.
__device__ __forceinline__ int partner_match_any(PackedWord w, int lane) {
    const bool in_range = lane < w.len;
    const Symbol c = in_range ? symbol(w, lane) : TC_N;
    int key = 0x100 + lane;
    if (c == TC_L) key = prefix_height(w, lane);
    else if (c == TC_R) key = prefix_height(w, lane + 1);

    const unsigned same_height = __match_any_sync(kFullWarp, key);
    const unsigned r_mask = __ballot_sync(
        kFullWarp, in_range && c == TC_R);
    if (c != TC_L) return -1;
    const unsigned later_r = same_height & r_mask & ~low_mask(lane + 1);
    return later_r ? (__ffs(static_cast<int>(later_r)) - 1) : -1;
}

__device__ __forceinline__ bool key_in_list(
    const PackedKey* xs,
    int n,
    PackedKey x
) {
    for (int q = 0; q < n; ++q)
        if (equal(xs[q], x)) return true;
    return false;
}

// All lanes in one warp call this helper. `label` must be a deep component.
// Prefix heights are popcount expressions, face boundaries use ballots, and all
// L/R matching partners are resolved by one match-any grouping.  There is no
// O(W) partner loop.  The marked-face construction itself guarantees Motzkin
// validity of the generated full/source words; exhaustive CPU enumeration
// through W=14 checks this, so the hot path does not rescan each candidate with
// valid_word().  Only the final <=34-candidate unique compaction is scalar.
__device__ __forceinline__ void deep_component_sources(
    PackedWord label,
    int W,
    int i,
    PackedKey* out,
    int* out_size,
    PackedKey* candidate,
    std::uint8_t* candidate_valid,
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
        for (int q = 0; q < kMaxCandidates; ++q) candidate_valid[q] = 0;
    }
    __syncwarp();

    PackedWord central = insert_symbol(collapsed, i, TC_N);
    central = insert_symbol(central, i, TC_N);
    const int j = i + 1;
    PackedWord z = insert_symbol(central, j + 1, TC_N);

    // The no-cut predecessor is the fourth canonical source.
    if (lane == 0) {
        PackedKey sparse{};
        if (!inverse_E(z, i, sparse)) {
            atomicCAS(error, 0, 212);
        } else {
            out[(*out_size)++] = sparse;
        }
    }

    const int boundary = lane <= W ? lane : W;
    const int h = lane <= W ? prefix_height(z, boundary) : 0x3fffffff;
    const int level = prefix_height(z, j);
    const unsigned left_bad = __ballot_sync(
        kFullWarp, lane < j && h < level);
    const unsigned right_bad = __ballot_sync(
        kFullWarp, lane >= j + 3 && lane <= W && h < level);
    const int face_left = left_bad ? (31 - __clz(left_bad)) + 1 : 0;
    const int face_right = right_bad
        ? (__ffs(static_cast<int>(right_bad)) - 1) - 1
        : W;

    // Every lane participates so __match_any_sync sees the complete warp.
    const int partner_q = partner_match_any(z, lane);

    PackedKey mine{};
    bool valid = false;
    if (lane < W && symbol(z, lane) == TC_L &&
        lane >= face_left && prefix_height(z, lane) == level) {
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
    candidate[lane] = mine;
    candidate_valid[lane] = static_cast<std::uint8_t>(valid);

    // The enclosing strand immediately outside the marked face is a separate
    // inverse-R case.  Its partner has already been computed by its own lane;
    // lane zero obtains it with one shuffle instead of a second scan.
    const int enclosing_lane = face_left > 0 ? face_left - 1 : 0;
    const int enclosing_partner = __shfl_sync(
        kFullWarp, partner_q, enclosing_lane);
    if (lane == 0) {
        PackedKey extra{};
        bool extra_valid = false;
        if (face_left > 0) {
            const int p = face_left - 1;
            if (symbol(z, p) == TC_L &&
                enclosing_partner == face_right && p < j && face_right > j + 1) {
                PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
                if (inverse_E(w, i, extra)) extra_valid = true;
            }
        }
        candidate[32] = extra;
        candidate_valid[32] = static_cast<std::uint8_t>(extra_valid);
    }

    const unsigned root_mask = __ballot_sync(
        kFullWarp,
        lane < W && symbol(z, lane) == TC_R && prefix_height(z, lane + 1) == 0);
    if (lane == 0) {
        const int root = root_mask ? (__ffs(static_cast<int>(root_mask)) - 1) : -1;
        PackedKey extra{};
        bool extra_valid = false;
        if (root >= 0 &&
            (level == 0 || (face_left == 0 && face_right < W && root == face_right))) {
            PackedWord w = z;
            if (root < j) {
                w = set_symbol(w, root, TC_L);
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_R);
                if (inverse_E(w, i, extra)) extra_valid = true;
            } else if (root > j + 1) {
                w = set_symbol(set_symbol(w, j, TC_R), j + 1, TC_L);
                if (inverse_E(w, i, extra)) extra_valid = true;
            }
        }
        candidate[33] = extra;
        candidate_valid[33] = static_cast<std::uint8_t>(extra_valid);
    }
    __syncwarp();

    // Keep the legacy counter field for A/B probes.  Zero now means the
    // rightward partner-search loop has been eliminated, not that no matching
    // work occurred.
    if (lane == 0) *max_partner_rounds = 0;

    if (lane == 0) {
        int n = *out_size;
        for (int q = 0; q < kMaxCandidates; ++q) {
            if (!candidate_valid[q]) continue;
            const PackedKey k = candidate[q];
            if (key_in_list(out, n, k)) continue;
            if (n >= kMaxComponent) {
                atomicCAS(error, 0, 213);
                break;
            }
            out[n++] = k;
        }
        *out_size = n;
    }
    __syncwarp();
}

} // namespace oneesan::twocell::cuda_face
