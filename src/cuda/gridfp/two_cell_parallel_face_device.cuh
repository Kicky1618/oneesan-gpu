#pragma once

#include "../../common/two_cell_component_device.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_face {

constexpr int kMaxComponent = 18;
constexpr int kMaxCandidates = 34;

__device__ __forceinline__ int prefix_height(PackedWord w, int boundary) {
    const std::uint32_t mask = low_mask(boundary);
    return 1 + 2 * __popc(w.left & mask) - __popc(w.support & mask);
}

__device__ __forceinline__ int partner_first_return(
    PackedWord w,
    int p,
    int* rounds
) {
    const int level = prefix_height(w, p);
    int n = 0;
    for (int q = p + 1; q < w.len; ++q) {
        ++n;
        if (symbol(w, q) == TC_R && prefix_height(w, q + 1) == level) {
            *rounds = n;
            return q;
        }
    }
    *rounds = n;
    return -1;
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
// Expensive prefix-height and matching-partner work is distributed over the W
// physical-position lanes. Only the final <=34-candidate unique compaction is
// scalar, and that compaction performs no partner/height rescans.
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

    if (lane == 0) {
        PackedKey sparse{};
        if (!inverse_E(z, i, sparse) || !in_source_layout(sparse, W, i)) {
            atomicCAS(error, 0, 212);
        } else {
            out[(*out_size)++] = sparse;
        }
    }

    const int boundary = lane <= W ? lane : W;
    const int h = lane <= W ? prefix_height(z, boundary) : 0x3fffffff;
    const int level = prefix_height(z, j);
    const unsigned left_bad = __ballot_sync(
        0xffffffffu, lane < j && h < level);
    const unsigned right_bad = __ballot_sync(
        0xffffffffu, lane >= j + 3 && lane <= W && h < level);
    const int face_left = left_bad ? (31 - __clz(left_bad)) + 1 : 0;
    const int face_right = right_bad
        ? (__ffs(static_cast<int>(right_bad)) - 1) - 1
        : W;

    PackedKey mine{};
    bool valid = false;
    int rounds = 0;
    if (lane < W && symbol(z, lane) == TC_L &&
        lane >= face_left && prefix_height(z, lane) == level) {
        const int q = partner_first_return(z, lane, &rounds);
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
            if (full_valid && valid_word(w) && inverse_E(w, i, mine) &&
                in_source_layout(mine, W, i))
                valid = true;
        }
    }
    candidate[lane] = mine;
    candidate_valid[lane] = static_cast<std::uint8_t>(valid);

    if (lane == 0) {
        PackedKey extra{};
        bool extra_valid = false;
        if (face_left > 0) {
            const int p = face_left - 1;
            if (symbol(z, p) == TC_L) {
                int qrounds = 0;
                const int q = partner_first_return(z, p, &qrounds);
                if (qrounds > *max_partner_rounds) *max_partner_rounds = qrounds;
                if (q == face_right && p < j && q > j + 1) {
                    PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
                    if (valid_word(w) && inverse_E(w, i, extra) &&
                        in_source_layout(extra, W, i))
                        extra_valid = true;
                }
            }
        }
        candidate[32] = extra;
        candidate_valid[32] = static_cast<std::uint8_t>(extra_valid);
    }

    const unsigned root_mask = __ballot_sync(
        0xffffffffu,
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
                if (valid_word(w) && inverse_E(w, i, extra) &&
                    in_source_layout(extra, W, i))
                    extra_valid = true;
            } else if (root > j + 1) {
                w = set_symbol(set_symbol(w, j, TC_R), j + 1, TC_L);
                if (valid_word(w) && inverse_E(w, i, extra) &&
                    in_source_layout(extra, W, i))
                    extra_valid = true;
            }
        }
        candidate[33] = extra;
        candidate_valid[33] = static_cast<std::uint8_t>(extra_valid);
    }

    int warp_max = rounds;
    for (int offset = 16; offset; offset >>= 1) {
        const int other = __shfl_down_sync(0xffffffffu, warp_max, offset);
        if (other > warp_max) warp_max = other;
    }
    if (lane == 0 && warp_max > *max_partner_rounds)
        *max_partner_rounds = warp_max;
    __syncwarp();

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
