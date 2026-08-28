#pragma once

#include "../../common/two_cell_turn_closed_device.cuh"
#include "two_cell_parallel_face_device.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_turn {

constexpr unsigned kTurnFullWarp = 0xffffffffu;

// All lanes call this helper. The output order exactly matches
// right_turn_closed_block(): alpha, beta, then passive states in increasing
// physical R position.
__device__ __forceinline__ void right_turn_closed_block_warp(
    PackedWord label,
    int W,
    PackedKey* out,
    int* out_size,
    int* singular,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int len = W - 2;
    if (label.len != len) {
        if (lane == 0) atomicCAS(error, 0, 501);
        return;
    }

    const Symbol last = symbol(label, len - 1);
    if (last == TC_R) {
        if (lane == 0) {
            *singular = 1;
            *out_size = 3;
            out[0] = make_state(0, insert_symbol(label, len - 1, TC_N));
            out[1] = make_state(1, label);
            out[2] = make_state(0, insert_symbol(label, len, TC_N));
        }
        __syncwarp();
        return;
    }
    if (last != TC_N) {
        if (lane == 0) atomicCAS(error, 0, 502);
        __syncwarp();
        return;
    }

    const PackedWord v = remove_symbol(label, len - 1);
    if (lane == 0) {
        *singular = 0;
        PackedWord alpha = insert_symbol(v, v.len, TC_L);
        alpha = insert_symbol(alpha, alpha.len, TC_R);
        PackedWord beta = insert_symbol(v, v.len, TC_N);
        beta = insert_symbol(beta, beta.len, TC_N);
        out[0] = make_state(0, alpha);
        out[1] = make_state(0, beta);
        *out_size = 2;
    }
    __syncwarp();

    const bool passive = lane < v.len &&
        symbol(v, lane) == TC_R &&
        oneesan::twocell::cuda_face::prefix_height(v, lane + 1) == 0;
    const unsigned mask = __ballot_sync(kTurnFullWarp, passive);
    if (passive) {
        const int rank = __popc(mask & low_mask(lane));
        PackedWord z = set_symbol(v, lane, TC_L);
        z = insert_symbol(z, z.len, TC_R);
        z = insert_symbol(z, z.len, TC_R);
        out[2 + rank] = make_state(0, z);
    }
    if (lane == 0) {
        const int n = 2 + __popc(mask);
        if (n < 3 || n > kMaxTurnStates) {
            atomicCAS(error, 0, 503);
        } else {
            *out_size = n;
        }
    }
    __syncwarp();
}

} // namespace oneesan::twocell::cuda_turn
