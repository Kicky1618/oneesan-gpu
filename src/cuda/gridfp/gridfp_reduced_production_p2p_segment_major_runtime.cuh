#pragma once

#include "gridfp_reduced_production_runtime_subwarp.cuh"

namespace oneesan::gridfp::reducedprod {

static constexpr int RP_P2P_MAJOR_PC_CLASSES = 14;

struct P2PMajorGroupMeta {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    Rank64 scratch_base = 0;
    Rank64 pc = 0;
};
static_assert(sizeof(P2PMajorGroupMeta) == 24);

__device__ __forceinline__ Rank64 p2p_major_unpack_local(
    std::uint32_t low, std::uint8_t high
) {
    return Rank64(low >> 3) | (Rank64(high & 0x7fu) << 29);
}

__device__ __forceinline__ int p2p_major_find_class(
    Rank64 index,
    const P2PMajorGroupMeta* group
) {
#pragma unroll
    for (int cls = 0; cls < RP_P2P_MAJOR_PC_CLASSES; ++cls)
        if (index < group[cls].end) return cls;
    return -1;
}

// All records in a local-only cycle belong to this GPU. One warp loads the
// compact 36-bit local bases once, then rotates every primitive lane in place.
__global__ void p2p_major_local_cycle_kernel(
    std::uint32_t* __restrict__ state,
    const std::uint32_t* __restrict__ run_begin,
    const std::uint32_t* __restrict__ local_low,
    const std::uint8_t* __restrict__ local_high,
    const P2PMajorGroupMeta* __restrict__ group,
    Rank64 cycles,
    int* error
) {
    __shared__ Rank64 sh_base[RP_RUNTIME_WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[RP_RUNTIME_WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[RP_RUNTIME_WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * RP_RUNTIME_WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * RP_RUNTIME_WARPS_PER_BLOCK;
    for (Rank64 ci = first; ci < cycles; ci += stride) {
        if (lane == 0) {
            sh_len[warp] = 0;
            const int cls = p2p_major_find_class(ci, group);
            const std::uint32_t begin = run_begin[ci];
            const std::uint32_t end = run_begin[ci + 1];
            if (cls < 0 || end <= begin || end - begin > RP_MAX_W) {
                atomicCAS(error, 0, 471);
            } else {
                const int len = int(end - begin);
                for (int h = 0; h < len; ++h)
                    sh_base[warp][h] = p2p_major_unpack_local(
                        local_low[begin + h], local_high[begin + h]);
                sh_len[warp] = len;
                sh_pc[warp] = group[cls].pc;
            }
        }
        __syncwarp();

        const int len = sh_len[warp];
        if (len > 1) {
            const Rank64 pc = sh_pc[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = state[sh_base[warp][len - 1] + i];
                for (int h = len - 1; h > 0; --h)
                    state[sh_base[warp][h] + i] = state[sh_base[warp][h - 1] + i];
                state[sh_base[warp][0] + i] = temp;
            }
        }
        __syncwarp();
    }
}

// Phase 1 of a network batch. Each owner-boundary predecessor is read exactly
// once across P2P and captured in destination-local scratch before any run in
// this batch is modified.
__global__ void p2p_major_gather_kernel(
    std::uint32_t** __restrict__ peer_state,
    std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ source_low,
    const std::uint8_t* __restrict__ source_high,
    const P2PMajorGroupMeta* __restrict__ group,
    Rank64 segments,
    int ngpu,
    int gpu_id,
    int* error
) {
    __shared__ Rank64 sh_source[RP_RUNTIME_WARPS_PER_BLOCK];
    __shared__ Rank64 sh_scratch[RP_RUNTIME_WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[RP_RUNTIME_WARPS_PER_BLOCK];
    __shared__ int sh_owner[RP_RUNTIME_WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * RP_RUNTIME_WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * RP_RUNTIME_WARPS_PER_BLOCK;
    for (Rank64 si = first; si < segments; si += stride) {
        if (lane == 0) {
            sh_pc[warp] = 0;
            const int cls = p2p_major_find_class(si, group);
            if (cls < 0) {
                atomicCAS(error, 0, 472);
            } else {
                const P2PMajorGroupMeta gm = group[cls];
                const std::uint32_t low = source_low[si];
                const int owner = int(low & 7u);
                if (owner < 0 || owner >= ngpu || owner == gpu_id) {
                    atomicCAS(error, 0, 473);
                } else {
                    sh_owner[warp] = owner;
                    sh_source[warp] = p2p_major_unpack_local(low, source_high[si]);
                    sh_pc[warp] = gm.pc;
                    sh_scratch[warp] = gm.scratch_base + (si - gm.begin) * gm.pc;
                }
            }
        }
        __syncwarp();

        const Rank64 pc = sh_pc[warp];
        if (pc) {
            const std::uint32_t* src = peer_state[sh_owner[warp]] + sh_source[warp];
            std::uint32_t* dst = scratch + sh_scratch[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32)
                dst[i] = src[i];
        }
        __syncwarp();
    }
}

// Phase 2 after a global device barrier. All remaining movement is local to the
// destination owner. The backwards traversal preserves source values in place;
// the first run receives the predecessor captured by phase 1.
__global__ void p2p_major_rotate_kernel(
    std::uint32_t* __restrict__ state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ run_begin,
    const std::uint32_t* __restrict__ local_low,
    const std::uint8_t* __restrict__ local_high,
    const P2PMajorGroupMeta* __restrict__ group,
    Rank64 segments,
    int* error
) {
    __shared__ Rank64 sh_base[RP_RUNTIME_WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[RP_RUNTIME_WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[RP_RUNTIME_WARPS_PER_BLOCK];
    __shared__ Rank64 sh_scratch[RP_RUNTIME_WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * RP_RUNTIME_WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * RP_RUNTIME_WARPS_PER_BLOCK;
    for (Rank64 si = first; si < segments; si += stride) {
        if (lane == 0) {
            sh_len[warp] = 0;
            const int cls = p2p_major_find_class(si, group);
            const std::uint32_t begin = run_begin[si];
            const std::uint32_t end = run_begin[si + 1];
            if (cls < 0 || end <= begin || end - begin > RP_MAX_W) {
                atomicCAS(error, 0, 474);
            } else {
                const P2PMajorGroupMeta gm = group[cls];
                const int len = int(end - begin);
                for (int h = 0; h < len; ++h)
                    sh_base[warp][h] = p2p_major_unpack_local(
                        local_low[begin + h], local_high[begin + h]);
                sh_len[warp] = len;
                sh_pc[warp] = gm.pc;
                sh_scratch[warp] = gm.scratch_base + (si - gm.begin) * gm.pc;
            }
        }
        __syncwarp();

        const int len = sh_len[warp];
        if (len > 0) {
            const Rank64 pc = sh_pc[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                for (int h = len - 1; h > 0; --h)
                    state[sh_base[warp][h] + i] = state[sh_base[warp][h - 1] + i];
                state[sh_base[warp][0] + i] = scratch[sh_scratch[warp] + i];
            }
        }
        __syncwarp();
    }
}

} // namespace oneesan::gridfp::reducedprod
