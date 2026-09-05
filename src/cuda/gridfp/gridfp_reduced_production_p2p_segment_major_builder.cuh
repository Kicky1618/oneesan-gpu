#pragma once

#include "gridfp_reduced_production_p2p_segment_major_runtime.cuh"
#include "gridfp_reduced_production_p2p_ownerfirst_device.cuh"
#include "gridfp_reduced_production_equal_tile_cycle_device.cuh"

namespace oneesan::gridfp::reducedprod {

static constexpr int RP_P2P_MAJOR_MAX_GPU = 8;
static constexpr unsigned long long RP_P2P_MAJOR_PAIR_ITEM_INC = 1ULL << 32;
static constexpr unsigned long long RP_P2P_MAJOR_PAIR_RUN_MASK = 0xffffffffULL;

__device__ __forceinline__ std::uint32_t p2p_major_batch_hash_device(
    std::uint32_t support,
    bool blocked,
    bool reverse,
    std::uint32_t salt = 0
) {
    std::uint32_t x = support ^ 0x9e3779b9u ^ salt;
    x ^= blocked ? 0x27d4eb2du : 0u;
    x ^= reverse ? 0x165667b1u : 0u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ __forceinline__ int p2p_major_group_index_device(
    int gpu,
    int batch,
    int cls,
    int batches
) {
    return (gpu * batches + batch) * RP_P2P_MAJOR_PC_CLASSES + cls;
}

__device__ __forceinline__ void p2p_major_pack_run39_device(
    int owner,
    Rank64 local,
    std::uint32_t& low,
    std::uint8_t& high
) {
    low = std::uint32_t(owner) |
          (std::uint32_t(local & ((Rank64(1) << 29) - 1)) << 3);
    high = static_cast<std::uint8_t>((local >> 29) & 0x7fu);
}

__global__ void p2p_major_count_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    std::uint32_t batch_salt,
    unsigned long long* __restrict__ network_segments,
    unsigned long long* __restrict__ network_runs,
    unsigned long long* __restrict__ local_cycles,
    unsigned long long* __restrict__ local_runs,
    unsigned long long* __restrict__ network_states,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed root = seeds[ri];
            const bool blocked = root.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                root.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 501);
                continue;
            }
            if (len <= 1) continue;

            const int occupied = __popc(root.support);
            if (!(occupied & 1)) {
                atomicCAS(error, 0, 502);
                continue;
            }
            const int cls = (occupied + 1) / 2 - 1;
            if (cls < 0 || cls >= RP_P2P_MAJOR_PC_CLASSES) {
                atomicCAS(error, 0, 503);
                continue;
            }
            const Rank64 pc = RP_PRIMITIVE[occupied][1];

            int owner[RP_MAX_W]{};
            std::uint32_t cur = root.support;
            bool ok = true;
            for (int h = 0; h < len; ++h) {
                owner[h] = p2p_support_owner_device(
                    cur, W, old_start, K, reverse, ngpu);
                if (owner[h] < 0 || owner[h] >= ngpu) {
                    ok = false;
                    break;
                }
                cur = shift_next_support_device(cur, blocked, W, q, K, K, reverse);
            }
            if (!ok) {
                atomicCAS(error, 0, 504);
                continue;
            }

            int boundaries = 0;
            for (int h = 0; h < len; ++h)
                boundaries += owner[h] != owner[(h + len - 1) % len];
            if (!boundaries) {
                const int idx = owner[0] * RP_P2P_MAJOR_PC_CLASSES + cls;
                atomicAdd(local_cycles + idx, 1ULL);
                atomicAdd(local_runs + idx, static_cast<unsigned long long>(len));
                continue;
            }

            const int batch = int(p2p_major_batch_hash_device(
                root.support, blocked, reverse, batch_salt) % std::uint32_t(batches));
            for (int start = 0; start < len; ++start) {
                const int pred = (start + len - 1) % len;
                if (owner[start] == owner[pred]) continue;
                const int g = owner[start];
                int seg_len = 1;
                while (seg_len < len && owner[(start + seg_len) % len] == g)
                    ++seg_len;
                if (seg_len >= len) {
                    atomicCAS(error, 0, 505);
                    break;
                }
                const int idx = p2p_major_group_index_device(g, batch, cls, batches);
                atomicAdd(network_segments + idx, 1ULL);
                atomicAdd(network_runs + idx, static_cast<unsigned long long>(seg_len));
                atomicAdd(network_states, static_cast<unsigned long long>(pc));
            }
        }
    }
}

__device__ __forceinline__ bool p2p_major_rank_cycle_device(
    std::uint32_t root_support,
    bool blocked,
    int len,
    int W,
    int q,
    int K,
    bool reverse,
    int old_start,
    int ngpu,
    const Rank64* owner_begin,
    int (&owner)[RP_MAX_W],
    Rank64 (&local)[RP_MAX_W]
) {
    std::uint32_t cur = root_support;
    for (int h = 0; h < len; ++h) {
        const DeviceKey key = equal_run_key0_device(cur, blocked, W, q, reverse);
        const GroupedDeviceRank gr = grouped_rank_device(
            key, W, q, reverse, old_start, K, ngpu, owner_begin);
        const int cheap = p2p_support_owner_device(
            cur, W, old_start, K, reverse, ngpu);
        if (gr.owner != cheap || gr.owner < 0 || gr.owner >= ngpu ||
            gr.local >= (Rank64(1) << 36))
            return false;
        owner[h] = gr.owner;
        local[h] = gr.local;
        cur = shift_next_support_device(cur, blocked, W, q, K, K, reverse);
    }
    return true;
}

__global__ void p2p_major_fill_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    std::uint32_t batch_salt,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ net_header_base,
    const Rank64* __restrict__ net_source_base,
    const Rank64* __restrict__ net_run_base,
    const Rank64* __restrict__ net_group_segment_begin,
    const Rank64* __restrict__ net_group_run_begin,
    const Rank64* __restrict__ local_header_base,
    const Rank64* __restrict__ local_run_base,
    const Rank64* __restrict__ local_group_cycle_begin,
    const Rank64* __restrict__ local_group_run_begin,
    unsigned long long* __restrict__ net_cursor,
    unsigned long long* __restrict__ local_cursor,
    std::uint32_t* __restrict__ net_run_begin,
    std::uint32_t* __restrict__ source_low,
    std::uint8_t* __restrict__ source_high,
    std::uint32_t* __restrict__ net_local_low,
    std::uint8_t* __restrict__ net_local_high,
    std::uint32_t* __restrict__ local_run_begin,
    std::uint32_t* __restrict__ local_low,
    std::uint8_t* __restrict__ local_high,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed root = seeds[ri];
            const bool blocked = root.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                root.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 511);
                continue;
            }
            if (len <= 1) continue;

            const int occupied = __popc(root.support);
            if (!(occupied & 1)) {
                atomicCAS(error, 0, 512);
                continue;
            }
            const int cls = (occupied + 1) / 2 - 1;
            if (cls < 0 || cls >= RP_P2P_MAJOR_PC_CLASSES) {
                atomicCAS(error, 0, 513);
                continue;
            }

            int owner[RP_MAX_W]{};
            Rank64 local_rank[RP_MAX_W]{};
            if (!p2p_major_rank_cycle_device(
                    root.support, blocked, len, W, q, K, reverse, old_start,
                    ngpu, owner_begin, owner, local_rank)) {
                atomicCAS(error, 0, 514);
                continue;
            }

            int boundaries = 0;
            for (int h = 0; h < len; ++h)
                boundaries += owner[h] != owner[(h + len - 1) % len];
            if (!boundaries) {
                const int g = owner[0];
                const int idx = g * RP_P2P_MAJOR_PC_CLASSES + cls;
                const unsigned long long old = atomicAdd(
                    local_cursor + idx,
                    RP_P2P_MAJOR_PAIR_ITEM_INC | static_cast<unsigned long long>(len));
                const Rank64 item = old >> 32;
                const Rank64 run = old & RP_P2P_MAJOR_PAIR_RUN_MASK;
                if (run + Rank64(len) >= (Rank64(1) << 32)) {
                    atomicCAS(error, 0, 515);
                    continue;
                }
                local_run_begin[
                    local_header_base[g] + local_group_cycle_begin[idx] + item] =
                    static_cast<std::uint32_t>(local_group_run_begin[idx] + run);
                const Rank64 dst = local_run_base[g] + local_group_run_begin[idx] + run;
                for (int h = 0; h < len; ++h)
                    p2p_major_pack_run39_device(
                        owner[h], local_rank[h], local_low[dst + h], local_high[dst + h]);
                continue;
            }

            const int batch = int(p2p_major_batch_hash_device(
                root.support, blocked, reverse, batch_salt) % std::uint32_t(batches));
            for (int start = 0; start < len; ++start) {
                const int pred = (start + len - 1) % len;
                if (owner[start] == owner[pred]) continue;
                const int g = owner[start];
                int seg_len = 1;
                while (seg_len < len && owner[(start + seg_len) % len] == g)
                    ++seg_len;
                if (seg_len >= len) {
                    atomicCAS(error, 0, 516);
                    break;
                }
                const int idx = p2p_major_group_index_device(g, batch, cls, batches);
                const unsigned long long old = atomicAdd(
                    net_cursor + idx,
                    RP_P2P_MAJOR_PAIR_ITEM_INC |
                        static_cast<unsigned long long>(seg_len));
                const Rank64 item = old >> 32;
                const Rank64 run = old & RP_P2P_MAJOR_PAIR_RUN_MASK;
                if (run + Rank64(seg_len) >= (Rank64(1) << 32)) {
                    atomicCAS(error, 0, 517);
                    break;
                }
                const int gb = g * batches + batch;
                const Rank64 segment_local = net_group_segment_begin[idx] + item;
                const Rank64 run_local = net_group_run_begin[idx] + run;
                net_run_begin[net_header_base[gb] + segment_local] =
                    static_cast<std::uint32_t>(run_local);
                p2p_major_pack_run39_device(
                    owner[pred], local_rank[pred],
                    source_low[net_source_base[gb] + segment_local],
                    source_high[net_source_base[gb] + segment_local]);
                const Rank64 dst = net_run_base[gb] + run_local;
                for (int j = 0; j < seg_len; ++j) {
                    const int h = (start + j) % len;
                    p2p_major_pack_run39_device(
                        owner[h], local_rank[h],
                        net_local_low[dst + j], net_local_high[dst + j]);
                }
            }
        }
    }
}

} // namespace oneesan::gridfp::reducedprod
