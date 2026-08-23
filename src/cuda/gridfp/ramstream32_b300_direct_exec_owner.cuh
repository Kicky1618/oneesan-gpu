#pragma once

#include "ramstream32_b300_direct_maskshard.cuh"

#include <array>
#include <cstdint>

// HIGH direct execution does not have to run on the source-state owner.
// Orbit records own all writes to their 3-state orbit, so choose the GPU that
// minimizes peer operands.  Closure sources are read-only in the closure pass,
// so execute every closure on its destination owner: the atomic update is then
// always device-local and the only P2P operation is a source read.
struct B300DirectExecOwnerStats {
    uint64_t orbit_source_owner_peer_bytes = 0;
    uint64_t orbit_exec_owner_peer_bytes = 0;
    uint64_t closure_peer_read_bytes = 0;
    uint64_t closure_remote_system_atomics = 0;
    std::array<uint64_t, MAXGPU> orbit_work{};
    std::array<uint64_t, MAXGPU> closure_work{};
};

static B300DirectSparsePartitionHost b300_direct_partition_high_by_exec_owner(
    const B300SparseActionsHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    const B300DirectMaskShardHost& shard,
    B300DirectExecOwnerStats* out_stats = nullptr
) {
    B300DirectSparsePartitionHost z;
    z.ngpu = shard.ngpu;
    for (int g = 0; g < shard.ngpu; ++g) {
        z.high_orbit_off[g].resize(HIGH_LUT_K + 1);
        z.high_closure_off[g].resize(HIGH_LUT_K + 1);
    }
    auto owner_of = [&](const StorageBlock& b, uint32_t hr) -> int {
        return shard.high_owner[storage.high_all_off[b.he] + hr];
    };

    B300DirectExecOwnerStats st{};
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (int g = 0; g < shard.ngpu; ++g) {
            z.high_orbit_off[g][pi] = uint32_t(z.high_orbit[g].size());
            z.high_closure_off[g][pi] = uint32_t(z.high_closure[g].size());
        }

        for (uint32_t q = sparse.high_orbit_off[pi]; q < sparse.high_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.high_orbit[q];
            const auto& x = layout.main_blocks[b300_sparse_sblock(op)];
            const auto& j = layout.main_blocks[b300_sparse_jblock(op)];
            const auto& d = layout.block_blocks[b300_sparse_dblock(op)];
            int os = owner_of(x, b300_sparse_src(op));
            int oj = owner_of(j, b300_sparse_jrank(op));
            int od = owner_of(d, b300_sparse_drank(op));
            uint32_t kind = b300_sparse_kind(op);

            // Bytes touched per LOW column on a peer.  Source and dropped block
            // are read/write.  NN also read/writes the partner; NR/NL only read
            // the partner because its identity value remains in place.
            uint64_t ws = 2 * sizeof(Count);
            uint64_t wj = (kind == HIGH_ORBIT_NN ? 2 : 1) * sizeof(Count);
            uint64_t wd = 2 * sizeof(Count);
            auto peer_cost = [&](int g) -> uint64_t {
                return (os == g ? 0 : ws) + (oj == g ? 0 : wj) + (od == g ? 0 : wd);
            };
            st.orbit_source_owner_peer_bytes += peer_cost(os) * uint64_t(x.cols);

            int best = os;
            uint64_t best_cost = peer_cost(best);
            int candidates[3] = {os, oj, od};
            for (int k = 0; k < 3; ++k) {
                int g = candidates[k];
                uint64_t c = peer_cost(g);
                if (c < best_cost || (c == best_cost && st.orbit_work[g] < st.orbit_work[best])) {
                    best = g; best_cost = c;
                }
            }
            st.orbit_exec_owner_peer_bytes += best_cost * uint64_t(x.cols);
            st.orbit_work[best] += x.cols;
            z.high_orbit[best].push_back(op);
        }

        for (uint32_t q = sparse.high_closure_off[pi]; q < sparse.high_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.high_closure[q];
            const auto& x = layout.main_blocks[b300_sparse_closure_sblock(op)];
            uint32_t desc = b300_sparse_closure_desc(op);
            const auto& d = layout.block_blocks[highdesc_block(desc)];
            int os = owner_of(x, b300_sparse_closure_src(op));
            int od = owner_of(d, highdesc_rank(desc));

            // Destination-owner execution converts a peer system-scope atomic
            // into one peer source load plus a normal local atomic.
            if (os != od) st.closure_peer_read_bytes += uint64_t(sizeof(Count)) * x.cols;
            st.closure_work[od] += x.cols;
            z.high_closure[od].push_back(op);
        }
    }
    for (int g = 0; g < shard.ngpu; ++g) {
        z.high_orbit_off[g][HIGH_LUT_K] = uint32_t(z.high_orbit[g].size());
        z.high_closure_off[g][HIGH_LUT_K] = uint32_t(z.high_closure[g].size());
    }
    // By construction every closure is executed by the owner of its blocked
    // destination, so there are no cross-GPU atomic RMWs.
    st.closure_remote_system_atomics = 0;
    if (out_stats) *out_stats = st;
    return z;
}
