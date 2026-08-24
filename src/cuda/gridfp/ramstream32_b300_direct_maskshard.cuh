#pragma once

#include "ramstream32_b300_direct_rowshard.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

// Preferred direct-HBM sharding: assign each complete HIGH occupancy class to
// one GPU.  LOW-window transitions, including boundary CROSS transitions,
// preserve HIGH occupancy and therefore never cross GPUs.
struct B300DirectMaskShardHost {
    int ngpu = 0;
    std::vector<uint8_t> mask_owner;
    std::vector<uint8_t> high_owner;
    std::vector<uint32_t> high_local;
    std::array<std::vector<uint32_t>, MAXGPU> owned_rows;
    std::array<std::array<uint32_t, MAXW + 2>, MAXGPU> owned_off{};
    std::array<Code, MAXGPU> main_count{};
    std::array<Code, MAXGPU> block_count{};
    std::array<std::array<Code, 64>, MAXGPU> main_off{};
    std::array<std::array<Code, 32>, MAXGPU> block_off{};
};

static B300DirectMaskShardHost build_b300_direct_mask_shards(
    const StorageFactorHost& storage, const StorageLayout& layout, int ngpu
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t NM = 1u << HIGH_LUT_K;
    if (ngpu < 1 || ngpu > MAXGPU) std::exit(460);

    std::vector<unsigned long long> weight(NM, 0);
    auto add_weight = [&](const StorageBlock& b) {
        if (!b.valid || !b.rows || !b.cols) return;
        for (uint32_t mask = 0; mask < NM; ++mask) {
            size_t ix = size_t(mask) * S + b.he;
            uint32_t a = G_FACTOR.high_mask_off[ix];
            uint32_t e = G_FACTOR.high_mask_off[ix + 1];
            weight[mask] += (unsigned long long)(e - a) * b.cols * sizeof(Count);
        }
    };
    for (const auto& b : layout.main_blocks) add_weight(b);
    for (const auto& b : layout.block_blocks) add_weight(b);

    std::vector<uint32_t> order(NM);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (weight[a] != weight[b]) return weight[a] > weight[b];
        return a < b;
    });

    B300DirectMaskShardHost z;
    z.ngpu = ngpu;
    z.mask_owner.assign(NM, 0);
    std::array<unsigned long long, MAXGPU> bytes{};
    for (uint32_t mask : order) {
        int g = 0;
        for (int q = 1; q < ngpu; ++q) if (bytes[q] < bytes[g]) g = q;
        z.mask_owner[mask] = uint8_t(g);
        bytes[g] += weight[mask];
    }

    z.high_owner.resize(storage.high_all_codes.size());
    z.high_local.resize(storage.high_all_codes.size());
    std::array<std::array<uint32_t, MAXW + 2>, MAXGPU> rows_per_h{};
    for (int h = 0; h <= MAXW; ++h) {
        std::array<uint32_t, MAXGPU> local{};
        uint32_t n = storage.high_all_off[h + 1] - storage.high_all_off[h];
        for (uint32_t hr = 0; hr < n; ++hr) {
            uint32_t ai = storage.high_all_off[h] + hr;
            uint32_t mask = seg_occ(storage.high_all_codes[ai], HIGH_LUT_K);
            int g = z.mask_owner[mask];
            z.high_owner[ai] = uint8_t(g);
            z.high_local[ai] = local[g]++;
        }
        for (int g = 0; g < ngpu; ++g) rows_per_h[g][h] = local[g];
    }

    for (int g = 0; g < ngpu; ++g) {
        for (int h = 0; h <= MAXW; ++h) {
            z.owned_off[g][h] = uint32_t(z.owned_rows[g].size());
            uint32_t n = storage.high_all_off[h + 1] - storage.high_all_off[h];
            for (uint32_t hr = 0; hr < n; ++hr) {
                uint32_t ai = storage.high_all_off[h] + hr;
                if (z.high_owner[ai] == g) z.owned_rows[g].push_back(hr);
            }
        }
        z.owned_off[g][MAXW + 1] = uint32_t(z.owned_rows[g].size());

        Code off = 0;
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            z.main_off[g][bid] = off;
            const auto& b = layout.main_blocks[bid];
            off += Code(rows_per_h[g][b.he]) * b.cols;
        }
        z.main_count[g] = off;
        off = 0;
        for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
            z.block_off[g][bid] = off;
            const auto& b = layout.block_blocks[bid];
            off += Code(rows_per_h[g][b.he]) * b.cols;
        }
        z.block_count[g] = off;
    }
    return z;
}

static B300DirectSparsePartitionHost b300_direct_partition_high_by_mask(
    const B300SparseActionsHost& sparse, const StorageFactorHost& storage,
    const StorageLayout& layout, const B300DirectMaskShardHost& shard
) {
    B300DirectSparsePartitionHost z;
    z.ngpu = shard.ngpu;
    for (int g = 0; g < shard.ngpu; ++g) {
        z.high_orbit_off[g].resize(HIGH_LUT_K + 1);
        z.high_closure_off[g].resize(HIGH_LUT_K + 1);
    }
    auto owner_of = [&](const StorageBlock& b, uint32_t hr) -> int {
        uint32_t ai = storage.high_all_off[b.he] + hr;
        return shard.high_owner[ai];
    };
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (int g = 0; g < shard.ngpu; ++g) {
            z.high_orbit_off[g][pi] = uint32_t(z.high_orbit[g].size());
            z.high_closure_off[g][pi] = uint32_t(z.high_closure[g].size());
        }
        for (uint32_t q = sparse.high_orbit_off[pi]; q < sparse.high_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.high_orbit[q];
            const auto& b = layout.main_blocks[b300_sparse_sblock(op)];
            int g = owner_of(b, b300_sparse_src(op));
            z.high_orbit[g].push_back(op);
        }
        for (uint32_t q = sparse.high_closure_off[pi]; q < sparse.high_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.high_closure[q];
            const auto& b = layout.main_blocks[b300_sparse_closure_sblock(op)];
            int g = owner_of(b, b300_sparse_closure_src(op));
            z.high_closure[g].push_back(op);
        }
    }
    for (int g = 0; g < shard.ngpu; ++g) {
        z.high_orbit_off[g][HIGH_LUT_K] = uint32_t(z.high_orbit[g].size());
        z.high_closure_off[g][HIGH_LUT_K] = uint32_t(z.high_closure[g].size());
    }
    return z;
}

__constant__ uint8_t* D_DM_HIGH_OWNER;
__constant__ uint32_t* D_DM_HIGH_LOCAL;
__constant__ uint32_t* D_DM_OWNED_ROWS;
__constant__ uint32_t D_DM_OWNED_OFF[MAXW + 2];
__constant__ Code D_DM_MAIN_OFF[MAXGPU][64];
__constant__ Code D_DM_BLOCK_OFF[MAXGPU][32];

__device__ __forceinline__ Count* b300_mask_main_ptr(uint32_t bid, uint32_t hr, uint32_t lr) {
    StorageBlock b = D_DR_MAIN_BLOCKS[bid];
    uint32_t ai = D_F_HIGH_ALL_OFF[b.he] + hr;
    uint32_t owner = D_DM_HIGH_OWNER[ai];
    uint32_t local_hr = D_DM_HIGH_LOCAL[ai];
    return D_MAIN_PTR[owner] + D_DM_MAIN_OFF[owner][bid] + Code(local_hr) * b.cols + lr;
}
__device__ __forceinline__ Count* b300_mask_block_ptr(uint32_t bid, uint32_t hr, uint32_t lr) {
    StorageBlock b = D_DR_BLOCK_BLOCKS[bid];
    uint32_t ai = D_F_HIGH_ALL_OFF[b.he] + hr;
    uint32_t owner = D_DM_HIGH_OWNER[ai];
    uint32_t local_hr = D_DM_HIGH_LOCAL[ai];
    return D_BLOCK_PTR[owner] + D_DM_BLOCK_OFF[owner][bid] + Code(local_hr) * b.cols + lr;
}

struct B300DirectMaskMapDeviceTables {
    uint8_t* high_owner = nullptr;
    uint32_t* high_local = nullptr;
    uint32_t* owned_rows = nullptr;

    void install(const B300DirectMaskShardHost& shard, int g) {
        ck(cudaMalloc(&high_owner, shard.high_owner.size() * sizeof(uint8_t)), "maskshard high owner");
        ck(cudaMemcpy(high_owner, shard.high_owner.data(), shard.high_owner.size() * sizeof(uint8_t),
                      cudaMemcpyHostToDevice), "maskshard copy high owner");
        ck(cudaMalloc(&high_local, shard.high_local.size() * sizeof(uint32_t)), "maskshard high local");
        ck(cudaMemcpy(high_local, shard.high_local.data(), shard.high_local.size() * sizeof(uint32_t),
                      cudaMemcpyHostToDevice), "maskshard copy high local");
        if (!shard.owned_rows[g].empty()) {
            ck(cudaMalloc(&owned_rows, shard.owned_rows[g].size() * sizeof(uint32_t)), "maskshard owned rows");
            ck(cudaMemcpy(owned_rows, shard.owned_rows[g].data(),
                          shard.owned_rows[g].size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "maskshard copy owned rows");
        }
        ck(cudaMemcpyToSymbol(D_DM_HIGH_OWNER, &high_owner, sizeof(high_owner)), "maskshard owner ptr");
        ck(cudaMemcpyToSymbol(D_DM_HIGH_LOCAL, &high_local, sizeof(high_local)), "maskshard local ptr");
        ck(cudaMemcpyToSymbol(D_DM_OWNED_ROWS, &owned_rows, sizeof(owned_rows)), "maskshard rows ptr");
        ck(cudaMemcpyToSymbol(D_DM_OWNED_OFF, shard.owned_off[g].data(),
                              sizeof(uint32_t) * (MAXW + 2)), "maskshard rows off");
        ck(cudaMemcpyToSymbol(D_DM_MAIN_OFF, shard.main_off.data(), sizeof(shard.main_off)), "maskshard main off");
        ck(cudaMemcpyToSymbol(D_DM_BLOCK_OFF, shard.block_off.data(), sizeof(shard.block_off)), "maskshard block off");
    }

    void release() {
        if (high_owner) cudaFree(high_owner);
        if (high_local) cudaFree(high_local);
        if (owned_rows) cudaFree(owned_rows);
        high_owner = nullptr; high_local = owned_rows = nullptr;
    }
};

static void b300_mask_install_layout(
    const StorageLayout& layout, const B300DirectMaskShardHost& shard,
    int self, Count** main_ptrs, Count** block_ptrs
) {
    int mn = int(layout.main_blocks.size()), bn = int(layout.block_blocks.size());
    ck(cudaMemcpyToSymbol(D_DR_MAIN_BLOCKS, layout.main_blocks.data(),
                          layout.main_blocks.size() * sizeof(StorageBlock)), "maskshard main blocks");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_BLOCKS, layout.block_blocks.data(),
                          layout.block_blocks.size() * sizeof(StorageBlock)), "maskshard block blocks");
    ck(cudaMemcpyToSymbol(D_DR_MAIN_NBLOCKS, &mn, sizeof(mn)), "maskshard main nblocks");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_NBLOCKS, &bn, sizeof(bn)), "maskshard block nblocks");
    ck(cudaMemcpyToSymbol(D_DR_SELF, &self, sizeof(self)), "maskshard self");
    ck(cudaMemcpyToSymbol(D_NGPU, &shard.ngpu, sizeof(shard.ngpu)), "maskshard ngpu");
    ck(cudaMemcpyToSymbol(D_MAIN_PTR, main_ptrs, sizeof(Count*) * MAXGPU), "maskshard main ptrs");
    ck(cudaMemcpyToSymbol(D_BLOCK_PTR, block_ptrs, sizeof(Count*) * MAXGPU), "maskshard block ptrs");
}

__global__ void b300_mask_high_orbit_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_ORBIT_OFF[pi], e = D_BS_HIGH_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    B300SparseOrbitOp op = D_BS_HIGH_ORBIT[q];
    uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t hr = b300_sparse_src(op), jhr = b300_sparse_jrank(op), dhr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
        Count* ip = b300_mask_main_ptr(sb, hr, lr);
        Count* jp = b300_mask_main_ptr(jb, jhr, lr);
        Count* dp = b300_mask_block_ptr(db, dhr, lr);
        Count c = *ip, d = *dp;
        if (kind == HIGH_ORBIT_NN) {
            *jp = b300_sparse_add(*jp, c);
            *ip = b300_sparse_add(c, d);
            *dp = 0;
        } else {
            Count cc = *jp;
            *ip = b300_sparse_add(b300_sparse_add(c, cc), d);
            *dp = c;
        }
    }
}

__global__ void b300_mask_high_closure_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi], e = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op), hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    uint32_t db = highdesc_block(desc), dhr = highdesc_rank(desc);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    StorageBlock y = D_DR_BLOCK_BLOCKS[db];
    uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_mask_main_ptr(sb, hr, lr);
            if (c) atomic_add_mod(b300_mask_block_ptr(db, dhr, lr), c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_mask_main_ptr(sb, hr, lr);
            if (!c) continue;
            uint32_t lc = D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs] + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = b300_direct_low_all_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu) atomic_add_mod(b300_mask_block_ptr(db, dhr, lr2), c);
        }
    }
}

__global__ void b300_mask_low_orbit_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_ORBIT_OFF[pi], e = D_BS_LOW_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    B300SparseOrbitOp op = D_BS_LOW_ORBIT[q];
    uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t lr = b300_sparse_src(op), jlr = b300_sparse_jrank(op), dlr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    uint32_t ra = D_DM_OWNED_OFF[x.he], re = D_DM_OWNED_OFF[x.he + 1];
    for (uint32_t k = ra + threadIdx.x; k < re; k += blockDim.x) {
        uint32_t hr = D_DM_OWNED_ROWS[k];
        Count* ip = b300_mask_main_ptr(sb, hr, lr);
        Count* jp = b300_mask_main_ptr(jb, hr, jlr);
        Count* dp = b300_mask_block_ptr(db, hr, dlr);
        Count c = *ip, d = *dp;
        if (kind == CPU_ORBIT_NN) {
            *jp = b300_sparse_add(*jp, c);
            *ip = b300_sparse_add(c, d);
            *dp = 0;
        } else {
            Count cc = *jp;
            Count all = b300_sparse_add(b300_sparse_add(c, cc), d);
            if (p == 1) {
                *ip = all;
                *jp = b300_sparse_add(c, cc);
                *dp = 0;
            } else {
                *ip = all;
                *dp = c;
            }
        }
    }
}

__global__ void b300_mask_low_closure_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_CLOSURE_OFF[pi], e = D_BS_LOW_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_LOW_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op), lr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t kind = lowdesc_kind(desc);
    uint32_t ra = D_DM_OWNED_OFF[x.he], re = D_DM_OWNED_OFF[x.he + 1];
    for (uint32_t k = ra + threadIdx.x; k < re; k += blockDim.x) {
        uint32_t hr = D_DM_OWNED_ROWS[k];
        Count c = *b300_mask_main_ptr(sb, hr, lr);
        if (!c) continue;
        if (kind == LOWDESC_MAIN) {
            uint32_t mb = lowdesc_block(desc);
            atomic_add_mod(b300_mask_main_ptr(mb, hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_BLOCK) {
            uint32_t db = lowdesc_block(desc);
            atomic_add_mod(b300_mask_block_ptr(db, hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t hc = D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he] + hr];
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                uint32_t mb = lowdesc_block(desc);
                StorageBlock y = D_DR_MAIN_BLOCKS[mb];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(b300_mask_main_ptr(mb, hr2, lowdesc_lr(desc)), c);
            } else {
                uint32_t db = lowdesc_block(desc);
                StorageBlock y = D_DR_BLOCK_BLOCKS[db];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(b300_mask_block_ptr(db, hr2, lowdesc_lr(desc)), c);
            }
        }
    }
}
