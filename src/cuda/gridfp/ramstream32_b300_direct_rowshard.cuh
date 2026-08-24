#pragma once

#include "ramstream32_b300_sparse_actions.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Direct factorized-HBM layout.
//
// Every StorageBlock is a [HIGH row][LOW column] matrix.  GPU g owns exactly
// the rows hr with hr % ngpu == g, packed locally in increasing hr order.
// Therefore LOW-window non-CROSS transitions never leave the owning GPU, while
// HIGH-window actions can be partitioned by their source row owner.
struct B300DirectRowShardHost {
    int ngpu = 0;
    std::array<Code, MAXGPU> main_count{};
    std::array<Code, MAXGPU> block_count{};
    std::array<std::array<Code, 64>, MAXGPU> main_off{};
    std::array<std::array<Code, 32>, MAXGPU> block_off{};
};

static inline uint32_t b300_direct_owned_rows(uint32_t rows, int g, int ngpu) {
    if (g < 0 || g >= ngpu || uint32_t(g) >= rows) return 0;
    return 1u + (rows - 1u - uint32_t(g)) / uint32_t(ngpu);
}

static B300DirectRowShardHost build_b300_direct_row_shards(
    const StorageLayout& layout, int ngpu
) {
    if (ngpu < 1 || ngpu > MAXGPU) std::exit(440);
    B300DirectRowShardHost z;
    z.ngpu = ngpu;
    for (int g = 0; g < ngpu; ++g) {
        Code off = 0;
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            z.main_off[g][bid] = off;
            const auto& b = layout.main_blocks[bid];
            off += Code(b300_direct_owned_rows(b.rows, g, ngpu)) * b.cols;
        }
        z.main_count[g] = off;
        off = 0;
        for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
            z.block_off[g][bid] = off;
            const auto& b = layout.block_blocks[bid];
            off += Code(b300_direct_owned_rows(b.rows, g, ngpu)) * b.cols;
        }
        z.block_count[g] = off;
    }
    return z;
}

__constant__ StorageBlock D_DR_MAIN_BLOCKS[64];
__constant__ StorageBlock D_DR_BLOCK_BLOCKS[32];
__constant__ Code D_DR_MAIN_OFF[MAXGPU][64];
__constant__ Code D_DR_BLOCK_OFF[MAXGPU][32];
__constant__ int D_DR_MAIN_NBLOCKS;
__constant__ int D_DR_BLOCK_NBLOCKS;
__constant__ int D_DR_SELF;
__constant__ uint32_t* D_DR_LOW_MASK_BEGIN;
__constant__ uint32_t* D_DR_HIGH_MASK_BEGIN;

__device__ __forceinline__ Count* b300_direct_main_ptr(uint32_t bid, uint32_t hr, uint32_t lr) {
    uint32_t owner = hr % uint32_t(D_NGPU);
    uint32_t local_hr = hr / uint32_t(D_NGPU);
    StorageBlock b = D_DR_MAIN_BLOCKS[bid];
    return D_MAIN_PTR[owner] + D_DR_MAIN_OFF[owner][bid] + Code(local_hr) * b.cols + lr;
}
__device__ __forceinline__ Count* b300_direct_block_ptr(uint32_t bid, uint32_t hr, uint32_t lr) {
    uint32_t owner = hr % uint32_t(D_NGPU);
    uint32_t local_hr = hr / uint32_t(D_NGPU);
    StorageBlock b = D_DR_BLOCK_BLOCKS[bid];
    return D_BLOCK_PTR[owner] + D_DR_BLOCK_OFF[owner][bid] + Code(local_hr) * b.cols + lr;
}

__device__ __forceinline__ uint32_t b300_direct_low_occ(uint32_t code) {
    uint32_t mask = 0;
#pragma unroll
    for (int p = 0; p < LOW_LUT_K; ++p)
        if ((code >> (2 * p)) & 3u) mask |= 1u << p;
    return mask;
}
__device__ __forceinline__ uint32_t b300_direct_high_occ(uint32_t code) {
    uint32_t mask = 0;
#pragma unroll
    for (int p = 0; p < HIGH_LUT_K; ++p)
        if ((code >> (2 * p)) & 3u) mask |= 1u << p;
    return mask;
}

__device__ __forceinline__ uint32_t b300_direct_low_all_rank(uint32_t code, int h) {
    constexpr int S = MAXW + 2;
    uint32_t mask = b300_direct_low_occ(code);
    uint32_t a = D_F_LOW_MASK_OFF[size_t(mask) * S + h];
    uint32_t b = D_F_LOW_MASK_OFF[size_t(mask) * S + h + 1];
    uint32_t lo = a, hi = b;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t v = D_F_LOW_MASK_CODES[mid];
        if (v < code) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= b || D_F_LOW_MASK_CODES[lo] != code) return 0xffffffffu;
    return D_DR_LOW_MASK_BEGIN[size_t(mask) * S + h] + (lo - a);
}
__device__ __forceinline__ uint32_t b300_direct_high_all_rank(uint32_t code, int h) {
    constexpr int S = MAXW + 2;
    uint32_t mask = b300_direct_high_occ(code);
    uint32_t a = D_F_HIGH_MASK_OFF[size_t(mask) * S + h];
    uint32_t b = D_F_HIGH_MASK_OFF[size_t(mask) * S + h + 1];
    uint32_t lo = a, hi = b;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t v = D_F_HIGH_MASK_CODES[mid];
        if (v < code) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= b || D_F_HIGH_MASK_CODES[lo] != code) return 0xffffffffu;
    return D_DR_HIGH_MASK_BEGIN[size_t(mask) * S + h] + (lo - a);
}

struct B300DirectStorageDeviceTables {
    uint32_t* low_all = nullptr;
    uint32_t* high_all = nullptr;
    uint32_t* low_begin = nullptr;
    uint32_t* high_begin = nullptr;

    static void upload(uint32_t** p, const std::vector<uint32_t>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(p, v.size() * sizeof(uint32_t)), what);
        ck(cudaMemcpy(*p, v.data(), v.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), what);
    }

    void install(const StorageFactorHost& storage) {
        upload(&low_all, storage.low_all_codes, "direct storage low all");
        upload(&high_all, storage.high_all_codes, "direct storage high all");
        upload(&low_begin, storage.low_mask_begin, "direct storage low mask begin");
        upload(&high_begin, storage.high_mask_begin, "direct storage high mask begin");
        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_CODES, &low_all, sizeof(low_all)), "direct low all ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_CODES, &high_all, sizeof(high_all)), "direct high all ptr");
        ck(cudaMemcpyToSymbol(D_DR_LOW_MASK_BEGIN, &low_begin, sizeof(low_begin)), "direct low begin ptr");
        ck(cudaMemcpyToSymbol(D_DR_HIGH_MASK_BEGIN, &high_begin, sizeof(high_begin)), "direct high begin ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF, storage.low_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "direct low all off");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF, storage.high_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "direct high all off");
    }

    uint64_t bytes(const StorageFactorHost& storage) const {
        return uint64_t(storage.low_all_codes.size() + storage.high_all_codes.size()
                      + storage.low_mask_begin.size() + storage.high_mask_begin.size())
             * sizeof(uint32_t);
    }

    void release() {
        if (low_all) cudaFree(low_all);
        if (high_all) cudaFree(high_all);
        if (low_begin) cudaFree(low_begin);
        if (high_begin) cudaFree(high_begin);
        low_all = high_all = low_begin = high_begin = nullptr;
    }
};

static void b300_direct_install_layout(
    const StorageLayout& layout, const B300DirectRowShardHost& shard, int self,
    Count** main_ptrs, Count** block_ptrs
) {
    int mn = int(layout.main_blocks.size());
    int bn = int(layout.block_blocks.size());
    ck(cudaMemcpyToSymbol(D_DR_MAIN_BLOCKS, layout.main_blocks.data(),
                          layout.main_blocks.size() * sizeof(StorageBlock)), "direct main blocks");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_BLOCKS, layout.block_blocks.data(),
                          layout.block_blocks.size() * sizeof(StorageBlock)), "direct block blocks");
    ck(cudaMemcpyToSymbol(D_DR_MAIN_OFF, shard.main_off.data(), sizeof(shard.main_off)), "direct main offsets");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_OFF, shard.block_off.data(), sizeof(shard.block_off)), "direct block offsets");
    ck(cudaMemcpyToSymbol(D_DR_MAIN_NBLOCKS, &mn, sizeof(mn)), "direct main nblocks");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_NBLOCKS, &bn, sizeof(bn)), "direct block nblocks");
    ck(cudaMemcpyToSymbol(D_DR_SELF, &self, sizeof(self)), "direct self");
    ck(cudaMemcpyToSymbol(D_NGPU, &shard.ngpu, sizeof(shard.ngpu)), "direct ngpu");
    ck(cudaMemcpyToSymbol(D_MAIN_PTR, main_ptrs, sizeof(Count*) * MAXGPU), "direct main ptrs");
    ck(cudaMemcpyToSymbol(D_BLOCK_PTR, block_ptrs, sizeof(Count*) * MAXGPU), "direct block ptrs");
}

struct B300DirectSparsePartitionHost {
    int ngpu = 0;
    std::array<std::vector<B300SparseOrbitOp>, MAXGPU> high_orbit;
    std::array<std::vector<uint64_t>, MAXGPU> high_closure;
    std::array<std::vector<uint32_t>, MAXGPU> high_orbit_off;
    std::array<std::vector<uint32_t>, MAXGPU> high_closure_off;
};

static B300DirectSparsePartitionHost b300_direct_partition_high(
    const B300SparseActionsHost& sparse, int ngpu
) {
    B300DirectSparsePartitionHost z;
    z.ngpu = ngpu;
    for (int g = 0; g < ngpu; ++g) {
        z.high_orbit_off[g].resize(HIGH_LUT_K + 1);
        z.high_closure_off[g].resize(HIGH_LUT_K + 1);
    }
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (int g = 0; g < ngpu; ++g) {
            z.high_orbit_off[g][pi] = uint32_t(z.high_orbit[g].size());
            z.high_closure_off[g][pi] = uint32_t(z.high_closure[g].size());
        }
        for (uint32_t q = sparse.high_orbit_off[pi]; q < sparse.high_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.high_orbit[q];
            int g = int(b300_sparse_src(op) % uint32_t(ngpu));
            z.high_orbit[g].push_back(op);
        }
        for (uint32_t q = sparse.high_closure_off[pi]; q < sparse.high_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.high_closure[q];
            int g = int(b300_sparse_closure_src(op) % uint32_t(ngpu));
            z.high_closure[g].push_back(op);
        }
    }
    for (int g = 0; g < ngpu; ++g) {
        z.high_orbit_off[g][HIGH_LUT_K] = uint32_t(z.high_orbit[g].size());
        z.high_closure_off[g][HIGH_LUT_K] = uint32_t(z.high_closure[g].size());
    }
    return z;
}

struct B300DirectSparseDeviceTables {
    B300SparseOrbitOp* low_orbit = nullptr;
    uint64_t* low_closure = nullptr;
    B300SparseOrbitOp* high_orbit = nullptr;
    uint64_t* high_closure = nullptr;

    template<class T>
    static void upload(T** p, const std::vector<T>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(p, v.size() * sizeof(T)), what);
        ck(cudaMemcpy(*p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(const B300SparseActionsHost& sparse,
                 const B300DirectSparsePartitionHost& part, int g) {
        upload(&low_orbit, sparse.low_orbit, "direct low orbit");
        upload(&low_closure, sparse.low_closure, "direct low closure");
        upload(&high_orbit, part.high_orbit[g], "direct high orbit");
        upload(&high_closure, part.high_closure[g], "direct high closure");
        ck(cudaMemcpyToSymbol(D_BS_LOW_ORBIT, &low_orbit, sizeof(low_orbit)), "direct low orbit ptr");
        ck(cudaMemcpyToSymbol(D_BS_LOW_CLOSURE, &low_closure, sizeof(low_closure)), "direct low closure ptr");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_ORBIT, &high_orbit, sizeof(high_orbit)), "direct high orbit ptr");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_CLOSURE, &high_closure, sizeof(high_closure)), "direct high closure ptr");

        std::array<uint32_t, MAXW + 1> lo{}, lc{}, ho{}, hc{};
        std::copy(sparse.low_orbit_off.begin(), sparse.low_orbit_off.end(), lo.begin());
        std::copy(sparse.low_closure_off.begin(), sparse.low_closure_off.end(), lc.begin());
        std::copy(part.high_orbit_off[g].begin(), part.high_orbit_off[g].end(), ho.begin());
        std::copy(part.high_closure_off[g].begin(), part.high_closure_off[g].end(), hc.begin());
        ck(cudaMemcpyToSymbol(D_BS_LOW_ORBIT_OFF, lo.data(), sizeof(lo)), "direct low orbit off");
        ck(cudaMemcpyToSymbol(D_BS_LOW_CLOSURE_OFF, lc.data(), sizeof(lc)), "direct low closure off");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_ORBIT_OFF, ho.data(), sizeof(ho)), "direct high orbit off");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_CLOSURE_OFF, hc.data(), sizeof(hc)), "direct high closure off");
    }

    void release() {
        if (low_orbit) cudaFree(low_orbit);
        if (low_closure) cudaFree(low_closure);
        if (high_orbit) cudaFree(high_orbit);
        if (high_closure) cudaFree(high_closure);
        low_orbit = high_orbit = nullptr;
        low_closure = high_closure = nullptr;
    }
};

// HIGH: this GPU owns every op in its partition.  The source row is local;
// partner/drop rows may be peer memory.
__global__ void b300_direct_high_orbit_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_ORBIT_OFF[pi], b = D_BS_HIGH_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    B300SparseOrbitOp op = D_BS_HIGH_ORBIT[q];
    uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t hr = b300_sparse_src(op), jhr = b300_sparse_jrank(op), dhr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
        Count* ip = b300_direct_main_ptr(sb, hr, lr);
        Count* jp = b300_direct_main_ptr(jb, jhr, lr);
        Count* dp = b300_direct_block_ptr(db, dhr, lr);
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

__global__ void b300_direct_high_closure_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi], b = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op);
    uint32_t hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    uint32_t db = highdesc_block(desc), dhr = highdesc_rank(desc);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    StorageBlock y = D_DR_BLOCK_BLOCKS[db];
    uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_direct_main_ptr(sb, hr, lr);
            if (c) atomic_add_mod(b300_direct_block_ptr(db, dhr, lr), c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_direct_main_ptr(sb, hr, lr);
            if (!c) continue;
            uint32_t lc = D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs] + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = b300_direct_low_all_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu) atomic_add_mod(b300_direct_block_ptr(db, dhr, lr2), c);
        }
    }
}

// LOW: every GPU launches the same sparse op stream, but touches only the HIGH
// rows it owns.  Non-CROSS source/destination rows are therefore always local.
__global__ void b300_direct_low_orbit_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_ORBIT_OFF[pi], b = D_BS_LOW_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    B300SparseOrbitOp op = D_BS_LOW_ORBIT[q];
    uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t lr = b300_sparse_src(op), jlr = b300_sparse_jrank(op), dlr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    uint32_t first = uint32_t(D_DR_SELF) + threadIdx.x * uint32_t(D_NGPU);
    uint32_t step = blockDim.x * uint32_t(D_NGPU);
    for (uint32_t hr = first; hr < x.rows; hr += step) {
        Count* ip = b300_direct_main_ptr(sb, hr, lr);
        Count* jp = b300_direct_main_ptr(jb, hr, jlr);
        Count* dp = b300_direct_block_ptr(db, hr, dlr);
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

__global__ void b300_direct_low_closure_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_CLOSURE_OFF[pi], b = D_BS_LOW_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    uint64_t op = D_BS_LOW_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op);
    uint32_t lr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t kind = lowdesc_kind(desc);
    uint32_t first = uint32_t(D_DR_SELF) + threadIdx.x * uint32_t(D_NGPU);
    uint32_t step = blockDim.x * uint32_t(D_NGPU);
    for (uint32_t hr = first; hr < x.rows; hr += step) {
        Count c = *b300_direct_main_ptr(sb, hr, lr);
        if (!c) continue;
        if (kind == LOWDESC_MAIN) {
            uint32_t mb = lowdesc_block(desc);
            atomic_add_mod(b300_direct_main_ptr(mb, hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_BLOCK) {
            uint32_t db = lowdesc_block(desc);
            atomic_add_mod(b300_direct_block_ptr(db, hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t hc = D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he] + hr];
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                uint32_t mb = lowdesc_block(desc);
                StorageBlock y = D_DR_MAIN_BLOCKS[mb];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(b300_direct_main_ptr(mb, hr2, lowdesc_lr(desc)), c);
            } else {
                uint32_t db = lowdesc_block(desc);
                StorageBlock y = D_DR_BLOCK_BLOCKS[db];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(b300_direct_block_ptr(db, hr2, lowdesc_lr(desc)), c);
            }
        }
    }
}

static inline uint32_t b300_direct_high_orbit_count(
    const B300DirectSparsePartitionHost& p, int g, int edge
) {
    uint32_t i = uint32_t((TARGET_W - 1) - edge);
    return p.high_orbit_off[g][i + 1] - p.high_orbit_off[g][i];
}
static inline uint32_t b300_direct_high_closure_count(
    const B300DirectSparsePartitionHost& p, int g, int edge
) {
    uint32_t i = uint32_t((TARGET_W - 1) - edge);
    return p.high_closure_off[g][i + 1] - p.high_closure_off[g][i];
}
