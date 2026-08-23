#pragma once

#include "ramstream32_high_orbit.cuh"
#include "ramstream32_lowmask_major_storage.hpp"

#include <array>
#include <cstdint>
#include <vector>

// Batch-wide HIGH executor.  It indexes the LOW-mask-major authoritative
// layout directly, so no per-mask FBlock constant-memory reconfiguration is
// needed.  One CUDA block owns one contiguous chunk of a (mask,main-block)
// rectangle; threads stream through that chunk.
struct HighRawTask {
    uint32_t elem0 = 0;
    uint32_t count = 0;
    uint16_t mask = 0;
    uint8_t bid = 0;
    uint8_t pad = 0;
};
static_assert(sizeof(HighRawTask) == 12);

static constexpr uint32_t HIGH_RAW_TASK_ELEMS = 65536;

struct HighRawBatchTasks {
    std::vector<HighRawTask> tasks;
};

static HighRawBatchTasks build_high_raw_tasks(
    const LowMaskMajorLayout& mm, const StorageLayout& logical,
    uint32_t first_mask, uint32_t last_mask
) {
    HighRawBatchTasks out;
    uint64_t nt = 0;
    for (uint32_t mask = first_mask; mask < last_mask; ++mask) {
        size_t row = size_t(mask) * (size_t(mm.main_nblocks) + 1);
        for (uint32_t bid = 0; bid < mm.main_nblocks; ++bid) {
            Code n = mm.main_block_off[row + bid + 1] - mm.main_block_off[row + bid];
            nt += (n + HIGH_RAW_TASK_ELEMS - 1) / HIGH_RAW_TASK_ELEMS;
        }
    }
    out.tasks.reserve(size_t(nt));
    for (uint32_t mask = first_mask; mask < last_mask; ++mask) {
        size_t row = size_t(mask) * (size_t(mm.main_nblocks) + 1);
        for (uint32_t bid = 0; bid < mm.main_nblocks; ++bid) {
            Code n = mm.main_block_off[row + bid + 1] - mm.main_block_off[row + bid];
            if (!n) continue;
            if (bid > 255) std::exit(200);
            for (Code e = 0; e < n; e += HIGH_RAW_TASK_ELEMS) {
                Code take = std::min<Code>(HIGH_RAW_TASK_ELEMS, n - e);
                if (e > 0xffffffffULL || take > 0xffffffffULL) std::exit(201);
                HighRawTask t;
                t.elem0 = uint32_t(e);
                t.count = uint32_t(take);
                t.mask = uint16_t(mask);
                t.bid = uint8_t(bid);
                out.tasks.push_back(t);
            }
        }
    }
    return out;
}

__constant__ const Code* D_MM_MAIN_BLOCK_OFF;
__constant__ const Code* D_MM_BLOCK_BLOCK_OFF;
__constant__ uint32_t D_MM_MAIN_NBLOCKS;
__constant__ uint32_t D_MM_BLOCK_NBLOCKS;
__constant__ uint8_t D_MM_MAIN_HS[64];
__constant__ uint8_t D_MM_BLOCK_HS[32];

struct HighRawLayoutDeviceTables {
    Code* main_off = nullptr;
    Code* block_off = nullptr;

    void install(const LowMaskMajorLayout& mm, const StorageLayout& logical) {
        if (logical.main_blocks.size() > 64 || logical.block_blocks.size() > 32)
            std::exit(202);
        if (!mm.main_block_off.empty()) {
            ck(cudaMalloc(&main_off, mm.main_block_off.size() * sizeof(Code)), "rawbatch main off alloc");
            ck(cudaMemcpy(main_off, mm.main_block_off.data(),
                          mm.main_block_off.size() * sizeof(Code), cudaMemcpyHostToDevice),
               "rawbatch main off copy");
        }
        if (!mm.block_block_off.empty()) {
            ck(cudaMalloc(&block_off, mm.block_block_off.size() * sizeof(Code)), "rawbatch block off alloc");
            ck(cudaMemcpy(block_off, mm.block_block_off.data(),
                          mm.block_block_off.size() * sizeof(Code), cudaMemcpyHostToDevice),
               "rawbatch block off copy");
        }
        std::array<uint8_t,64> mhs{};
        std::array<uint8_t,32> dhs{};
        for (size_t i = 0; i < logical.main_blocks.size(); ++i) mhs[i] = logical.main_blocks[i].hs;
        for (size_t i = 0; i < logical.block_blocks.size(); ++i) dhs[i] = logical.block_blocks[i].hs;
        uint32_t mn = mm.main_nblocks, dn = mm.block_nblocks;
        ck(cudaMemcpyToSymbol(D_MM_MAIN_BLOCK_OFF, &main_off, sizeof(main_off)), "rawbatch main off ptr");
        ck(cudaMemcpyToSymbol(D_MM_BLOCK_BLOCK_OFF, &block_off, sizeof(block_off)), "rawbatch block off ptr");
        ck(cudaMemcpyToSymbol(D_MM_MAIN_NBLOCKS, &mn, sizeof(mn)), "rawbatch main nblocks");
        ck(cudaMemcpyToSymbol(D_MM_BLOCK_NBLOCKS, &dn, sizeof(dn)), "rawbatch block nblocks");
        ck(cudaMemcpyToSymbol(D_MM_MAIN_HS, mhs.data(), sizeof(mhs)), "rawbatch main hs");
        ck(cudaMemcpyToSymbol(D_MM_BLOCK_HS, dhs.data(), sizeof(dhs)), "rawbatch block hs");
    }
    void release() {
        if (main_off) cudaFree(main_off);
        if (block_off) cudaFree(block_off);
        main_off = block_off = nullptr;
    }
};

struct HighRawTaskDeviceBuffer {
    HighRawTask* ptr = nullptr;
    size_t cap = 0;
    void ensure(size_t n) {
        if (n <= cap) return;
        if (ptr) cudaFree(ptr);
        ck(cudaMalloc(&ptr, n * sizeof(HighRawTask)), "rawbatch task alloc");
        cap = n;
    }
    void upload(const std::vector<HighRawTask>& v) {
        ensure(v.size());
        if (!v.empty())
            ck(cudaMemcpy(ptr, v.data(), v.size() * sizeof(HighRawTask), cudaMemcpyHostToDevice),
               "rawbatch task upload");
    }
    void release() { if (ptr) cudaFree(ptr); ptr = nullptr; cap = 0; }
};

__device__ __forceinline__ Code mm_main_block_base_dev(uint32_t mask, uint32_t bid) {
    return D_MM_MAIN_BLOCK_OFF[size_t(mask) * (size_t(D_MM_MAIN_NBLOCKS) + 1) + bid];
}
__device__ __forceinline__ Code mm_block_block_base_dev(uint32_t mask, uint32_t bid) {
    return D_MM_BLOCK_BLOCK_OFF[size_t(mask) * (size_t(D_MM_BLOCK_NBLOCKS) + 1) + bid];
}
__device__ __forceinline__ uint32_t mm_low_width_dev(uint32_t mask, uint32_t hs) {
    constexpr int S = MAXW + 2;
    size_t ix = size_t(mask) * S + hs;
    return D_F_LOW_MASK_OFF[ix + 1] - D_F_LOW_MASK_OFF[ix];
}

// Unlike bidesc_low_mask_rank(), raw-batch execution has many LOW masks live in
// one kernel launch and therefore must not read the single per-group D_F_MASK
// constant.  Make the mask an explicit argument.
__device__ __forceinline__ uint32_t rawbatch_low_mask_rank(
    uint32_t mask, uint32_t code, int h
) {
    constexpr int S = MAXW + 2;
    uint32_t a = D_F_LOW_MASK_OFF[size_t(mask) * S + h];
    uint32_t b = D_F_LOW_MASK_OFF[size_t(mask) * S + h + 1];
    uint32_t lo = a, hi = b;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t v = D_F_LOW_MASK_CODES[mid];
        if (v < code) lo = mid + 1;
        else hi = mid;
    }
    return (lo < b && D_F_LOW_MASK_CODES[lo] == code) ? lo - a : 0xffffffffu;
}

__global__ void high_rawbatch_orbit_kernel(
    Count* mainv, Count* blockv,
    const HighRawTask* tasks, uint32_t ntasks,
    Code main_begin, Code block_begin, int p
) {
    uint32_t ti = blockIdx.x;
    if (ti >= ntasks) return;
    HighRawTask t = tasks[ti];
    uint32_t mask = t.mask, bid = t.bid;
    uint32_t sw = mm_low_width_dev(mask, D_MM_MAIN_HS[bid]);
    if (!sw) return;
    Code src_base = mm_main_block_base_dev(mask, bid) - main_begin;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);

    for (uint32_t k = threadIdx.x; k < t.count; k += blockDim.x) {
        uint32_t e = t.elem0 + k;
        uint32_t hr = e / sw;
        uint32_t lr = e - hr * sw;
        uint64_t ow = D_HIGH_ORBIT[size_t(pi) * D_HIGH_ORBIT_MAIN_TOTAL
                                  + D_HIGH_ORBIT_MAIN_BASE[bid] + hr];
        uint32_t kind = high_orbit_kind(ow);
        if (kind < HIGH_ORBIT_NN || kind > HIGH_ORBIT_NL) continue;

        uint32_t jbid = high_orbit_jblock(ow);
        uint32_t dbid = high_orbit_dblock(ow);
        uint32_t jw = mm_low_width_dev(mask, D_MM_MAIN_HS[jbid]);
        uint32_t dw = mm_low_width_dev(mask, D_MM_BLOCK_HS[dbid]);
        Code i = src_base + e;
        Code j = mm_main_block_base_dev(mask, jbid) - main_begin
               + Code(high_orbit_jhr(ow)) * jw + lr;
        Code dj = mm_block_block_base_dev(mask, dbid) - block_begin
                + Code(high_orbit_dhr(ow)) * dw + lr;
        Count c = mainv[i];
        Count d = blockv[dj];
        if (kind == HIGH_ORBIT_NN) {
            mainv[j] = high_orbit_add(mainv[j], c);
            mainv[i] = high_orbit_add(c, d);
            blockv[dj] = 0;
        } else {
            Count cc = mainv[j];
            mainv[i] = high_orbit_add(high_orbit_add(c, cc), d);
            blockv[dj] = c;
        }
    }
}

__global__ void high_rawbatch_closure_kernel(
    Count* mainv, Count* blockv,
    const HighRawTask* tasks, uint32_t ntasks,
    Code main_begin, Code block_begin, int p
) {
    constexpr int S = MAXW + 2;
    uint32_t ti = blockIdx.x;
    if (ti >= ntasks) return;
    HighRawTask t = tasks[ti];
    uint32_t mask = t.mask, bid = t.bid;
    uint32_t sw = mm_low_width_dev(mask, D_MM_MAIN_HS[bid]);
    if (!sw) return;
    Code src_base = mm_main_block_base_dev(mask, bid) - main_begin;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);

    for (uint32_t k = threadIdx.x; k < t.count; k += blockDim.x) {
        uint32_t e = t.elem0 + k;
        uint32_t hr = e / sw;
        uint32_t lr = e - hr * sw;
        uint64_t ow = D_HIGH_ORBIT[size_t(pi) * D_HIGH_ORBIT_MAIN_TOTAL
                                  + D_HIGH_ORBIT_MAIN_BASE[bid] + hr];
        if (high_orbit_kind(ow) != HIGH_ORBIT_CLOSURE) continue;
        Count c = mainv[src_base + e];
        if (!c) continue;
        uint32_t desc = D_HIGHDESC_MAIN[size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                                       + D_HIGHDESC_MAIN_BASE[bid] + hr];
        uint32_t kind = highdesc_kind(desc);
        uint32_t dbid = highdesc_block(desc);
        uint32_t dw = mm_low_width_dev(mask, D_MM_BLOCK_HS[dbid]);
        Code db = mm_block_block_base_dev(mask, dbid) - block_begin;
        if (kind == HIGHDESC_BLOCK) {
            atomic_add_mod(blockv + db + Code(highdesc_rank(desc)) * dw + lr, c);
        } else if (kind == HIGHDESC_CROSS) {
            uint32_t a = D_F_LOW_MASK_OFF[size_t(mask) * S + D_MM_MAIN_HS[bid]];
            uint32_t lc = D_F_LOW_MASK_CODES[a + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = rawbatch_low_mask_rank(mask, lc2, D_MM_BLOCK_HS[dbid]);
            if (lr2 == 0xffffffffu) continue;
            atomic_add_mod(blockv + db + Code(highdesc_rank(desc)) * dw + lr2, c);
        } else {
            asm("trap;");
        }
    }
}
