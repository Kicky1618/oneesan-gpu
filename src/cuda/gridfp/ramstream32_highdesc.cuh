#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

// Symmetric descriptor for the HIGH window.  HIGH + center are active while
// exact LOW topology is normally unchanged.  Only LL whose partner search
// crosses below the center needs the full topology codec.
static constexpr uint32_t HIGHDESC_RANK_MASK = (1u << 20) - 1u;
static constexpr uint32_t HIGHDESC_BLOCK_MASK = 0x3fu;
static constexpr int HIGHDESC_BLOCK_SHIFT = 20;
static constexpr int HIGHDESC_KIND_SHIFT = 30;

enum HighDescKind : uint32_t {
    HIGHDESC_INVALID = 0,
    HIGHDESC_MAIN = 1,
    HIGHDESC_BLOCK = 2,
    HIGHDESC_CROSS = 3,
};

static inline uint32_t highdesc_pack(HighDescKind kind, uint32_t block, uint32_t rank) {
    if (rank > HIGHDESC_RANK_MASK || block > HIGHDESC_BLOCK_MASK) {
        std::cerr << "highdesc encoding overflow block=" << block << " rank=" << rank << "\n";
        std::exit(60);
    }
    return (uint32_t(kind) << HIGHDESC_KIND_SHIFT)
        | (block << HIGHDESC_BLOCK_SHIFT) | rank;
}

struct HighDescHost {
    std::vector<uint32_t> main_desc;
    std::vector<uint32_t> block_desc;
    std::array<uint32_t, 64> main_base{};
    std::array<uint32_t, 32> block_base{};
    uint32_t main_total = 0;
    uint32_t block_total = 0;
    uint64_t main_observations = 0;
    uint64_t main_cross = 0;
    uint64_t main_invalid = 0;
};

static HighDescHost build_high_descriptors(
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    HighDescHost d;
    uint32_t mt = 0;
    for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
        d.main_base[bid] = mt;
        mt += layout.main_blocks[bid].rows;
    }
    uint32_t bt = 0;
    for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
        d.block_base[bid] = bt;
        bt += layout.block_blocks[bid].rows;
    }
    d.main_total = mt;
    d.block_total = bt;
    d.main_desc.assign(size_t(mt) * H, highdesc_pack(HIGHDESC_INVALID, 0, 0));
    d.block_desc.assign(size_t(bt) * H, highdesc_pack(HIGHDESC_INVALID, 0, 0));

    auto representative_low = [&](int hs) -> uint32_t {
        uint32_t a = storage.low_all_off[hs];
        uint32_t b = storage.low_all_off[hs + 1];
        return a < b ? storage.low_all_codes[a] : 0xffffffffu;
    };

    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        int pi = (TARGET_W - 1) - p;

        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.cols || !sb.rows) continue;
            uint32_t lc = representative_low(sb.hs);
            if (lc == 0xffffffffu) continue;
            uint32_t high0 = storage.high_all_off[sb.he];

            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                ++d.main_observations;
                uint32_t hc = storage.high_all_codes[high0 + hr];
                MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                uint32_t out = highdesc_pack(HIGHDESC_INVALID, 0, 0);

                if (!z.valid) {
                    ++d.main_invalid;
                } else {
                    uint32_t lc2 = uint32_t(z.mate) & LM;
                    if (lc2 != lc) {
                        out = highdesc_pack(HIGHDESC_CROSS, 0, 0);
                        ++d.main_cross;
                    } else if (z.blocked) {
                        uint32_t hc2 = uint32_t((z.mate >> (2 * L)) & HM);
                        uint32_t packed = storage.high_packed_rank[hc2];
                        if (packed == 0xffffffffu) {
                            std::cerr << "highdesc blocked HIGH code missing\n";
                            std::exit(61);
                        }
                        uint32_t hr2 = packed >> H;
                        int h2 = seg_end_height_host(hc2, H);
                        uint32_t dbid = uint32_t(h2);
                        if (dbid >= layout.block_blocks.size()
                            || hr2 >= layout.block_blocks[dbid].rows) {
                            std::cerr << "highdesc blocked destination mismatch\n";
                            std::exit(62);
                        }
                        out = highdesc_pack(HIGHDESC_BLOCK, dbid, hr2);
                    } else {
                        uint32_t hc2 = uint32_t((z.mate >> (2 * (L + 1))) & HM);
                        uint32_t packed = storage.high_packed_rank[hc2];
                        if (packed == 0xffffffffu) {
                            std::cerr << "highdesc main HIGH code missing\n";
                            std::exit(63);
                        }
                        uint32_t hr2 = packed >> H;
                        int he2 = seg_end_height_host(hc2, H);
                        int cv2 = int(mget(z.mate, L));
                        uint32_t dbid = uint32_t(3 * he2 + cv2);
                        if (dbid >= layout.main_blocks.size()
                            || hr2 >= layout.main_blocks[dbid].rows) {
                            std::cerr << "highdesc main destination mismatch\n";
                            std::exit(64);
                        }
                        out = highdesc_pack(HIGHDESC_MAIN, dbid, hr2);
                    }
                }
                d.main_desc[size_t(pi) * mt + d.main_base[bid] + hr] = out;
            }
        }

        for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.block_blocks[bid];
            if (!sb.valid || !sb.cols || !sb.rows) continue;
            uint32_t lc = representative_low(sb.hs);
            if (lc == 0xffffffffu) continue;
            uint32_t high0 = storage.high_all_off[sb.he];

            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                uint32_t hc = storage.high_all_codes[high0 + hr];
                MateID m = MateID(lc) | (MateID(hc) << (2 * L));
                MateID z = oneesan::gridfp::blocked_exclude(m, p);
                uint32_t lc2 = uint32_t(z) & LM;
                if (lc2 != lc) {
                    std::cerr << "blocked highdesc unexpectedly changed LOW\n";
                    std::exit(65);
                }
                uint32_t hc2 = uint32_t((z >> (2 * (L + 1))) & HM);
                uint32_t packed = storage.high_packed_rank[hc2];
                if (packed == 0xffffffffu) {
                    std::cerr << "blocked highdesc HIGH code missing\n";
                    std::exit(66);
                }
                uint32_t hr2 = packed >> H;
                int he2 = seg_end_height_host(hc2, H);
                int cv2 = int(mget(z, L));
                uint32_t dbid = uint32_t(3 * he2 + cv2);
                if (dbid >= layout.main_blocks.size()
                    || hr2 >= layout.main_blocks[dbid].rows) {
                    std::cerr << "blocked highdesc destination mismatch\n";
                    std::exit(67);
                }
                d.block_desc[size_t(pi) * bt + d.block_base[bid] + hr]
                    = highdesc_pack(HIGHDESC_MAIN, dbid, hr2);
            }
        }
    }

    double cross_frac = d.main_observations
        ? double(d.main_cross) / double(d.main_observations) : 0.0;
    double invalid_frac = d.main_observations
        ? double(d.main_invalid) / double(d.main_observations) : 0.0;
    std::cerr
        << "highdesc main_active=" << d.main_total
        << " block_active=" << d.block_total
        << " main_desc_mib=" << double(d.main_desc.size() * sizeof(uint32_t)) / (1 << 20)
        << " block_desc_mib=" << double(d.block_desc.size() * sizeof(uint32_t)) / (1 << 20)
        << " cross_frac=" << cross_frac
        << " invalid_frac=" << invalid_frac
        << "\n";
    return d;
}

__constant__ uint32_t* D_HIGHDESC_MAIN;
__constant__ uint32_t* D_HIGHDESC_BLOCK;
__constant__ uint32_t D_HIGHDESC_MAIN_BASE[64];
__constant__ uint32_t D_HIGHDESC_BLOCK_BASE[32];
__constant__ uint32_t D_HIGHDESC_MAIN_TOTAL;
__constant__ uint32_t D_HIGHDESC_BLOCK_TOTAL;

struct HighDescDeviceTables {
    uint32_t* main_desc = nullptr;
    uint32_t* block_desc = nullptr;

    void install(const HighDescHost& d) {
        if (!d.main_desc.empty()) {
            ck(cudaMalloc(&main_desc, d.main_desc.size() * sizeof(uint32_t)), "highdesc main alloc");
            ck(cudaMemcpy(main_desc, d.main_desc.data(), d.main_desc.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "highdesc main copy");
        }
        if (!d.block_desc.empty()) {
            ck(cudaMalloc(&block_desc, d.block_desc.size() * sizeof(uint32_t)), "highdesc block alloc");
            ck(cudaMemcpy(block_desc, d.block_desc.data(), d.block_desc.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "highdesc block copy");
        }
        ck(cudaMemcpyToSymbol(D_HIGHDESC_MAIN, &main_desc, sizeof(main_desc)), "highdesc main ptr");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_BLOCK, &block_desc, sizeof(block_desc)), "highdesc block ptr");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_MAIN_BASE, d.main_base.data(),
                              sizeof(uint32_t) * d.main_base.size()), "highdesc main base");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_BLOCK_BASE, d.block_base.data(),
                              sizeof(uint32_t) * d.block_base.size()), "highdesc block base");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_MAIN_TOTAL, &d.main_total, sizeof(d.main_total)),
                              "highdesc main total");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_BLOCK_TOTAL, &d.block_total, sizeof(d.block_total)),
                              "highdesc block total");
    }

    void release() {
        if (main_desc) cudaFree(main_desc);
        if (block_desc) cudaFree(block_desc);
        main_desc = block_desc = nullptr;
    }
};

__device__ __forceinline__ uint32_t highdesc_kind(uint32_t x) {
    return x >> HIGHDESC_KIND_SHIFT;
}
__device__ __forceinline__ uint32_t highdesc_block(uint32_t x) {
    return (x >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK;
}
__device__ __forceinline__ uint32_t highdesc_rank(uint32_t x) {
    return x & HIGHDESC_RANK_MASK;
}

__global__ void main_group_highdesc_kernel(
    const Count* in, Code n, Count* out_main, Count* out_block, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (; i < n; i += step) {
        Count c = in[i];
        if (!c) continue;

        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint32_t desc = D_HIGHDESC_MAIN[size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                                         + D_HIGHDESC_MAIN_BASE[bid] + hr];
        uint32_t kind = highdesc_kind(desc);

        if (kind == HIGHDESC_MAIN) {
            FBlock y = D_F_MAIN_BLOCKS[highdesc_block(desc)];
            Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
            atomic_add_mod(out_main + j, c);
        } else if (kind == HIGHDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
            atomic_add_mod(out_block + j, c);
        } else if (kind == HIGHDESC_CROSS) {
            MateID m = factor_unrank_main(i);
            auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
            if (!z.valid) continue;
            if (z.blocked) atomic_add_mod(out_block + factor_rank_block(z.mate), c);
            else atomic_add_mod(out_main + factor_rank_main(z.mate), c);
        }
    }
}

__global__ void blocked_group_highdesc_kernel(
    const Count* in, Code n, Count* out_main, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (; i < n; i += step) {
        Count c = in[i];
        if (!c) continue;

        int bid = f_find_block(i);
        FBlock x = D_F_BLOCK_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint32_t desc = D_HIGHDESC_BLOCK[size_t(pi) * D_HIGHDESC_BLOCK_TOTAL
                                          + D_HIGHDESC_BLOCK_BASE[bid] + hr];
        if (highdesc_kind(desc) != HIGHDESC_MAIN) continue;
        FBlock y = D_F_MAIN_BLOCKS[highdesc_block(desc)];
        Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
        atomic_add_mod(out_main + j, c);
    }
}
