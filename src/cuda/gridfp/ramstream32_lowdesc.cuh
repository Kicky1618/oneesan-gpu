#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

// Low-window transition descriptor.
//
// The low window touches LOW + center.  For almost every included branch the
// exact HIGH topology is unchanged; only RR whose partner search escapes above
// the center modifies HIGH.  We precompute the LOW-side result once and fall
// back to the full factorized topology codec only for those boundary-crossing
// cases.
//
// bits 31..30: kind
// bits 25..20: destination FBlock index (same-HIGH cases)
// bits 19..0 : destination LOW all-rank
static constexpr uint32_t LOWDESC_LR_MASK = (1u << 20) - 1u;
static constexpr uint32_t LOWDESC_BLOCK_MASK = 0x3fu;
static constexpr int LOWDESC_BLOCK_SHIFT = 20;
static constexpr int LOWDESC_KIND_SHIFT = 30;

enum LowDescKind : uint32_t {
    LOWDESC_INVALID = 0,
    LOWDESC_MAIN = 1,
    LOWDESC_BLOCK = 2,
    LOWDESC_CROSS = 3,
};

static inline uint32_t lowdesc_pack(LowDescKind kind, uint32_t block, uint32_t lr) {
    if (lr > LOWDESC_LR_MASK || block > LOWDESC_BLOCK_MASK) {
        std::cerr << "lowdesc encoding overflow block=" << block << " lr=" << lr << "\n";
        std::exit(50);
    }
    return (uint32_t(kind) << LOWDESC_KIND_SHIFT)
        | (block << LOWDESC_BLOCK_SHIFT) | lr;
}

struct LowDescHost {
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

static LowDescHost build_low_descriptors(
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    LowDescHost d;

    // Descriptor indexing is independent of occupancy mask.  It is the
    // concatenation of each factor block's LOW all-rank dimension.
    uint32_t mt = 0;
    for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
        d.main_base[bid] = mt;
        mt += layout.main_blocks[bid].cols;
    }
    uint32_t bt = 0;
    for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
        d.block_base[bid] = bt;
        bt += layout.block_blocks[bid].cols;
    }
    d.main_total = mt;
    d.block_total = bt;
    d.main_desc.assign(size_t(mt) * L, lowdesc_pack(LOWDESC_INVALID, 0, 0));
    d.block_desc.assign(size_t(bt) * L, lowdesc_pack(LOWDESC_INVALID, 0, 0));

    auto representative_high = [&](int he) -> uint32_t {
        uint32_t a = storage.high_all_off[he];
        uint32_t b = storage.high_all_off[he + 1];
        return a < b ? storage.high_all_codes[a] : 0xffffffffu;
    };

    for (int p = L; p >= 1; --p) {
        int pi = L - p;

        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.cols || !sb.rows) continue;
            int he = sb.he;
            uint32_t hc = representative_high(he);
            if (hc == 0xffffffffu) continue;
            uint32_t low0 = storage.low_all_off[sb.hs];

            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                ++d.main_observations;
                uint32_t lc = storage.low_all_codes[low0 + lr];
                MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                uint32_t out = lowdesc_pack(LOWDESC_INVALID, 0, 0);

                if (!z.valid) {
                    ++d.main_invalid;
                } else {
                    uint32_t hc2 = z.blocked
                        ? uint32_t((z.mate >> (2 * L)) & HM)
                        : uint32_t((z.mate >> (2 * (L + 1))) & HM);
                    if (hc2 != hc) {
                        out = lowdesc_pack(LOWDESC_CROSS, 0, 0);
                        ++d.main_cross;
                    } else {
                        uint32_t lc2 = uint32_t(z.mate) & LM;
                        uint32_t packed = storage.low_packed_rank[lc2];
                        if (packed == 0xffffffffu) {
                            std::cerr << "lowdesc destination LOW code missing\n";
                            std::exit(51);
                        }
                        uint32_t lr2 = packed >> L;
                        if (z.blocked) {
                            uint32_t dbid = uint32_t(he);
                            if (dbid >= layout.block_blocks.size()
                                || lr2 >= layout.block_blocks[dbid].cols) {
                                std::cerr << "lowdesc blocked destination mismatch\n";
                                std::exit(52);
                            }
                            out = lowdesc_pack(LOWDESC_BLOCK, dbid, lr2);
                        } else {
                            int cv2 = int(mget(z.mate, L));
                            uint32_t dbid = uint32_t(3 * he + cv2);
                            if (dbid >= layout.main_blocks.size()
                                || lr2 >= layout.main_blocks[dbid].cols) {
                                std::cerr << "lowdesc main destination mismatch\n";
                                std::exit(53);
                            }
                            out = lowdesc_pack(LOWDESC_MAIN, dbid, lr2);
                        }
                    }
                }
                d.main_desc[size_t(pi) * mt + d.main_base[bid] + lr] = out;
            }
        }

        // blocked_exclude only inserts N.  It never flips HIGH topology, so the
        // blocked branch is descriptor-only with no fallback path.
        for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.block_blocks[bid];
            if (!sb.valid || !sb.cols || !sb.rows) continue;
            int he = sb.he;
            uint32_t hc = representative_high(he);
            if (hc == 0xffffffffu) continue;
            uint32_t low0 = storage.low_all_off[sb.hs];

            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                uint32_t lc = storage.low_all_codes[low0 + lr];
                MateID m = MateID(lc) | (MateID(hc) << (2 * L));
                MateID z = oneesan::gridfp::blocked_exclude(m, p);
                uint32_t hc2 = uint32_t((z >> (2 * (L + 1))) & HM);
                if (hc2 != hc) {
                    std::cerr << "blocked lowdesc unexpectedly changed HIGH\n";
                    std::exit(54);
                }
                uint32_t lc2 = uint32_t(z) & LM;
                uint32_t packed = storage.low_packed_rank[lc2];
                if (packed == 0xffffffffu) {
                    std::cerr << "blocked lowdesc LOW code missing\n";
                    std::exit(55);
                }
                uint32_t lr2 = packed >> L;
                int cv2 = int(mget(z, L));
                uint32_t dbid = uint32_t(3 * he + cv2);
                if (dbid >= layout.main_blocks.size()
                    || lr2 >= layout.main_blocks[dbid].cols) {
                    std::cerr << "blocked lowdesc destination mismatch\n";
                    std::exit(56);
                }
                d.block_desc[size_t(pi) * bt + d.block_base[bid] + lr]
                    = lowdesc_pack(LOWDESC_MAIN, dbid, lr2);
            }
        }
    }

    double cross_frac = d.main_observations
        ? double(d.main_cross) / double(d.main_observations) : 0.0;
    double invalid_frac = d.main_observations
        ? double(d.main_invalid) / double(d.main_observations) : 0.0;
    std::cerr
        << "lowdesc main_active=" << d.main_total
        << " block_active=" << d.block_total
        << " main_desc_mib=" << double(d.main_desc.size() * sizeof(uint32_t)) / (1 << 20)
        << " block_desc_mib=" << double(d.block_desc.size() * sizeof(uint32_t)) / (1 << 20)
        << " cross_frac=" << cross_frac
        << " invalid_frac=" << invalid_frac
        << "\n";
    return d;
}

__constant__ uint32_t* D_LOWDESC_MAIN;
__constant__ uint32_t* D_LOWDESC_BLOCK;
__constant__ uint32_t D_LOWDESC_MAIN_BASE[64];
__constant__ uint32_t D_LOWDESC_BLOCK_BASE[32];
__constant__ uint32_t D_LOWDESC_MAIN_TOTAL;
__constant__ uint32_t D_LOWDESC_BLOCK_TOTAL;

struct LowDescDeviceTables {
    uint32_t* main_desc = nullptr;
    uint32_t* block_desc = nullptr;

    void install(const LowDescHost& d) {
        if (!d.main_desc.empty()) {
            ck(cudaMalloc(&main_desc, d.main_desc.size() * sizeof(uint32_t)), "lowdesc main alloc");
            ck(cudaMemcpy(main_desc, d.main_desc.data(), d.main_desc.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "lowdesc main copy");
        }
        if (!d.block_desc.empty()) {
            ck(cudaMalloc(&block_desc, d.block_desc.size() * sizeof(uint32_t)), "lowdesc block alloc");
            ck(cudaMemcpy(block_desc, d.block_desc.data(), d.block_desc.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "lowdesc block copy");
        }
        ck(cudaMemcpyToSymbol(D_LOWDESC_MAIN, &main_desc, sizeof(main_desc)), "lowdesc main ptr");
        ck(cudaMemcpyToSymbol(D_LOWDESC_BLOCK, &block_desc, sizeof(block_desc)), "lowdesc block ptr");
        ck(cudaMemcpyToSymbol(D_LOWDESC_MAIN_BASE, d.main_base.data(),
                              sizeof(uint32_t) * d.main_base.size()), "lowdesc main base");
        ck(cudaMemcpyToSymbol(D_LOWDESC_BLOCK_BASE, d.block_base.data(),
                              sizeof(uint32_t) * d.block_base.size()), "lowdesc block base");
        ck(cudaMemcpyToSymbol(D_LOWDESC_MAIN_TOTAL, &d.main_total, sizeof(d.main_total)),
                              "lowdesc main total");
        ck(cudaMemcpyToSymbol(D_LOWDESC_BLOCK_TOTAL, &d.block_total, sizeof(d.block_total)),
                              "lowdesc block total");
    }

    void release() {
        if (main_desc) cudaFree(main_desc);
        if (block_desc) cudaFree(block_desc);
        main_desc = block_desc = nullptr;
    }
};

__device__ __forceinline__ uint32_t lowdesc_kind(uint32_t x) {
    return x >> LOWDESC_KIND_SHIFT;
}
__device__ __forceinline__ uint32_t lowdesc_block(uint32_t x) {
    return (x >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK;
}
__device__ __forceinline__ uint32_t lowdesc_lr(uint32_t x) {
    return x & LOWDESC_LR_MASK;
}

__global__ void main_group_lowdesc_kernel(
    const Count* in, Code n, Count* out_main, Count* out_block, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (; i < n; i += step) {
        Count c = in[i];
        if (!c) continue;

        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint32_t desc = D_LOWDESC_MAIN[size_t(pi) * D_LOWDESC_MAIN_TOTAL
                                        + D_LOWDESC_MAIN_BASE[bid] + lr];
        uint32_t kind = lowdesc_kind(desc);

        if (kind == LOWDESC_MAIN) {
            FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(out_main + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(out_block + j, c);
        } else if (kind == LOWDESC_CROSS) {
            // Rare boundary-crossing RR: exact HIGH topology determines which
            // partner endpoint flips, so use the proven full codec here.
            MateID m = factor_unrank_main(i);
            auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
            if (!z.valid) continue;
            if (z.blocked) atomic_add_mod(out_block + factor_rank_block(z.mate), c);
            else atomic_add_mod(out_main + factor_rank_main(z.mate), c);
        }
    }
}

__global__ void blocked_group_lowdesc_kernel(
    const Count* in, Code n, Count* out_main, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (; i < n; i += step) {
        Count c = in[i];
        if (!c) continue;

        int bid = f_find_block(i);
        FBlock x = D_F_BLOCK_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint32_t desc = D_LOWDESC_BLOCK[size_t(pi) * D_LOWDESC_BLOCK_TOTAL
                                         + D_LOWDESC_BLOCK_BASE[bid] + lr];
        if (lowdesc_kind(desc) != LOWDESC_MAIN) continue;
        FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
        Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
        atomic_add_mod(out_main + j, c);
    }
}
