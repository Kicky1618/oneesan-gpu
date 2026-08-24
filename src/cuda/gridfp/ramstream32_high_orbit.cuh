#pragma once

#include "ramstream32_highdesc.cuh"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

// In-place HIGH-window orbit table.  It is the HIGH-side analogue of
// LowOrbitHost: one record per (edge position, logical main factor block,
// HIGH-all-rank).  LOW topology is invariant for the N* owner orbits, so the
// same record applies to every LOW-mask-local column in that HIGH row.
//
// bits  0..19  partner-main HIGH storage rank
// bits 20..25  partner-main block
// bits 26..45  dropped-N blocked HIGH storage rank
// bits 46..51  dropped-N blocked block
// bits 52..54  kind: 0 none, 1 NN, 2 NR, 3 NL, 4 closure
static constexpr uint64_t HIGH_ORBIT_RANK_MASK = (1ull << 20) - 1ull;
static constexpr uint64_t HIGH_ORBIT_BLOCK_MASK = 0x3full;
static constexpr int HIGH_ORBIT_JBLOCK_SHIFT = 20;
static constexpr int HIGH_ORBIT_DHR_SHIFT = 26;
static constexpr int HIGH_ORBIT_DBLOCK_SHIFT = 46;
static constexpr int HIGH_ORBIT_KIND_SHIFT = 52;

enum HighOrbitKind : uint32_t {
    HIGH_ORBIT_NONE = 0,
    HIGH_ORBIT_NN = 1,
    HIGH_ORBIT_NR = 2,
    HIGH_ORBIT_NL = 3,
    HIGH_ORBIT_CLOSURE = 4,
};

static inline uint64_t high_orbit_pack(
    HighOrbitKind kind, uint32_t jblock = 0, uint32_t jhr = 0,
    uint32_t dblock = 0, uint32_t dhr = 0
) {
    if (jhr > HIGH_ORBIT_RANK_MASK || dhr > HIGH_ORBIT_RANK_MASK
        || jblock > HIGH_ORBIT_BLOCK_MASK || dblock > HIGH_ORBIT_BLOCK_MASK) {
        std::cerr << "high orbit encoding overflow\n";
        std::exit(160);
    }
    return uint64_t(jhr)
        | (uint64_t(jblock) << HIGH_ORBIT_JBLOCK_SHIFT)
        | (uint64_t(dhr) << HIGH_ORBIT_DHR_SHIFT)
        | (uint64_t(dblock) << HIGH_ORBIT_DBLOCK_SHIFT)
        | (uint64_t(kind) << HIGH_ORBIT_KIND_SHIFT);
}

__host__ __device__ static inline uint32_t high_orbit_kind(uint64_t x) {
    return uint32_t((x >> HIGH_ORBIT_KIND_SHIFT) & 7u);
}
__host__ __device__ static inline uint32_t high_orbit_jhr(uint64_t x) {
    return uint32_t(x & HIGH_ORBIT_RANK_MASK);
}
__host__ __device__ static inline uint32_t high_orbit_jblock(uint64_t x) {
    return uint32_t((x >> HIGH_ORBIT_JBLOCK_SHIFT) & HIGH_ORBIT_BLOCK_MASK);
}
__host__ __device__ static inline uint32_t high_orbit_dhr(uint64_t x) {
    return uint32_t((x >> HIGH_ORBIT_DHR_SHIFT) & HIGH_ORBIT_RANK_MASK);
}
__host__ __device__ static inline uint32_t high_orbit_dblock(uint64_t x) {
    return uint32_t((x >> HIGH_ORBIT_DBLOCK_SHIFT) & HIGH_ORBIT_BLOCK_MASK);
}

struct HighOrbitHost {
    std::vector<uint64_t> rec;
    std::array<uint32_t, 64> main_base{};
    uint32_t main_total = 0;
    uint64_t orbit_sources = 0;
    uint64_t closures = 0;
};

static HighOrbitHost build_high_orbit(
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    HighOrbitHost o;
    uint32_t total = 0;
    for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
        o.main_base[bid] = total;
        total += layout.main_blocks[bid].rows;
    }
    o.main_total = total;
    o.rec.assign(size_t(total) * H, 0);

    auto representative_low = [&](int hs) -> uint32_t {
        uint32_t a = storage.low_all_off[hs];
        uint32_t b = storage.low_all_off[hs + 1];
        return a < b ? storage.low_all_codes[a] : 0xffffffffu;
    };

    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            uint32_t lc = representative_low(sb.hs);
            if (lc == 0xffffffffu) continue;
            uint32_t high0 = storage.high_all_off[sb.he];

            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                uint32_t hc = storage.high_all_codes[high0 + hr];
                MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                MateValue a = mget(m, p);
                MateValue b = mget(m, p - 1);
                uint64_t word = 0;

                if (a == N) {
                    HighOrbitKind kind = HIGH_ORBIT_NONE;
                    MateValuePair pair = NN;
                    if (b == N) { kind = HIGH_ORBIT_NN; pair = LR; }
                    else if (b == R) { kind = HIGH_ORBIT_NR; pair = RN; }
                    else if (b == ::L) { kind = HIGH_ORBIT_NL; pair = LN; }

                    if (kind != HIGH_ORBIT_NONE) {
                        MateID jm = msetpair(m, p, pair);
                        uint32_t jlc = uint32_t(jm) & LM;
                        if (jlc != lc) {
                            std::cerr << "high orbit partner unexpectedly changes LOW\n";
                            std::exit(161);
                        }
                        uint32_t jhc = uint32_t((jm >> (2 * (L + 1))) & HM);
                        uint32_t jp = storage.high_packed_rank[jhc];
                        if (jp == 0xffffffffu) std::exit(162);
                        uint32_t jhr = jp >> H;
                        int jhe = seg_end_height_host(jhc, H);
                        int jcv = int(mget(jm, L));
                        uint32_t jbid = uint32_t(3 * jhe + jcv);
                        if (jbid >= layout.main_blocks.size()
                            || jhr >= layout.main_blocks[jbid].rows) {
                            std::cerr << "high orbit partner block mismatch\n";
                            std::exit(163);
                        }

                        // All owner states have N at position p.  Removing that
                        // N is exactly the inverse of blocked_exclude's insert.
                        MateID dm = mshrink(m, p);
                        uint32_t dlc = uint32_t(dm) & LM;
                        if (dlc != lc) {
                            std::cerr << "high orbit drop unexpectedly changes LOW\n";
                            std::exit(164);
                        }
                        uint32_t dhc = uint32_t((dm >> (2 * L)) & HM);
                        uint32_t dp = storage.high_packed_rank[dhc];
                        if (dp == 0xffffffffu) std::exit(165);
                        uint32_t dhr = dp >> H;
                        int dh = seg_end_height_host(dhc, H);
                        uint32_t dbid = uint32_t(dh);
                        if (dbid >= layout.block_blocks.size()
                            || dhr >= layout.block_blocks[dbid].rows) {
                            std::cerr << "high orbit dropped block mismatch\n";
                            std::exit(166);
                        }
                        word = high_orbit_pack(kind, jbid, jhr, dbid, dhr);
                        ++o.orbit_sources;
                    }
                } else if ((a == ::L && b == ::L) || (a == R && b == R)
                           || (a == R && b == ::L)) {
                    word = high_orbit_pack(HIGH_ORBIT_CLOSURE);
                    ++o.closures;
                }
                o.rec[size_t(pi) * o.main_total + o.main_base[bid] + hr] = word;
            }
        }
    }

    std::cerr << "high_orbit mib="
              << double(o.rec.size() * sizeof(uint64_t)) / double(1 << 20)
              << " orbit_sources=" << o.orbit_sources
              << " closures=" << o.closures << '\n';
    return o;
}

__constant__ uint64_t* D_HIGH_ORBIT;
__constant__ uint32_t D_HIGH_ORBIT_MAIN_BASE[64];
__constant__ uint32_t D_HIGH_ORBIT_MAIN_TOTAL;

struct HighOrbitDeviceTables {
    uint64_t* rec = nullptr;

    void install(const HighOrbitHost& o) {
        if (!o.rec.empty()) {
            ck(cudaMalloc(&rec, o.rec.size() * sizeof(uint64_t)), "high orbit alloc");
            ck(cudaMemcpy(rec, o.rec.data(), o.rec.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "high orbit copy");
        }
        ck(cudaMemcpyToSymbol(D_HIGH_ORBIT, &rec, sizeof(rec)), "high orbit ptr");
        ck(cudaMemcpyToSymbol(D_HIGH_ORBIT_MAIN_BASE, o.main_base.data(),
                              sizeof(uint32_t) * o.main_base.size()), "high orbit bases");
        ck(cudaMemcpyToSymbol(D_HIGH_ORBIT_MAIN_TOTAL, &o.main_total,
                              sizeof(o.main_total)), "high orbit total");
    }

    void release() {
        if (rec) cudaFree(rec);
        rec = nullptr;
    }
};

__device__ __forceinline__ Count high_orbit_add(Count a, Count b) {
    if (!b) return a;
    Count mod = D_MOD;
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

// HIGH window is p >= LOW_LUT_K+1 > 1.  Thus the p=1 special orbit does not
// exist here.  Each N* owner consumes exactly one blocked state and pairs with
// one main state while preserving the LOW column.
__global__ void main_group_high_orbit_inplace_kernel(
    Count* mainv, Code n, Count* blockv, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (; i < n; i += step) {
        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint64_t ow = D_HIGH_ORBIT[size_t(pi) * D_HIGH_ORBIT_MAIN_TOTAL
                                  + D_HIGH_ORBIT_MAIN_BASE[bid] + hr];
        uint32_t kind = high_orbit_kind(ow);
        if (kind < HIGH_ORBIT_NN || kind > HIGH_ORBIT_NL) continue;

        FBlock jy = D_F_MAIN_BLOCKS[high_orbit_jblock(ow)];
        FBlock dy = D_F_BLOCK_BLOCKS[high_orbit_dblock(ow)];
        Code j = jy.off + Code(high_orbit_jhr(ow)) * jy.stride + lr;
        Code dj = dy.off + Code(high_orbit_dhr(ow)) * dy.stride + lr;
        Count c = mainv[i];
        Count d = blockv[dj];

        if (kind == HIGH_ORBIT_NN) {
            mainv[j] = high_orbit_add(mainv[j], c); // injective NN -> LR
            mainv[i] = high_orbit_add(c, d);
            blockv[dj] = 0;
        } else {
            Count cc = mainv[j];
            mainv[i] = high_orbit_add(high_orbit_add(c, cc), d);
            blockv[dj] = c;
            // RN/LN partner keeps its identity value in mainv[j].
        }
    }
}

// LL/RR/RL closure sources are marked by the orbit table.  Reuse the compact
// HIGH descriptor for the actual blocked destination, including boundary-cross
// LOW mate flips.  This pass runs after the orbit pass on the same CUDA stream.
__global__ void main_group_high_closure_inplace_kernel(
    Count* mainv, Code n, Count* blockv, int p
) {
    constexpr int S = MAXW + 2;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (; i < n; i += step) {
        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint64_t ow = D_HIGH_ORBIT[size_t(pi) * D_HIGH_ORBIT_MAIN_TOTAL
                                  + D_HIGH_ORBIT_MAIN_BASE[bid] + hr];
        if (high_orbit_kind(ow) != HIGH_ORBIT_CLOSURE) continue;
        Count c = mainv[i];
        if (!c) continue;

        uint32_t desc = D_HIGHDESC_MAIN[size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                                       + D_HIGHDESC_MAIN_BASE[bid] + hr];
        uint32_t kind = highdesc_kind(desc);
        if (kind == HIGHDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            atomic_add_mod(blockv + y.off + Code(highdesc_rank(desc)) * y.stride + lr, c);
        } else if (kind == HIGHDESC_CROSS) {
            uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
            uint32_t lc = D_F_LOW_MASK_CODES[a + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            uint32_t lr2 = bidesc_low_mask_rank(lc2, y.hs);
            if (lr2 == 0xffffffffu) continue;
            atomic_add_mod(blockv + y.off + Code(highdesc_rank(desc)) * y.stride + lr2, c);
        } else {
            // A closure source at p>1 must either close locally into blocked or
            // cross the LOW boundary into blocked.  Anything else is a table bug.
            asm("trap;");
        }
    }
}
