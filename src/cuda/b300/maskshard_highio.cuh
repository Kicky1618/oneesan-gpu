#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "maskshard_index.cuh"

// Direct HIGH-window I/O for HIGH-mask-sharded authoritative HBM.

__constant__ Count* D_MS_MAIN_PTR[8];
__constant__ Count* D_MS_BLOCK_PTR[8];
__constant__ const uint8_t* D_MS_OWNER;
__constant__ const Code* D_MS_MAIN_BASE;
__constant__ const Code* D_MS_BLOCK_BASE;
__constant__ const Code* D_MS_MAIN_BLOCK_OFF;
__constant__ const Code* D_MS_BLOCK_BLOCK_OFF;
__constant__ const uint32_t* D_MS_HIGH_ROUTE;
__constant__ const uint32_t* D_MS_LOW_BEGIN;
__constant__ uint32_t D_MS_MAIN_NBLOCKS;
__constant__ uint32_t D_MS_BLOCK_NBLOCKS;
__constant__ uint32_t D_MS_MAIN_COLS[64];
__constant__ uint32_t D_MS_BLOCK_COLS[32];

#ifdef MASKSHARD_ORBIT_AUX
static constexpr uint32_t MS_ORBIT_AUX_RANK_MASK = (1u << 20) - 1u;
static constexpr uint32_t MS_ORBIT_AUX_BLOCK_MASK = 0x3fu;
static constexpr int MS_ORBIT_AUX_BLOCK_SHIFT = 20;
static constexpr int MS_ORBIT_AUX_KIND_SHIFT = 30;
enum MaskShardOrbitAuxKind : uint32_t {
    MS_ORBIT_AUX_INVALID = 0,
    MS_ORBIT_AUX_NN = 1,
    MS_ORBIT_AUX_PAIR = 2,
};
static inline uint32_t maskshard_orbit_aux_pack(
    MaskShardOrbitAuxKind kind, uint32_t block, uint32_t rank
) {
    if (block > MS_ORBIT_AUX_BLOCK_MASK || rank > MS_ORBIT_AUX_RANK_MASK) {
        std::cerr << "maskshard orbit aux overflow block=" << block
                  << " rank=" << rank << '\n';
        std::exit(150);
    }
    return (uint32_t(kind) << MS_ORBIT_AUX_KIND_SHIFT)
        | (block << MS_ORBIT_AUX_BLOCK_SHIFT) | rank;
}
__constant__ const uint32_t* D_MS_HIGH_ORBIT_AUX;
__constant__ const uint32_t* D_MS_LOW_ORBIT_AUX;
#endif

struct MaskShardDeviceMeta {
    int dev = -1;
    uint8_t* owner = nullptr;
    Code* main_base = nullptr;
    Code* block_base = nullptr;
    Code* main_block_off = nullptr;
    Code* block_block_off = nullptr;
    uint32_t* high_route = nullptr;
    uint32_t* low_begin = nullptr;
#ifdef MASKSHARD_ORBIT_AUX
    uint32_t* high_orbit_aux = nullptr;
    uint32_t* low_orbit_aux = nullptr;
#endif

    template<class T>
    static void copy_vec(T** dst, const std::vector<T>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(dst, v.size() * sizeof(T)), what);
        ck(cudaMemcpy(*dst, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

#ifdef MASKSHARD_ORBIT_AUX
    struct OrbitAuxHostCache {
        std::vector<uint32_t> high_aux;
        std::vector<uint32_t> low_aux;
    };

    static const OrbitAuxHostCache& orbit_aux_host(const StorageLayout& layout) {
        static OrbitAuxHostCache cache;
        static bool built = false;
        if (built) return cache;
        built = true;

        constexpr int L = LOW_LUT_K;
        constexpr int H = HIGH_LUT_K;
        constexpr int S = MAXW + 2;
        constexpr uint32_t LM = (1u << L) - 1u;
        constexpr uint32_t HM = (1u << H) - 1u;
        constexpr uint32_t LC = (1u << (2 * L)) - 1u;
        constexpr uint32_t HC = (1u << (2 * H)) - 1u;
        const uint32_t LNM = 1u << L;
        const uint32_t HNM = 1u << H;

        // Prefixes of occupancy-major storage ranks, independently rebuilt
        // from the base mask tables. These are ranks within one height.
        std::vector<uint32_t> lb(size_t(LNM) * S), hb(size_t(HNM) * S);
        for (int h = 0; h <= L + 1; ++h) {
            uint32_t rank = 0;
            for (uint32_t mask = 0; mask < LNM; ++mask) {
                const size_t ix = size_t(mask) * S + h;
                lb[ix] = rank;
                rank += G_FACTOR.low_mask_off[ix + 1] - G_FACTOR.low_mask_off[ix];
            }
            if (rank != G_FACTOR.low_all_off[h + 1] - G_FACTOR.low_all_off[h]) {
                std::cerr << "maskshard orbit aux LOW prefix mismatch h=" << h << '\n';
                std::exit(151);
            }
        }
        for (int h = 0; h <= H + 1; ++h) {
            uint32_t rank = 0;
            for (uint32_t mask = 0; mask < HNM; ++mask) {
                const size_t ix = size_t(mask) * S + h;
                hb[ix] = rank;
                rank += G_FACTOR.high_mask_off[ix + 1] - G_FACTOR.high_mask_off[ix];
            }
            if (rank != G_FACTOR.high_all_off[h + 1] - G_FACTOR.high_all_off[h]) {
                std::cerr << "maskshard orbit aux HIGH prefix mismatch h=" << h << '\n';
                std::exit(152);
            }
        }

        auto low_storage_rank = [&](uint32_t code, int h) -> uint32_t {
            const uint32_t packed = G_FACTOR.low_packed_rank[code];
            if (packed == 0xffffffffu) return 0xffffffffu;
            const uint32_t mask = seg_occ(code, L);
            return lb[size_t(mask) * S + h] + (packed & LM);
        };
        auto high_storage_rank = [&](uint32_t code, int h) -> uint32_t {
            const uint32_t packed = G_FACTOR.high_packed_rank[code];
            if (packed == 0xffffffffu) return 0xffffffffu;
            const uint32_t mask = seg_occ(code, H);
            return hb[size_t(mask) * S + h] + (packed & HM);
        };

        // Reconstruct storage-order code lists. Only ~2M codes total, unlike
        // the dense 4^L/4^H rank arrays which already exist elsewhere.
        std::vector<uint32_t> low_codes, high_codes;
        low_codes.reserve(G_FACTOR.low_all_codes.size());
        high_codes.reserve(G_FACTOR.high_all_codes.size());
        std::array<uint32_t, MAXW + 2> low_off{}, high_off{};
        for (int h = 0; h <= L + 1; ++h) {
            low_off[h] = uint32_t(low_codes.size());
            for (uint32_t mask = 0; mask < LNM; ++mask) {
                const size_t ix = size_t(mask) * S + h;
                for (uint32_t q = G_FACTOR.low_mask_off[ix]; q < G_FACTOR.low_mask_off[ix + 1]; ++q)
                    low_codes.push_back(G_FACTOR.low_mask_codes[q]);
            }
        }
        low_off[L + 2] = uint32_t(low_codes.size());
        for (int h = 0; h <= H + 1; ++h) {
            high_off[h] = uint32_t(high_codes.size());
            for (uint32_t mask = 0; mask < HNM; ++mask) {
                const size_t ix = size_t(mask) * S + h;
                for (uint32_t q = G_FACTOR.high_mask_off[ix]; q < G_FACTOR.high_mask_off[ix + 1]; ++q)
                    high_codes.push_back(G_FACTOR.high_mask_codes[q]);
            }
        }
        high_off[H + 2] = uint32_t(high_codes.size());

        auto set_pair32 = [](uint32_t active, int q, MateValuePair v) {
            const uint32_t sh = uint32_t(2 * (q - 1));
            return (active & ~(15u << sh)) | (uint32_t(v) << sh);
        };
        auto drop_symbol32 = [](uint32_t active, int q) {
            const uint32_t sh = uint32_t(2 * q);
            const uint32_t lo = active & ((uint32_t(1) << sh) - 1u);
            return lo | ((active & ~((uint32_t(1) << sh) - 1u)) >> 2);
        };

#ifdef MASKSHARD_BLOCK_ORBIT_AUX
        // v0.7: store one word per BLOCKED active coordinate, in exactly the
        // same [p][FBlock active-rank] order as HighDesc/LowDesc.block_desc.
        // block_desc itself yields the representative main state. The aux word
        // only distinguishes NN from NR/NL and, for NR/NL, stores companion.
        std::array<uint32_t, 32> high_base{};
        std::array<uint32_t, 32> low_base{};
        uint32_t high_total = 0, low_total = 0;
        for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
            high_base[bid] = high_total;
            low_base[bid] = low_total;
            high_total += layout.block_blocks[bid].rows;
            low_total += layout.block_blocks[bid].cols;
        }
        cache.high_aux.assign(size_t(high_total) * H, 0u);
        cache.low_aux.assign(size_t(low_total) * L, 0u);

        for (int p = TARGET_W - 1; p >= L + 1; --p) {
            const uint32_t pi = uint32_t((TARGET_W - 1) - p);
            for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
                const StorageBlock& sb = layout.block_blocks[bid];
                if (!sb.valid || !sb.rows || !sb.cols) continue;
                const uint32_t lc = low_codes[low_off[sb.hs]];
                for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                    const uint32_t hc = high_codes[high_off[sb.he] + hr];
                    const MateID blocked = MateID(lc) | (MateID(hc) << (2 * L));
                    const MateID rep = oneesan::gridfp::blocked_exclude(blocked, p);
                    const MateValuePair w = mpair(rep, p);
                    uint32_t aux = 0;
                    if (w == NN) {
                        aux = maskshard_orbit_aux_pack(MS_ORBIT_AUX_NN, 0, 0);
                    } else if (w == NR || w == NL) {
                        const MateValuePair cw = w == NR ? RN : LN;
                        const MateID companion = msetpair(rep, p, cw);
                        const uint32_t hc2 = uint32_t((companion >> (2 * (L + 1))) & HC);
                        const uint32_t cv2 = uint32_t(mget(companion, L));
                        const int he2 = seg_end_height_host(hc2, H);
                        const uint32_t hr2 = high_storage_rank(hc2, he2);
                        aux = maskshard_orbit_aux_pack(
                            MS_ORBIT_AUX_PAIR, uint32_t(3 * he2 + int(cv2)), hr2);
                    } else {
                        std::cerr << "maskshard compact HIGH aux non-orbit pair="
                                  << uint32_t(w) << " p=" << p << '\n';
                        std::exit(153);
                    }
                    cache.high_aux[size_t(pi) * high_total + high_base[bid] + hr] = aux;
                }
            }
        }

        for (int p = L; p >= 1; --p) {
            const uint32_t pi = uint32_t(L - p);
            for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
                const StorageBlock& sb = layout.block_blocks[bid];
                if (!sb.valid || !sb.rows || !sb.cols) continue;
                const uint32_t hc = high_codes[high_off[sb.he]];
                for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                    const uint32_t lc = low_codes[low_off[sb.hs] + lr];
                    const MateID blocked = MateID(lc) | (MateID(hc) << (2 * L));
                    const MateID rep = oneesan::gridfp::blocked_exclude(blocked, p);
                    const MateValuePair w = mpair(rep, p);
                    uint32_t aux = 0;
                    if (w == NN) {
                        aux = maskshard_orbit_aux_pack(MS_ORBIT_AUX_NN, 0, 0);
                    } else if (w == NR || w == NL) {
                        if (p == 1) {
                            // The included target is the companion and already
                            // lives in LowDesc.main_desc; only the pair kind is needed.
                            aux = maskshard_orbit_aux_pack(MS_ORBIT_AUX_PAIR, 0, 0);
                        } else {
                            const MateValuePair cw = w == NR ? RN : LN;
                            const MateID companion = msetpair(rep, p, cw);
                            const uint32_t lc2 = uint32_t(companion) & LC;
                            const uint32_t cv2 = uint32_t(mget(companion, L));
                            const int hs2 = int(sb.he)
                                + (cv2 == uint32_t(::L) ? 1 : cv2 == uint32_t(R) ? -1 : 0);
                            const uint32_t lr2 = low_storage_rank(lc2, hs2);
                            aux = maskshard_orbit_aux_pack(
                                MS_ORBIT_AUX_PAIR,
                                uint32_t(3 * int(sb.he) + int(cv2)), lr2);
                        }
                    } else {
                        std::cerr << "maskshard compact LOW aux non-orbit pair="
                                  << uint32_t(w) << " p=" << p << '\n';
                        std::exit(154);
                    }
                    cache.low_aux[size_t(pi) * low_total + low_base[bid] + lr] = aux;
                }
            }
        }
#else
        std::array<uint32_t, 64> high_base{};
        std::array<uint32_t, 64> low_base{};
        uint32_t high_total = 0, low_total = 0;
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            high_base[bid] = high_total;
            low_base[bid] = low_total;
            high_total += layout.main_blocks[bid].rows;
            low_total += layout.main_blocks[bid].cols;
        }
        cache.high_aux.assign(size_t(high_total) * H, 0u);
        cache.low_aux.assign(size_t(low_total) * L, 0u);

        // v0.5/v0.6 full MAIN-coordinate aux.
        for (int p = TARGET_W - 1; p >= L + 1; --p) {
            const uint32_t pi = uint32_t((TARGET_W - 1) - p);
            const int q = p - L;
            for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
                const StorageBlock& sb = layout.main_blocks[bid];
                if (!sb.valid || !sb.rows || !sb.cols) continue;
                for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                    const uint32_t hc = high_codes[high_off[sb.he] + hr];
                    const uint32_t active = (hc << 2) | uint32_t(sb.c);
                    const MateValuePair w = MateValuePair((active >> (2 * (q - 1))) & 15u);
                    uint32_t aux = 0;
                    if (w == NN) {
                        const uint32_t dropped = drop_symbol32(active, q);
                        const int dh = seg_end_height_host(dropped, H);
                        const uint32_t dr = high_storage_rank(dropped, dh);
                        aux = maskshard_orbit_aux_pack(MS_ORBIT_AUX_NN, uint32_t(dh), dr);
                    } else if (w == NR || w == NL) {
                        const MateValuePair cw = w == NR ? RN : LN;
                        const uint32_t companion = set_pair32(active, q, cw);
                        const uint32_t hc2 = companion >> 2;
                        const uint32_t cv2 = companion & 3u;
                        const int he2 = seg_end_height_host(hc2, H);
                        const uint32_t hr2 = high_storage_rank(hc2, he2);
                        aux = maskshard_orbit_aux_pack(
                            MS_ORBIT_AUX_PAIR, uint32_t(3 * he2 + int(cv2)), hr2);
                    }
                    cache.high_aux[size_t(pi) * high_total + high_base[bid] + hr] = aux;
                }
            }
        }

        for (int p = L; p >= 1; --p) {
            const uint32_t pi = uint32_t(L - p);
            for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
                const StorageBlock& sb = layout.main_blocks[bid];
                if (!sb.valid || !sb.rows || !sb.cols) continue;
                for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                    const uint32_t lc = low_codes[low_off[sb.hs] + lr];
                    const uint32_t active = lc | (uint32_t(sb.c) << (2 * L));
                    const MateValuePair w = MateValuePair((active >> (2 * (p - 1))) & 15u);
                    uint32_t aux = 0;
                    if (w == NN || w == NR || w == NL) {
                        if (w == NN || p == 1) {
                            const uint32_t dropped = drop_symbol32(active, p);
                            const uint32_t lr2 = low_storage_rank(dropped, sb.he);
                            aux = maskshard_orbit_aux_pack(
                                w == NN ? MS_ORBIT_AUX_NN : MS_ORBIT_AUX_PAIR,
                                uint32_t(sb.he), lr2);
                        } else {
                            const MateValuePair cw = w == NR ? RN : LN;
                            const uint32_t companion = set_pair32(active, p, cw);
                            const uint32_t lc2 = companion & LC;
                            const uint32_t cv2 = (companion >> (2 * L)) & 3u;
                            const int hs2 = int(sb.he)
                                + (cv2 == uint32_t(::L) ? 1 : cv2 == uint32_t(R) ? -1 : 0);
                            const uint32_t lr2 = low_storage_rank(lc2, hs2);
                            aux = maskshard_orbit_aux_pack(
                                MS_ORBIT_AUX_PAIR, uint32_t(3 * int(sb.he) + int(cv2)), lr2);
                        }
                    }
                    cache.low_aux[size_t(pi) * low_total + low_base[bid] + lr] = aux;
                }
            }
        }
#endif

        std::cerr << "maskshard orbit_aux"
#ifdef MASKSHARD_BLOCK_ORBIT_AUX
                  << " coordinate=blocked"
#else
                  << " coordinate=main"
#endif
                  << " high_mib="
                  << double(cache.high_aux.size() * sizeof(uint32_t)) / double(1 << 20)
                  << " low_mib="
                  << double(cache.low_aux.size() * sizeof(uint32_t)) / double(1 << 20)
                  << '\n';
        return cache;
    }
#endif

    void install(
        int device,
        const MaskShardLayout& shard,
        const StorageLayout& layout,
        const std::vector<uint32_t>& route,
        Count* const main_ptr[8],
        Count* const block_ptr[8]
    ) {
        dev = device;
        ck(cudaSetDevice(dev), "maskshard meta set device");
        copy_vec(&owner, shard.owner, "maskshard owner");
        copy_vec(&main_base, shard.main_base, "maskshard main base");
        copy_vec(&block_base, shard.block_base, "maskshard block base");
        copy_vec(&main_block_off, shard.main_block_off, "maskshard main block off");
        copy_vec(&block_block_off, shard.block_block_off, "maskshard block block off");
        copy_vec(&high_route, route, "maskshard high route");

        constexpr int S = MAXW + 2;
        constexpr uint32_t NM = 1u << LOW_LUT_K;
        std::vector<uint32_t> lb(size_t(NM) * S, 0u);
        for (int h = 0; h <= LOW_LUT_K + 1; ++h) {
            uint32_t rank = 0;
            for (uint32_t mask = 0; mask < NM; ++mask) {
                const size_t ix = size_t(mask) * S + h;
                lb[ix] = rank;
                rank += G_FACTOR.low_mask_off[ix + 1] - G_FACTOR.low_mask_off[ix];
            }
            const uint32_t expected = G_FACTOR.low_all_off[h + 1] - G_FACTOR.low_all_off[h];
            if (rank != expected) {
                std::cerr << "maskshard LOW storage begin mismatch h=" << h
                          << " got=" << rank << " expected=" << expected << '\n';
                std::exit(129);
            }
        }
        copy_vec(&low_begin, lb, "maskshard low storage begin");
#ifdef MASKSHARD_ORBIT_AUX
        const OrbitAuxHostCache& oa = orbit_aux_host(layout);
        copy_vec(&high_orbit_aux, oa.high_aux, "maskshard high orbit aux");
        copy_vec(&low_orbit_aux, oa.low_aux, "maskshard low orbit aux");
#endif

        ck(cudaMemcpyToSymbol(D_MS_MAIN_PTR, main_ptr, sizeof(Count*) * 8), "maskshard main ptrs");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_PTR, block_ptr, sizeof(Count*) * 8), "maskshard block ptrs");
        ck(cudaMemcpyToSymbol(D_MS_OWNER, &owner, sizeof(owner)), "maskshard owner ptr");
        ck(cudaMemcpyToSymbol(D_MS_MAIN_BASE, &main_base, sizeof(main_base)), "maskshard main base ptr");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_BASE, &block_base, sizeof(block_base)), "maskshard block base ptr");
        ck(cudaMemcpyToSymbol(D_MS_MAIN_BLOCK_OFF, &main_block_off, sizeof(main_block_off)), "maskshard main block off ptr");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_BLOCK_OFF, &block_block_off, sizeof(block_block_off)), "maskshard block block off ptr");
        ck(cudaMemcpyToSymbol(D_MS_HIGH_ROUTE, &high_route, sizeof(high_route)), "maskshard high route ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_BEGIN, &low_begin, sizeof(low_begin)), "maskshard low begin ptr");
#ifdef MASKSHARD_ORBIT_AUX
        ck(cudaMemcpyToSymbol(D_MS_HIGH_ORBIT_AUX, &high_orbit_aux, sizeof(high_orbit_aux)), "maskshard high orbit aux ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_ORBIT_AUX, &low_orbit_aux, sizeof(low_orbit_aux)), "maskshard low orbit aux ptr");
#endif
        ck(cudaMemcpyToSymbol(D_MS_MAIN_NBLOCKS, &shard.main_nblocks, sizeof(shard.main_nblocks)), "maskshard main nblocks");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_NBLOCKS, &shard.block_nblocks, sizeof(shard.block_nblocks)), "maskshard block nblocks");

        std::array<uint32_t, 64> mc{};
        std::array<uint32_t, 32> bc{};
        for (size_t i = 0; i < layout.main_blocks.size(); ++i) mc[i] = layout.main_blocks[i].cols;
        for (size_t i = 0; i < layout.block_blocks.size(); ++i) bc[i] = layout.block_blocks[i].cols;
        ck(cudaMemcpyToSymbol(D_MS_MAIN_COLS, mc.data(), sizeof(mc)), "maskshard main cols");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_COLS, bc.data(), sizeof(bc)), "maskshard block cols");
    }

    void release() {
        if (dev < 0) return;
        cudaSetDevice(dev);
        if (owner) cudaFree(owner);
        if (main_base) cudaFree(main_base);
        if (block_base) cudaFree(block_base);
        if (main_block_off) cudaFree(main_block_off);
        if (block_block_off) cudaFree(block_block_off);
        if (high_route) cudaFree(high_route);
        if (low_begin) cudaFree(low_begin);
#ifdef MASKSHARD_ORBIT_AUX
        if (high_orbit_aux) cudaFree(high_orbit_aux);
        if (low_orbit_aux) cudaFree(low_orbit_aux);
        high_orbit_aux = low_orbit_aux = nullptr;
#endif
        owner = nullptr;
        main_base = block_base = nullptr;
        main_block_off = block_block_off = nullptr;
        high_route = low_begin = nullptr;
        dev = -1;
    }
};

#ifdef MASKSHARD_ORBIT_AUX
__device__ __forceinline__ uint32_t maskshard_orbit_aux_kind(uint32_t x) {
    return x >> MS_ORBIT_AUX_KIND_SHIFT;
}
__device__ __forceinline__ uint32_t maskshard_orbit_aux_block(uint32_t x) {
    return (x >> MS_ORBIT_AUX_BLOCK_SHIFT) & MS_ORBIT_AUX_BLOCK_MASK;
}
__device__ __forceinline__ uint32_t maskshard_orbit_aux_rank(uint32_t x) {
    return x & MS_ORBIT_AUX_RANK_MASK;
}
#endif

__device__ __forceinline__ uint32_t maskshard_low_all_rank(
    uint32_t low_mask, uint32_t hs, uint32_t low_mask_rank
) {
    constexpr int S = MAXW + 2;
    return D_MS_LOW_BEGIN[size_t(low_mask) * S + hs] + low_mask_rank;
}

__device__ __forceinline__ void maskshard_high_route(
    uint32_t he, uint32_t high_all_rank, uint32_t& mask, uint32_t& mask_rank
) {
    constexpr uint32_t HM = (1u << HIGH_LUT_K) - 1u;
    const uint32_t packed = D_MS_HIGH_ROUTE[D_F_HIGH_ALL_OFF[he] + high_all_rank];
    mask = packed & HM;
    mask_rank = packed >> HIGH_LUT_K;
}

__device__ __forceinline__ Count* maskshard_main_addr(
    int bid, uint32_t high_all_rank, uint32_t low_mask_rank
) {
    const FBlock x = D_F_MAIN_BLOCKS[bid];
    uint32_t mask = 0, mr = 0;
    maskshard_high_route(x.he, high_all_rank, mask, mr);
    const uint32_t lar = maskshard_low_all_rank(D_F_MASK, x.hs, low_mask_rank);
    const int owner = D_MS_OWNER[mask];
    const Code off = D_MS_MAIN_BASE[mask]
        + D_MS_MAIN_BLOCK_OFF[size_t(mask) * D_MS_MAIN_NBLOCKS + bid]
        + Code(mr) * D_MS_MAIN_COLS[bid] + lar;
    return D_MS_MAIN_PTR[owner] + off;
}

__device__ __forceinline__ Count* maskshard_block_addr(
    int bid, uint32_t high_all_rank, uint32_t low_mask_rank
) {
    const FBlock x = D_F_BLOCK_BLOCKS[bid];
    uint32_t mask = 0, mr = 0;
    maskshard_high_route(x.he, high_all_rank, mask, mr);
    const uint32_t lar = maskshard_low_all_rank(D_F_MASK, x.hs, low_mask_rank);
    const int owner = D_MS_OWNER[mask];
    const Code off = D_MS_BLOCK_BASE[mask]
        + D_MS_BLOCK_BLOCK_OFF[size_t(mask) * D_MS_BLOCK_NBLOCKS + bid]
        + Code(mr) * D_MS_BLOCK_COLS[bid] + lar;
    return D_MS_BLOCK_PTR[owner] + off;
}

template<bool SCATTER>
__global__ void maskshard_high_main_io_kernel(Count* scratch, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
        Count* p = maskshard_main_addr(bid, hr, lr);
        if constexpr (SCATTER) *p = scratch[i];
        else scratch[i] = *p;
    }
}

template<bool SCATTER>
__global__ void maskshard_high_block_io_kernel(Count* scratch, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) {
        const int bid = f_find_block(i);
        const FBlock x = D_F_BLOCK_BLOCKS[bid];
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
        Count* p = maskshard_block_addr(bid, hr, lr);
        if constexpr (SCATTER) *p = scratch[i];
        else scratch[i] = *p;
    }
}
