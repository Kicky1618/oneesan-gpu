#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef MASKSHARD_LOW_CLOSURE_COLS

struct MaskShardLowClosureColsHost {
    std::vector<uint32_t> cols;
    std::array<uint32_t, MAXW + 2> off{};
    std::vector<uint32_t> block_off;
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
    std::vector<uint32_t> compact_cols;
    std::vector<uint32_t> compact_active_count;
    std::vector<uint16_t> high_compact_rank;
    std::vector<uint16_t> high_active_count;
#endif
};

#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
static int maskshard_lowclosure_peak_host(uint32_t code, int len, int start_h) {
    int h = start_h, peak = h;
    for (int pos = len - 1; pos >= 0; --pos) {
        const uint32_t v = (code >> (2 * pos)) & 3u;
        if (v == uint32_t(R)) --h;
        else if (v == uint32_t(::L)) {
            ++h;
            peak = std::max(peak, h);
        }
    }
    return peak;
}
#endif

static MaskShardLowClosureColsHost build_maskshard_low_closure_cols(
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    const LowDescHost& low_desc
) {
    constexpr int L = LOW_LUT_K;
    MaskShardLowClosureColsHost out;
    out.block_off.assign(size_t(L) * 65, 0u);

    for (int p = L; p >= 1; --p) {
        const uint32_t pi = uint32_t(L - p);
        const uint32_t begin = uint32_t(out.cols.size());
        out.off[pi] = begin;

        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            out.block_off[size_t(pi) * 65 + bid] = uint32_t(out.cols.size());
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            const uint32_t low0 = storage.low_all_off[sb.hs];
            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                const uint32_t lc = storage.low_all_codes[low0 + lr];
                const uint32_t active = lc | (uint32_t(sb.c) << (2 * L));
                const MateValuePair w = MateValuePair(
                    (active >> (2 * (p - 1))) & 15u);
                if (w != LL && w != RR && w != RL) continue;

                const uint32_t desc = low_desc.main_desc[
                    size_t(pi) * low_desc.main_total + low_desc.main_base[bid] + lr];
                const uint32_t kind = desc >> LOWDESC_KIND_SHIFT;
                if (kind == LOWDESC_INVALID) continue;
                out.cols.push_back(lowdesc_pack(
                    LOWDESC_MAIN, uint32_t(bid), lr));
            }
        }
        out.block_off[size_t(pi) * 65 + layout.main_blocks.size()]
            = uint32_t(out.cols.size());
        out.off[pi + 1] = uint32_t(out.cols.size());
    }

    if (out.cols.size() > 0xffffffffULL) {
        std::cerr << "maskshard LOW closure column table exceeds uint32 offsets\n";
        std::exit(170);
    }

#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
    constexpr int H = HIGH_LUT_K;
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr int CAP_STRIDE = FULL_CAP + 1;
    constexpr uint32_t HNM = 1u << H;

    out.compact_cols.resize(out.cols.size());
    out.compact_active_count.assign(size_t(L) * 65 * CAP_STRIDE, 0u);
    std::vector<uint32_t> order;
    for (int pi = 0; pi < L; ++pi) {
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const uint32_t a = out.block_off[size_t(pi) * 65 + bid];
            const uint32_t z = out.block_off[size_t(pi) * 65 + bid + 1];
            if (z < a || z > out.cols.size()) {
                std::cerr << "LOW closure compact invalid column range\n";
                std::exit(171);
            }
            const StorageBlock& sb = layout.main_blocks[bid];
            const uint32_t n = z - a;
            order.resize(n);
            for (uint32_t q = 0; q < n; ++q) order[q] = q;
            const uint32_t low0 = storage.low_all_off[sb.hs];
            std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
                const uint32_t lrx = lowdesc_lr(out.cols[a + x]);
                const uint32_t lry = lowdesc_lr(out.cols[a + y]);
                const int px = maskshard_lowclosure_peak_host(
                    storage.low_all_codes[low0 + lrx], L, sb.hs);
                const int py = maskshard_lowclosure_peak_host(
                    storage.low_all_codes[low0 + lry], L, sb.hs);
                return px != py ? px < py : lrx < lry;
            });
            for (uint32_t q = 0; q < n; ++q)
                out.compact_cols[a + q] = out.cols[a + order[q]];
            for (int cap = 0; cap <= FULL_CAP; ++cap) {
                uint32_t count = 0;
                while (count < n) {
                    const uint32_t lr = lowdesc_lr(out.compact_cols[a + count]);
                    const int pk = maskshard_lowclosure_peak_host(
                        storage.low_all_codes[low0 + lr], L, sb.hs);
                    if (pk > cap) break;
                    ++count;
                }
                out.compact_active_count[
                    (size_t(pi) * 65 + bid) * CAP_STRIDE + size_t(cap)] = count;
            }
            if (out.compact_active_count[
                    (size_t(pi) * 65 + bid) * CAP_STRIDE + FULL_CAP] != n) {
                std::cerr << "LOW closure compact full-cap column mismatch pi="
                          << pi << " bid=" << bid << '\n';
                std::exit(172);
            }
        }
    }

    out.high_compact_rank.resize(G_FACTOR.high_mask_codes.size());
    out.high_active_count.assign(
        size_t(HNM) * (H + 2) * CAP_STRIDE, uint16_t(0));
    for (uint32_t mask = 0; mask < HNM; ++mask) {
        for (int he = 0; he <= H + 1; ++he) {
            const size_t ix = size_t(mask) * S + he;
            const uint32_t a = G_FACTOR.high_mask_off[ix];
            const uint32_t n = factor_count(G_FACTOR.high_mask_off, mask, he);
            if (n > 0xffffu) {
                std::cerr << "LOW closure compact HIGH mask rank exceeds uint16 mask="
                          << mask << " he=" << he << " n=" << n << '\n';
                std::exit(173);
            }
            order.resize(n);
            for (uint32_t r = 0; r < n; ++r) order[r] = r;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
                const int px = maskshard_lowclosure_peak_host(
                    G_FACTOR.high_mask_codes[a + x], H, 1);
                const int py = maskshard_lowclosure_peak_host(
                    G_FACTOR.high_mask_codes[a + y], H, 1);
                return px != py ? px < py : x < y;
            });
            for (uint32_t q = 0; q < n; ++q)
                out.high_compact_rank[a + q] = uint16_t(order[q]);
            for (int cap = 0; cap <= FULL_CAP; ++cap) {
                uint32_t count = 0;
                while (count < n) {
                    const uint32_t r = out.high_compact_rank[a + count];
                    const int pk = maskshard_lowclosure_peak_host(
                        G_FACTOR.high_mask_codes[a + r], H, 1);
                    if (pk > cap) break;
                    ++count;
                }
                out.high_active_count[
                    (size_t(mask) * (H + 2) + size_t(he)) * CAP_STRIDE
                    + size_t(cap)] = uint16_t(count);
            }
            if (out.high_active_count[
                    (size_t(mask) * (H + 2) + size_t(he)) * CAP_STRIDE
                    + FULL_CAP] != n) {
                std::cerr << "LOW closure compact full-cap HIGH mismatch mask="
                          << mask << " he=" << he << '\n';
                std::exit(174);
            }
        }
    }
#endif

    std::cerr << "maskshard low_closure_cols entries=" << out.cols.size()
              << " cols_mib="
              << double(out.cols.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " block_off_kib="
              << double(out.block_off.size() * sizeof(uint32_t)) / 1024.0
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
              << " compact_cols_mib="
              << double(out.compact_cols.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " compact_low_count_kib="
              << double(out.compact_active_count.size() * sizeof(uint32_t)) / 1024.0
              << " compact_high_rank_mib="
              << double(out.high_compact_rank.size() * sizeof(uint16_t)) / double(1ULL << 20)
              << " compact_high_count_mib="
              << double(out.high_active_count.size() * sizeof(uint16_t)) / double(1ULL << 20)
#endif
              << '\n';
    return out;
}

__constant__ const uint32_t* D_MS_LOW_CLOSURE_COLS;
__constant__ const uint32_t* D_MS_LOW_CLOSURE_BLOCK_OFF;
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
__constant__ const uint32_t* D_MS_LOW_CLOSURE_COMPACT_COLS;
__constant__ const uint32_t* D_MS_LOW_CLOSURE_COMPACT_ACTIVE_COUNT;
__constant__ const uint16_t* D_MS_LOW_CLOSURE_HIGH_COMPACT_RANK;
__constant__ const uint16_t* D_MS_LOW_CLOSURE_HIGH_ACTIVE_COUNT;
#endif

struct MaskShardLowClosureColsDeviceTables {
    uint32_t* cols = nullptr;
    uint32_t* block_off = nullptr;
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
    uint32_t* compact_cols = nullptr;
    uint32_t* compact_active_count = nullptr;
    uint16_t* high_compact_rank = nullptr;
    uint16_t* high_active_count = nullptr;
#endif

    void install(const MaskShardLowClosureColsHost& h) {
        if (!h.cols.empty()) {
            ck(cudaMalloc(&cols, h.cols.size() * sizeof(uint32_t)),
               "maskshard LOW closure cols alloc");
            ck(cudaMemcpy(cols, h.cols.data(), h.cols.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure cols copy");
        }
        if (!h.block_off.empty()) {
            ck(cudaMalloc(&block_off,
                          h.block_off.size() * sizeof(uint32_t)),
               "maskshard LOW closure block off alloc");
            ck(cudaMemcpy(block_off, h.block_off.data(),
                          h.block_off.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure block off copy");
        }
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
        if (!h.compact_cols.empty()) {
            ck(cudaMalloc(&compact_cols, h.compact_cols.size() * sizeof(uint32_t)),
               "maskshard LOW closure compact cols alloc");
            ck(cudaMemcpy(compact_cols, h.compact_cols.data(),
                          h.compact_cols.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure compact cols copy");
        }
        if (!h.compact_active_count.empty()) {
            ck(cudaMalloc(&compact_active_count,
                          h.compact_active_count.size() * sizeof(uint32_t)),
               "maskshard LOW closure compact counts alloc");
            ck(cudaMemcpy(compact_active_count, h.compact_active_count.data(),
                          h.compact_active_count.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure compact counts copy");
        }
        if (!h.high_compact_rank.empty()) {
            ck(cudaMalloc(&high_compact_rank,
                          h.high_compact_rank.size() * sizeof(uint16_t)),
               "maskshard LOW closure HIGH compact rank alloc");
            ck(cudaMemcpy(high_compact_rank, h.high_compact_rank.data(),
                          h.high_compact_rank.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure HIGH compact rank copy");
        }
        if (!h.high_active_count.empty()) {
            ck(cudaMalloc(&high_active_count,
                          h.high_active_count.size() * sizeof(uint16_t)),
               "maskshard LOW closure HIGH active count alloc");
            ck(cudaMemcpy(high_active_count, h.high_active_count.data(),
                          h.high_active_count.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure HIGH active count copy");
        }
#endif
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_COLS, &cols, sizeof(cols)),
           "maskshard LOW closure cols ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_BLOCK_OFF, &block_off,
                              sizeof(block_off)),
           "maskshard LOW closure block off ptr");
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_COMPACT_COLS,
                              &compact_cols, sizeof(compact_cols)),
           "maskshard LOW closure compact cols ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_COMPACT_ACTIVE_COUNT,
                              &compact_active_count, sizeof(compact_active_count)),
           "maskshard LOW closure compact counts ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_HIGH_COMPACT_RANK,
                              &high_compact_rank, sizeof(high_compact_rank)),
           "maskshard LOW closure HIGH compact rank ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_HIGH_ACTIVE_COUNT,
                              &high_active_count, sizeof(high_active_count)),
           "maskshard LOW closure HIGH active count ptr");
#endif
    }

    void release() {
        if (cols) cudaFree(cols);
        if (block_off) cudaFree(block_off);
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
        if (compact_cols) cudaFree(compact_cols);
        if (compact_active_count) cudaFree(compact_active_count);
        if (high_compact_rank) cudaFree(high_compact_rank);
        if (high_active_count) cudaFree(high_active_count);
        compact_cols = compact_active_count = nullptr;
        high_compact_rank = high_active_count = nullptr;
#endif
        cols = block_off = nullptr;
    }
};

#endif
