#pragma once

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
};

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
    std::cerr << "maskshard low_closure_cols entries=" << out.cols.size()
              << " cols_mib="
              << double(out.cols.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " block_off_kib="
              << double(out.block_off.size() * sizeof(uint32_t)) / 1024.0
              << '\n';
    return out;
}

__constant__ const uint32_t* D_MS_LOW_CLOSURE_COLS;
__constant__ const uint32_t* D_MS_LOW_CLOSURE_BLOCK_OFF;

struct MaskShardLowClosureColsDeviceTables {
    uint32_t* cols = nullptr;
    uint32_t* block_off = nullptr;

    void install(const MaskShardLowClosureColsHost& h) {
        if (!h.cols.empty()) {
            ck(cudaMalloc(&cols, h.cols.size() * sizeof(uint32_t)),
               "maskshard LOW closure cols alloc");
            ck(cudaMemcpy(cols, h.cols.data(), h.cols.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure cols copy");
        }
        if (!h.block_off.empty()) {
            ck(cudaMalloc(&block_off, h.block_off.size() * sizeof(uint32_t)),
               "maskshard LOW closure block off alloc");
            ck(cudaMemcpy(block_off, h.block_off.data(),
                          h.block_off.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "maskshard LOW closure block off copy");
        }
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_COLS, &cols, sizeof(cols)),
           "maskshard LOW closure cols ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_BLOCK_OFF, &block_off,
                              sizeof(block_off)),
           "maskshard LOW closure block off ptr");
    }

    void release() {
        if (cols) cudaFree(cols);
        if (block_off) cudaFree(block_off);
        cols = block_off = nullptr;
    }
};

#endif
