#pragma once

#include "ramstream32_bidesc_compact.cuh"

// Compact canonical-shard I/O for the B300 resident backend.  A factorized
// local index already tells us its factor block plus the row/column ranks.  We
// therefore do not reconstruct MateID during gather/scatter.  For the factor
// whose occupancy is fixed, only a mask-local-rank -> all-rank permutation is
// required.  These two compact arrays replace the 4^LOW/4^HIGH dense ranks.
__constant__ uint32_t* D_CF_LOW_MASK_ALL_RANK;
__constant__ uint32_t* D_CF_HIGH_MASK_ALL_RANK;

struct CompactCanonicalRankHost {
    std::vector<uint32_t> low_mask_all_rank;
    std::vector<uint32_t> high_mask_all_rank;
};

static CompactCanonicalRankHost build_compact_canonical_ranks() {
    CompactCanonicalRankHost t;
    t.low_mask_all_rank.resize(G_FACTOR.low_mask_codes.size());
    for (size_t i = 0; i < G_FACTOR.low_mask_codes.size(); ++i) {
        uint32_t code = G_FACTOR.low_mask_codes[i];
        uint32_t p = G_FACTOR.low_packed_rank[code];
        if (p == 0xffffffffu) std::exit(270);
        t.low_mask_all_rank[i] = p >> LOW_LUT_K;
    }
    t.high_mask_all_rank.resize(G_FACTOR.high_mask_codes.size());
    for (size_t i = 0; i < G_FACTOR.high_mask_codes.size(); ++i) {
        uint32_t code = G_FACTOR.high_mask_codes[i];
        uint32_t p = G_FACTOR.high_packed_rank[code];
        if (p == 0xffffffffu) std::exit(271);
        t.high_mask_all_rank[i] = p >> HIGH_LUT_K;
    }
    std::cerr << "compact canonical rank maps low_mib="
              << double(t.low_mask_all_rank.size() * sizeof(uint32_t)) / (1 << 20)
              << " high_mib="
              << double(t.high_mask_all_rank.size() * sizeof(uint32_t)) / (1 << 20)
              << '\n';
    return t;
}

struct CompactCanonicalDeviceTables {
    uint32_t* low_all_rank = nullptr;
    uint32_t* high_all_rank = nullptr;
    Code* high_main_base = nullptr;
    Code* high_block_base = nullptr;

    template<class T>
    static void upload(T** dst, const std::vector<T>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(dst, v.size() * sizeof(T)), what);
        ck(cudaMemcpy(*dst, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(const CompactCanonicalRankHost& t) {
        upload(&low_all_rank, t.low_mask_all_rank, "compact low mask all-rank");
        upload(&high_all_rank, t.high_mask_all_rank, "compact high mask all-rank");
        upload(&high_main_base, G_FACTOR.high_main_base, "compact high main base");
        upload(&high_block_base, G_FACTOR.high_block_base, "compact high block base");
        ck(cudaMemcpyToSymbol(D_CF_LOW_MASK_ALL_RANK, &low_all_rank, sizeof(low_all_rank)),
           "compact low all-rank ptr");
        ck(cudaMemcpyToSymbol(D_CF_HIGH_MASK_ALL_RANK, &high_all_rank, sizeof(high_all_rank)),
           "compact high all-rank ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MAIN_BASE, &high_main_base, sizeof(high_main_base)),
           "compact high main base ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_BLOCK_BASE, &high_block_base, sizeof(high_block_base)),
           "compact high block base ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF, G_FACTOR.low_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "compact low all off");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF, G_FACTOR.high_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "compact high all off");
        ck(cudaMemcpyToSymbol(D_FULL_DP, H_DP, sizeof(H_DP)), "compact full dp");
    }

    size_t bytes(const CompactCanonicalRankHost& t) const {
        return (t.low_mask_all_rank.size() + t.high_mask_all_rank.size()) * sizeof(uint32_t)
             + (G_FACTOR.high_main_base.size() + G_FACTOR.high_block_base.size()) * sizeof(Code);
    }

    void release() {
        if (low_all_rank) cudaFree(low_all_rank);
        if (high_all_rank) cudaFree(high_all_rank);
        if (high_main_base) cudaFree(high_main_base);
        if (high_block_base) cudaFree(high_block_base);
        low_all_rank = high_all_rank = nullptr;
        high_main_base = high_block_base = nullptr;
    }
};

__device__ __forceinline__ void compact_factor_all_ranks(
    const FBlock& x, uint32_t hr, uint32_t lr, uint32_t& har, uint32_t& lar
) {
    constexpr int S = MAXW + 2;
    if (D_F_FIX_LOW) {
        har = hr;
        uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
        lar = D_CF_LOW_MASK_ALL_RANK[a + lr];
    } else {
        uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
        har = D_CF_HIGH_MASK_ALL_RANK[a + hr];
        lar = lr;
    }
}

__device__ __forceinline__ Code compact_factor_global_main(Code i) {
    int bid = f_find_main(i);
    FBlock x = D_F_MAIN_BLOCKS[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t har, lar;
    compact_factor_all_ranks(x, hr, lr, har, lar);
    Code rank = D_F_HIGH_MAIN_BASE[D_F_HIGH_ALL_OFF[x.he] + har];
    MateValue c = MateValue(x.c);
    if (c > N) rank += D_FULL_DP[LOW_LUT_K][x.he];
    if (c > R && x.he > 0) rank += D_FULL_DP[LOW_LUT_K][x.he - 1];
    return rank + lar;
}

__device__ __forceinline__ Code compact_factor_global_block(Code i) {
    int bid = f_find_block(i);
    FBlock x = D_F_BLOCK_BLOCKS[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t har, lar;
    compact_factor_all_ranks(x, hr, lr, har, lar);
    return D_F_HIGH_BLOCK_BASE[D_F_HIGH_ALL_OFF[x.he] + har] + lar;
}

__global__ void compact_gather_main_kernel(Count* out, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) out[i] = global_load_main(compact_factor_global_main(i));
}
__global__ void compact_gather_block_kernel(Count* out, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) out[i] = global_load_block(compact_factor_global_block(i));
}
__global__ void compact_scatter_main_kernel(const Count* in, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) global_store_main(compact_factor_global_main(i), in[i]);
}
__global__ void compact_scatter_block_kernel(const Count* in, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) global_store_block(compact_factor_global_block(i), in[i]);
}
