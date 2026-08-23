#pragma once

#ifndef MASKSHARD_ROW_DEPTH_EXACT_IO
#error "maskshard_rowdepth_exact_io.cuh requires MASKSHARD_ROW_DEPTH_EXACT_IO"
#endif
#ifndef MASKSHARD_ROW_DEPTH_FBLOCK_IO
#error "exact row-depth I/O layers on the v0.14 FBlock row-cap hook"
#endif

// Exact structural max-height filter using one byte per factor code. HIGH peak
// entries are aligned with D_F_HIGH_ALL_CODES; LOW entries are aligned with
// D_F_LOW_MASK_CODES and are computed with the appropriate LOW starting height.
// A composed state's exact frontier depth is max(high_peak, low_peak), because
// low_peak includes the post-center starting height hs.
__device__ __constant__ std::uint8_t* D_MS_ROW_DEPTH_LOW_PEAK;
__device__ __constant__ std::uint8_t* D_MS_ROW_DEPTH_HIGH_PEAK;

struct MaskShardRowDepthExactCache {
    std::vector<std::uint8_t> low_peak;
    std::vector<std::uint8_t> high_peak;
    std::array<std::uint8_t*, 8> d_low{};
    std::array<std::uint8_t*, 8> d_high{};
    std::array<bool, 8> installed{};
    bool built = false;

    static std::uint8_t peak_code(std::uint32_t code, int len, int start_h) {
        int h = start_h;
        int peak = h;
        for (int p = len - 1; p >= 0; --p) {
            const std::uint32_t v = (code >> (2 * p)) & 3u;
            if (v == std::uint32_t(R)) --h;
            else if (v == std::uint32_t(::L)) {
                ++h;
                peak = std::max(peak, h);
            }
        }
        return std::uint8_t(peak);
    }

    void build() {
        if (built) return;
        constexpr int L = LOW_LUT_K;
        constexpr int H = HIGH_LUT_K;
        constexpr int S = FactorTablesHost::STRIDE;
        constexpr std::uint32_t NM = 1u << L;

        low_peak.assign(G_FACTOR.low_mask_codes.size(), 0xffu);
        for (std::uint32_t mask = 0; mask < NM; ++mask) {
            for (int hs = 0; hs <= L + 1; ++hs) {
                const std::size_t ix = std::size_t(mask) * S + hs;
                const std::uint32_t a = G_FACTOR.low_mask_off[ix];
                const std::uint32_t n = factor_count(G_FACTOR.low_mask_off, mask, hs);
                for (std::uint32_t r = 0; r < n; ++r) {
                    const std::uint32_t qi = a + r;
                    low_peak[qi] = peak_code(G_FACTOR.low_mask_codes[qi], L, hs);
                }
            }
        }
        for (std::uint8_t x : low_peak) if (x == 0xffu) {
            std::cerr << "row-depth exact LOW peak table has unfilled entry\n";
            std::exit(260);
        }

        high_peak.assign(G_FACTOR.high_all_codes.size(), 0xffu);
        for (int he = 0; he <= H + 1; ++he) {
            const std::uint32_t a = G_FACTOR.high_all_off[he];
            const std::uint32_t z = G_FACTOR.high_all_off[he + 1];
            for (std::uint32_t qi = a; qi < z; ++qi)
                high_peak[qi] = peak_code(G_FACTOR.high_all_codes[qi], H, 1);
        }
        for (std::uint8_t x : high_peak) if (x == 0xffu) {
            std::cerr << "row-depth exact HIGH peak table has unfilled entry\n";
            std::exit(261);
        }

        built = true;
        std::cerr << "row-depth exact peak metadata low_entries=" << low_peak.size()
                  << " high_entries=" << high_peak.size()
                  << " mib=" << double(low_peak.size() + high_peak.size())
                                  / double(1ULL << 20) << '\n';
    }

    void install_current_device() {
        build();
        int dev = -1;
        ck(cudaGetDevice(&dev), "row-depth exact get device");
        if (dev < 0 || dev >= int(installed.size())) {
            std::cerr << "row-depth exact unsupported device id " << dev << '\n';
            std::exit(262);
        }
        if (!installed[dev]) {
            if (!low_peak.empty()) {
                ck(cudaMalloc(&d_low[dev], low_peak.size() * sizeof(std::uint8_t)),
                   "row-depth exact alloc LOW peaks");
                ck(cudaMemcpy(d_low[dev], low_peak.data(),
                              low_peak.size() * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "row-depth exact copy LOW peaks");
            }
            if (!high_peak.empty()) {
                ck(cudaMalloc(&d_high[dev], high_peak.size() * sizeof(std::uint8_t)),
                   "row-depth exact alloc HIGH peaks");
                ck(cudaMemcpy(d_high[dev], high_peak.data(),
                              high_peak.size() * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "row-depth exact copy HIGH peaks");
            }
            installed[dev] = true;
        }
        ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_LOW_PEAK, &d_low[dev], sizeof(d_low[dev])),
           "row-depth exact LOW peak ptr");
        ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_HIGH_PEAK, &d_high[dev], sizeof(d_high[dev])),
           "row-depth exact HIGH peak ptr");
    }
};

static MaskShardRowDepthExactCache& maskshard_row_depth_exact_cache() {
    static MaskShardRowDepthExactCache cache;
    return cache;
}

static void maskshard_set_row_depth_exact_io_row(int zero_based_row) {
    maskshard_row_depth_exact_cache().install_current_device();
    ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_INDEX, &zero_based_row,
                          sizeof(zero_based_row)),
       "maskshard exact row-depth row index");
}

#define maskshard_set_row_depth_fblock_io_row maskshard_set_row_depth_exact_io_row

template<bool SCATTER>
__global__ void maskshard_high_main_io_rowdepth_exact_kernel(Count* scratch, Code n) {
    constexpr int S = MAXW + 2;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const int cap = maskshard_row_depth_io_cap(SCATTER);
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        if (int(x.he) > cap || int(x.hs) > cap) {
            if constexpr (!SCATTER) scratch[i] = 0;
            continue;
        }
        std::uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
        const std::uint32_t hi = D_F_HIGH_ALL_OFF[x.he] + hr;
        const std::uint32_t lo = D_F_LOW_MASK_OFF[
            std::size_t(D_F_MASK) * S + x.hs] + lr;
        const int hp = int(D_MS_ROW_DEPTH_HIGH_PEAK[hi]);
        const int lp = int(D_MS_ROW_DEPTH_LOW_PEAK[lo]);
        const int peak = hp > lp ? hp : lp;
        if (peak > cap) {
            if constexpr (!SCATTER) scratch[i] = 0;
            continue;
        }
        Count* p = maskshard_main_addr(bid, hr, lr);
        if constexpr (SCATTER) *p = scratch[i];
        else scratch[i] = *p;
    }
}

template<bool SCATTER>
__global__ void maskshard_high_block_io_rowdepth_exact_kernel(Count* scratch, Code n) {
    constexpr int S = MAXW + 2;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const int cap = maskshard_row_depth_io_cap(SCATTER);
    for (; i < n; i += step) {
        const int bid = f_find_block(i);
        const FBlock x = D_F_BLOCK_BLOCKS[bid];
        if (int(x.he) > cap) {
            if constexpr (!SCATTER) scratch[i] = 0;
            continue;
        }
        if constexpr (!SCATTER) {
            scratch[i] = 0;
        } else {
            std::uint32_t hr = 0, lr = 0;
            maskshard_split_rank(i, x, hr, lr);
            const std::uint32_t hi = D_F_HIGH_ALL_OFF[x.he] + hr;
            const std::uint32_t lo = D_F_LOW_MASK_OFF[
                std::size_t(D_F_MASK) * S + x.he] + lr;
            const int hp = int(D_MS_ROW_DEPTH_HIGH_PEAK[hi]);
            const int lp = int(D_MS_ROW_DEPTH_LOW_PEAK[lo]);
            const int peak = hp > lp ? hp : lp;
            if (peak > cap) continue;
            Count* p = maskshard_block_addr(bid, hr, lr);
            *p = scratch[i];
        }
    }
}

#ifdef maskshard_high_main_io_kernel
#undef maskshard_high_main_io_kernel
#endif
#define maskshard_high_main_io_kernel maskshard_high_main_io_rowdepth_exact_kernel
#ifdef maskshard_high_block_io_kernel
#undef maskshard_high_block_io_kernel
#endif
#define maskshard_high_block_io_kernel maskshard_high_block_io_rowdepth_exact_kernel
