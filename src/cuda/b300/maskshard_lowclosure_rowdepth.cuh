#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH
#error "maskshard_lowclosure_rowdepth.cuh requires MASKSHARD_LOW_CLOSURE_ROW_DEPTH"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_COLS
#error "LOW closure row-depth pruning requires v0.9 closure columns"
#endif
#ifndef MASKSHARD_ROW_DEPTH_EXACT_IO
#error "LOW closure row-depth pruning reuses v0.15 exact HIGH peaks"
#endif
#ifndef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#error "v0.24 currently layers on the v0.19+ setup chain"
#endif

// LOW closure is executed in fixed-HIGH-mask groups. Its local LOW coordinate is
// a rank in occupancy-major LOW-all storage order, not in D_F_LOW_MASK_CODES
// order. Keep one extra byte per LOW-all code so source depth can be checked
// without reconstructing a 14-symbol path in the hot closure kernel.
__device__ __constant__ std::uint8_t* D_MS_LOW_CLOSURE_ROW_DEPTH_LOW_ALL_PEAK;

struct MaskShardLowClosureRowDepthCache {
    std::vector<std::uint8_t> low_all_peak;
    std::array<std::uint8_t*, 8> d_low_all_peak{};
    std::array<bool, 8> installed{};
    bool built = false;

    void build() {
        if (built) return;
        constexpr int L = LOW_LUT_K;
        constexpr int S = FactorTablesHost::STRIDE;
        constexpr std::uint32_t NM = 1u << L;

        low_all_peak.assign(G_FACTOR.low_all_codes.size(), 0xffu);
        for (int hs = 0; hs <= L + 1; ++hs) {
            std::uint32_t dst = G_FACTOR.low_all_off[hs];
            for (std::uint32_t mask = 0; mask < NM; ++mask) {
                const std::size_t ix = std::size_t(mask) * S + hs;
                const std::uint32_t a = G_FACTOR.low_mask_off[ix];
                const std::uint32_t z = G_FACTOR.low_mask_off[ix + 1];
                for (std::uint32_t qi = a; qi < z; ++qi) {
                    if (dst >= low_all_peak.size()) {
                        std::cerr << "LOW closure row-depth storage-order overflow hs="
                                  << hs << '\n';
                        std::exit(300);
                    }
                    low_all_peak[dst++] = MaskShardRowDepthExactCache::peak_code(
                        G_FACTOR.low_mask_codes[qi], L, hs);
                }
            }
            if (dst != G_FACTOR.low_all_off[hs + 1]) {
                std::cerr << "LOW closure row-depth storage-order count mismatch hs="
                          << hs << " got=" << dst
                          << " expected=" << G_FACTOR.low_all_off[hs + 1] << '\n';
                std::exit(301);
            }
        }
        for (std::uint8_t x : low_all_peak) if (x == 0xffu) {
            std::cerr << "LOW closure row-depth LOW-all peak has unfilled entry\n";
            std::exit(302);
        }
        built = true;
        std::cerr << "LOW closure row-depth metadata low_all_entries="
                  << low_all_peak.size() << " mib="
                  << double(low_all_peak.size()) / double(1ULL << 20) << '\n';
    }

    void install_current_device() {
        build();
        int dev = -1;
        ck(cudaGetDevice(&dev), "LOW closure row-depth get device");
        if (dev < 0 || dev >= int(installed.size())) {
            std::cerr << "LOW closure row-depth unsupported device " << dev << '\n';
            std::exit(303);
        }
        if (installed[dev]) return;
        if (!low_all_peak.empty()) {
            ck(cudaMalloc(&d_low_all_peak[dev],
                          low_all_peak.size() * sizeof(std::uint8_t)),
               "LOW closure row-depth alloc LOW-all peaks");
            ck(cudaMemcpy(d_low_all_peak[dev], low_all_peak.data(),
                          low_all_peak.size() * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "LOW closure row-depth copy LOW-all peaks");
        }
        ck(cudaMemcpyToSymbol(D_MS_LOW_CLOSURE_ROW_DEPTH_LOW_ALL_PEAK,
                              &d_low_all_peak[dev], sizeof(d_low_all_peak[dev])),
           "LOW closure row-depth LOW-all peak ptr");
        installed[dev] = true;
    }

    void release() {
        for (int dev = 0; dev < int(installed.size()); ++dev) {
            if (!installed[dev]) continue;
            cudaSetDevice(dev);
            if (d_low_all_peak[dev]) cudaFree(d_low_all_peak[dev]);
            d_low_all_peak[dev] = nullptr;
            installed[dev] = false;
        }
    }
};

static MaskShardLowClosureRowDepthCache& maskshard_lowclosure_rowdepth_cache() {
    static MaskShardLowClosureRowDepthCache cache;
    return cache;
}

// Preserve the v0.19 setup chain and append the v0.24 LOW-all peak upload.
#ifdef report_high_mask_shard_layout
#undef report_high_mask_shard_layout
#endif
static void maskshard_report_high_mask_shard_layout_lowclosure_rowdepth(
    const MaskShardLayout& s
) {
    maskshard_report_high_mask_shard_layout_orbit_compact(s);
    auto& cache = maskshard_lowclosure_rowdepth_cache();
    cache.build();
    for (int d = 0; d < s.ngpu; ++d) {
        ck(cudaSetDevice(d), "LOW closure row-depth setup device");
        cache.install_current_device();
    }
}
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_lowclosure_rowdepth

static void maskshard_release_lowclosure_rowdepth() {
    maskshard_lowclosure_rowdepth_cache().release();
}

__global__ void maskshard_main_lowdesc_closure_cols_rowdepth_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr std::uint32_t HR_MASK = (1u << HIGH_LUT_K) - 1u;
    __shared__ Code prefix[65];
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    const int nb = D_F_MAIN_NBLOCKS;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;

    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int b = 0; b < nb; ++b) {
            const FBlock x = D_F_MAIN_BLOCKS[b];
            const std::uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * 65 + b];
            const std::uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * 65 + b + 1];
            const std::uint32_t chunks = (z - a + 31u) >> 5;
            const Code rows = x.stride ? (x.end - x.off) / x.stride : 0;
            prefix[b + 1] = prefix[b] + rows * Code(chunks);
        }
    }
    __syncthreads();

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int((blockDim.x + 31) >> 5);
    Code task = Code(blockIdx.x) * Code(warps_per_block) + Code(warp_in_block);
    const Code task_step = Code(gridDim.x) * Code(warps_per_block);
    const Code total = prefix[nb];

    for (; task < total; task += task_step) {
        int bid = 0;
        Code local = 0;
        if (lane == 0) {
            int lo = 0, hi = nb + 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (prefix[mid] <= task) lo = mid + 1;
                else hi = mid;
            }
            bid = lo - 1;
            local = task - prefix[bid];
        }
        bid = __shfl_sync(active, bid, 0);
        const std::uint32_t local_lo = __shfl_sync(active, std::uint32_t(local), 0);
        const std::uint32_t local_hi = __shfl_sync(active, std::uint32_t(local >> 32), 0);
        local = Code(local_lo) | (Code(local_hi) << 32);

        const std::uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[
            std::size_t(pi) * 65 + bid];
        const std::uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[
            std::size_t(pi) * 65 + bid + 1];
        const std::uint32_t chunks = (z - a + 31u) >> 5;
        if (!chunks) continue;
        const std::uint32_t hr = std::uint32_t(local / chunks);
        const std::uint32_t chunk = std::uint32_t(local - Code(hr) * chunks);
        const std::uint32_t qi = a + (chunk << 5) + std::uint32_t(lane);
        if (qi >= z) continue;

        const std::uint32_t source = D_MS_LOW_CLOSURE_COLS[qi];
        const std::uint32_t lr = lowdesc_lr(source);
        const FBlock x = D_F_MAIN_BLOCKS[bid];

        if (!saturated) {
            if (int(x.he) > cap || int(x.hs) > cap) continue;
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(D_F_MASK) * S + x.he];
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
            const std::uint32_t packed = D_F_HIGH_PACKED_RANK[hc];
            if (packed == 0xffffffffu) continue;
            const std::uint32_t high_storage_rank = packed >> HIGH_LUT_K;
            const std::uint32_t hi = D_F_HIGH_ALL_OFF[x.he] + high_storage_rank;
            const std::uint32_t lo = D_F_LOW_ALL_OFF[x.hs] + lr;
            const int hp = int(D_MS_ROW_DEPTH_HIGH_PEAK[hi]);
            const int lp = int(D_MS_LOW_CLOSURE_ROW_DEPTH_LOW_ALL_PEAK[lo]);
            if ((hp > lp ? hp : lp) > cap) continue;
        }

        const Code i = x.off + Code(hr) * x.stride + lr;
        const Count c = mainv[i];
        if (!c) continue;

        const std::uint32_t desc = D_LOWDESC_MAIN[
            std::size_t(pi) * D_LOWDESC_MAIN_TOTAL + D_LOWDESC_MAIN_BASE[bid] + lr];
        const std::uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_MAIN) {
            const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(mainv + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            const FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(blockv + j, c);
        } else if (kind == LOWDESC_CROSS) {
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(D_F_MASK) * S + x.he];
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
            const std::uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            const std::uint32_t hp = D_F_HIGH_PACKED_RANK[hc2];
            const std::uint32_t hr2 = hp & HR_MASK;
            if (p == 1) {
                const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(mainv + j, c);
            } else {
                const FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(blockv + j, c);
            }
        }
    }
    (void)n;
}

#ifdef maskshard_main_lowdesc_closure_inplace_kernel
#undef maskshard_main_lowdesc_closure_inplace_kernel
#endif
#define maskshard_main_lowdesc_closure_inplace_kernel \
        maskshard_main_lowdesc_closure_cols_rowdepth_inplace_kernel
