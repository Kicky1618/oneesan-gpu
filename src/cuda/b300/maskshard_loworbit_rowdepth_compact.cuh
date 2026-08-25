#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "maskshard_loworbit_rowdepth_compact.cuh requires compact LOW orbit macro"
#endif
#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH
#error "exact LOW orbit tasks layer on v0.28 row-depth semantics"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
#error "exact LOW orbit tasks reuse v0.26 HIGH compact ranks"
#endif
#ifndef MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH
#error "exact LOW orbit tasks layer on v0.29 BLOCKED-domain launch"
#endif
static_assert(LOW_LUT_K == HIGH_LUT_K + 1,
              "v0.30 compact LOW orbit currently assumes LOW=HIGH+1");

// v0.30 exact active BLOCKED coordinates factor as
//
//   {HIGH mask-local ranks with peak <= cap}
//       x
//   {LOW-all storage ranks with peak <= cap}
//
// for each BLOCKED ending-height FBlock.  v0.26 already supplies the first
// compact-rank permutation.  Store only a LOW-all compact ordinal -> physical
// storage-rank permutation here.  A tiny host cache supplies per-height active
// counts and a per-group constant task prefix.
__device__ __constant__ const std::uint32_t* D_MS_LOW_ORBIT_COMPACT_LOW_RANK;
__device__ __constant__ Code D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[HIGH_LUT_K + 3];
__device__ __constant__ std::uint32_t
    D_MS_LOW_ORBIT_COMPACT_JOB_LOW_COUNT[HIGH_LUT_K + 2];

struct MaskShardLowOrbitRowDepthCompactCache {
    static constexpr int H = HIGH_LUT_K;
    static constexpr int L = LOW_LUT_K;
    static constexpr int S = FactorTablesHost::STRIDE;
    static constexpr int FULL_CAP = TARGET_W / 2;
    static constexpr int CAP_STRIDE = FULL_CAP + 1;
    static constexpr std::uint32_t HNM = 1u << H;

    std::vector<std::uint32_t> low_rank;
    std::array<std::array<std::uint32_t, CAP_STRIDE>, H + 2> low_count{};
    std::vector<std::uint16_t> high_count;
    std::array<std::uint32_t*, 8> d_low_rank{};
    std::array<bool, 8> installed{};
    bool built = false;

    static std::size_t high_count_index(
        std::uint32_t mask, int h, int cap
    ) {
        return (std::size_t(mask) * (H + 2) + std::size_t(h)) * CAP_STRIDE
             + std::size_t(cap);
    }

    void build() {
        if (built) return;
        auto& peaks = maskshard_lowclosure_rowdepth_cache();
        peaks.build();

        low_rank.resize(G_FACTOR.low_all_codes.size());
        std::vector<std::uint32_t> order;
        for (int h = 0; h <= H + 1; ++h) {
            const std::uint32_t a = G_FACTOR.low_all_off[h];
            const std::uint32_t z = G_FACTOR.low_all_off[h + 1];
            const std::uint32_t n = z - a;
            order.resize(n);
            for (std::uint32_t r = 0; r < n; ++r) order[r] = r;
            std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x,
                                                             std::uint32_t y) {
                const std::uint8_t px = peaks.low_all_peak[a + x];
                const std::uint8_t py = peaks.low_all_peak[a + y];
                return px != py ? px < py : x < y;
            });
            for (std::uint32_t q = 0; q < n; ++q)
                low_rank[a + q] = order[q];
            for (int cap = 0; cap <= FULL_CAP; ++cap) {
                std::uint32_t count = 0;
                while (count < n
                       && int(peaks.low_all_peak[a + low_rank[a + count]]) <= cap)
                    ++count;
                low_count[size_t(h)][size_t(cap)] = count;
            }
            if (low_count[size_t(h)][FULL_CAP] != n) {
                std::cerr << "LOW orbit compact full-cap LOW mismatch h=" << h
                          << " got=" << low_count[size_t(h)][FULL_CAP]
                          << " expected=" << n << '\n';
                std::exit(320);
            }
        }

        high_count.assign(
            std::size_t(HNM) * (H + 2) * CAP_STRIDE, std::uint16_t(0));
        for (std::uint32_t mask = 0; mask < HNM; ++mask) {
            for (int h = 0; h <= H + 1; ++h) {
                const std::size_t ix = std::size_t(mask) * S + h;
                const std::uint32_t a = G_FACTOR.high_mask_off[ix];
                const std::uint32_t n = factor_count(
                    G_FACTOR.high_mask_off, mask, h);
                if (n > 0xffffu) {
                    std::cerr << "LOW orbit compact HIGH rank count overflow mask="
                              << mask << " h=" << h << " n=" << n << '\n';
                    std::exit(321);
                }
                for (int cap = 0; cap <= FULL_CAP; ++cap) {
                    std::uint32_t count = 0;
                    for (std::uint32_t r = 0; r < n; ++r) {
                        const std::uint32_t code = G_FACTOR.high_mask_codes[a + r];
                        if (int(MaskShardRowDepthExactCache::peak_code(code, H, 1))
                            <= cap)
                            ++count;
                    }
                    high_count[high_count_index(mask, h, cap)] =
                        std::uint16_t(count);
                }
                if (high_count[high_count_index(mask, h, FULL_CAP)] != n) {
                    std::cerr << "LOW orbit compact full-cap HIGH mismatch mask="
                              << mask << " h=" << h << '\n';
                    std::exit(322);
                }
            }
        }

        built = true;
        std::cerr << "LOW orbit row-depth compact metadata low_rank_entries="
                  << low_rank.size() << " gpu_mib="
                  << double(low_rank.size() * sizeof(std::uint32_t))
                       / double(1ULL << 20)
                  << " host_high_count_mib="
                  << double(high_count.size() * sizeof(std::uint16_t))
                       / double(1ULL << 20) << '\n';
    }

    Code make_job_plan(
        std::uint32_t mask, int cap,
        std::array<Code, H + 3>& prefix,
        std::array<std::uint32_t, H + 2>& job_low_count
    ) {
        build();
        cap = std::max(1, std::min(cap, FULL_CAP));
        prefix.fill(0);
        job_low_count.fill(0);
        for (int h = 0; h <= H + 1; ++h) {
            const std::uint32_t hc = high_count[
                high_count_index(mask, h, cap)];
            const std::uint32_t lc = low_count[size_t(h)][size_t(cap)];
            job_low_count[size_t(h)] = lc;
            prefix[size_t(h + 1)] = prefix[size_t(h)] + Code(hc) * Code(lc);
        }
        return prefix[size_t(H + 2)];
    }

    void install_current_device() {
        build();
        int dev = -1;
        ck(cudaGetDevice(&dev), "LOW orbit compact get device");
        if (dev < 0 || dev >= int(installed.size())) {
            std::cerr << "LOW orbit compact unsupported device " << dev << '\n';
            std::exit(323);
        }
        if (installed[dev]) return;
        if (!low_rank.empty()) {
            ck(cudaMalloc(&d_low_rank[dev],
                          low_rank.size() * sizeof(std::uint32_t)),
               "LOW orbit compact LOW rank alloc");
            ck(cudaMemcpy(d_low_rank[dev], low_rank.data(),
                          low_rank.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "LOW orbit compact LOW rank copy");
        }
        ck(cudaMemcpyToSymbol(D_MS_LOW_ORBIT_COMPACT_LOW_RANK,
                              &d_low_rank[dev], sizeof(d_low_rank[dev])),
           "LOW orbit compact LOW rank ptr");
        installed[dev] = true;
    }

    void release() {
        for (int dev = 0; dev < int(installed.size()); ++dev) {
            if (!installed[dev]) continue;
            cudaSetDevice(dev);
            if (d_low_rank[dev]) cudaFree(d_low_rank[dev]);
            d_low_rank[dev] = nullptr;
            installed[dev] = false;
        }
    }
};

static MaskShardLowOrbitRowDepthCompactCache&
maskshard_loworbit_rowdepth_compact_cache() {
    static MaskShardLowOrbitRowDepthCompactCache cache;
    return cache;
}

// Extend the v0.24 setup hook: LOW orbit compact ranks are independent of the
// per-mask LOW closure tables and can be uploaded before setup_s is finalized.
#ifdef report_high_mask_shard_layout
#undef report_high_mask_shard_layout
#endif
static void maskshard_report_high_mask_shard_layout_loworbit_compact(
    const MaskShardLayout& s
) {
    maskshard_report_high_mask_shard_layout_lowclosure_rowdepth(s);
    auto& cache = maskshard_loworbit_rowdepth_compact_cache();
    cache.build();
    for (int d = 0; d < s.ngpu; ++d) {
        ck(cudaSetDevice(d), "LOW orbit compact setup device");
        cache.install_current_device();
    }
}
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_loworbit_compact

static Code maskshard_configure_loworbit_rowdepth_compact_group(
    std::uint32_t mask, int cap
) {
    std::array<Code, HIGH_LUT_K + 3> prefix{};
    std::array<std::uint32_t, HIGH_LUT_K + 2> low_count{};
    const Code total = maskshard_loworbit_rowdepth_compact_cache().make_job_plan(
        mask, cap, prefix, low_count);
    ck(cudaMemcpyToSymbol(D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX,
                          prefix.data(), sizeof(prefix)),
       "LOW orbit compact task prefix");
    ck(cudaMemcpyToSymbol(D_MS_LOW_ORBIT_COMPACT_JOB_LOW_COUNT,
                          low_count.data(), sizeof(low_count)),
       "LOW orbit compact job LOW counts");
    return total;
}

static void maskshard_release_loworbit_rowdepth_compact() {
    maskshard_loworbit_rowdepth_compact_cache().release();
}

__global__ void maskshard_main_block_loworbit_rowdepth_compact_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int FULL_CAP = TARGET_W / 2;
    const int nb = D_F_BLOCK_NBLOCKS;
    if (nb <= 0) return;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;
    const Code total = saturated
        ? D_F_BLOCK_BLOCKS[nb - 1].end
        : D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[nb];
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    Code task = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;

    for (; task < total; task += step) {
        int dbid = 0;
        FBlock dx{};
        std::uint32_t dhr = 0, dlr = 0;
        Code di = 0;

        if (saturated) {
            di = task;
            dbid = f_find_block(di);
            dx = D_F_BLOCK_BLOCKS[dbid];
            maskshard_split_rank(di, dx, dhr, dlr);
        } else {
            int lo = 0, hi = nb + 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[mid] <= task) lo = mid + 1;
                else hi = mid;
            }
            dbid = lo - 1;
            dx = D_F_BLOCK_BLOCKS[dbid];
            const Code local = task - D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[dbid];
            const std::uint32_t lc =
                D_MS_LOW_ORBIT_COMPACT_JOB_LOW_COUNT[dbid];
            if (!lc) continue;
            const std::uint32_t hq = std::uint32_t(local / Code(lc));
            const std::uint32_t lq = std::uint32_t(local - Code(hq) * Code(lc));
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(D_F_MASK) * S + dx.he];
            dhr = std::uint32_t(D_MS_LOW_CLOSURE_HIGH_COMPACT_RANK[ha + hq]);
            const std::uint32_t la = D_F_LOW_ALL_OFF[dx.he];
            dlr = D_MS_LOW_ORBIT_COMPACT_LOW_RANK[la + lq];
            di = dx.off + Code(dhr) * dx.stride + dlr;
        }

        const std::size_t bdi = std::size_t(pi) * D_LOWDESC_BLOCK_TOTAL
                              + D_LOWDESC_BLOCK_BASE[dbid] + dlr;
        const std::uint32_t bdesc = D_LOWDESC_BLOCK[bdi];
        if (lowdesc_kind(bdesc) != LOWDESC_MAIN) continue;
        const std::uint32_t sbid = lowdesc_block(bdesc);
        const std::uint32_t slr = lowdesc_lr(bdesc);
        const FBlock sx = D_F_MAIN_BLOCKS[sbid];
        const Code i = sx.off + Code(dhr) * sx.stride + slr;

        const std::size_t sdi = std::size_t(pi) * D_LOWDESC_MAIN_TOTAL
                              + D_LOWDESC_MAIN_BASE[sbid] + slr;
        const std::uint32_t aux = D_MS_LOW_ORBIT_AUX[bdi];
        const std::uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) continue;

        const Count c = mainv[i];
        const Count d = blockv[di];
        if (ak == MS_ORBIT_AUX_NN || p == 1) {
            const std::uint32_t desc = D_LOWDESC_MAIN[sdi];
            if (lowdesc_kind(desc) != LOWDESC_MAIN) continue;
            const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(dhr) * y.stride + lowdesc_lr(desc);
            if (ak == MS_ORBIT_AUX_NN) {
                mainv[j] = maskshard_add_mod_plain(mainv[j], c);
                mainv[i] = maskshard_add_mod_plain(c, d);
                blockv[di] = 0;
            } else {
                const Count cc = mainv[j];
                mainv[i] = maskshard_add_mod_plain(
                    maskshard_add_mod_plain(c, cc), d);
                mainv[j] = maskshard_add_mod_plain(c, cc);
                blockv[di] = 0;
            }
        } else {
            const FBlock y = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(dhr) * y.stride
                         + maskshard_orbit_aux_rank(aux);
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(
                maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
    (void)n;
}

#ifdef maskshard_main_block_loworbit_kernel
#undef maskshard_main_block_loworbit_kernel
#endif
#define maskshard_main_block_loworbit_kernel \
        maskshard_main_block_loworbit_rowdepth_compact_kernel
