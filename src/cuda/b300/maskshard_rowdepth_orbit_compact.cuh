#pragma once

#ifndef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#error "maskshard_rowdepth_orbit_compact.cuh requires MASKSHARD_ROW_DEPTH_ORBIT_COMPACT"
#endif
#ifndef MASKSHARD_ROW_DEPTH_ORBIT
#error "compact row-depth orbit layers on v0.17 semantics"
#endif
#ifndef MASKSHARD_ROW_DEPTH_EXACT_IO
#error "compact row-depth orbit reuses v0.15 exact peak metadata"
#endif
#ifndef MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH
#error "compact row-depth orbit requires BLOCKED-domain launch geometry"
#endif
static_assert(HIGH_LUT_K <= LOW_LUT_K,
              "compact BLOCKED task metadata assumes HIGH<=LOW");

// v0.19: exact active BLOCKED tasks are a Cartesian product inside each
// boundary-height FBlock:
//
//   {HIGH ranks with peak <= cap} x {LOW mask-ranks with peak <= cap}.
//
// Build one peak-sorted rank permutation per factor group during setup. The
// per-cap cumulative counts stay host-only and generate a tiny per-job constant
// plan (task prefix + LOW active count, ~150 B at W=28). Thus persistent GPU
// metadata is only the two compact-rank permutations.
__device__ __constant__ std::uint16_t* D_MS_ROW_DEPTH_LOW_COMPACT_RANK;
__device__ __constant__ std::uint32_t* D_MS_ROW_DEPTH_HIGH_COMPACT_RANK;
__device__ __constant__ Code D_MS_ROW_DEPTH_COMPACT_TASK_PREFIX[HIGH_LUT_K + 3];
__device__ __constant__ std::uint16_t D_MS_ROW_DEPTH_COMPACT_JOB_LOW_COUNT[HIGH_LUT_K + 2];

struct MaskShardRowDepthOrbitCompactCache {
    static constexpr int FULL_CAP = TARGET_W / 2;
    static constexpr int CAP_STRIDE = FULL_CAP + 1;

    std::vector<std::uint16_t> low_rank;
    std::vector<std::uint32_t> high_rank;
    // Host-only cumulative active counts used to construct each job plan.
    std::vector<std::uint16_t> low_count;
    std::vector<std::uint32_t> high_count;

    std::array<std::uint16_t*, 8> d_low_rank{};
    std::array<std::uint32_t*, 8> d_high_rank{};
    std::array<bool, 8> installed{};
    bool built = false;

    static std::size_t low_count_index(std::uint32_t mask, int h, int cap) {
        constexpr int L = LOW_LUT_K;
        return (std::size_t(mask) * (L + 2) + std::size_t(h)) * CAP_STRIDE
             + std::size_t(cap);
    }
    static std::size_t high_count_index(int h, int cap) {
        return std::size_t(h) * CAP_STRIDE + std::size_t(cap);
    }

    void build() {
        if (built) return;
        constexpr int L = LOW_LUT_K;
        constexpr int H = HIGH_LUT_K;
        constexpr int S = FactorTablesHost::STRIDE;
        constexpr std::uint32_t NM = 1u << L;

        MaskShardRowDepthExactCache& exact = maskshard_row_depth_exact_cache();
        exact.build();

        low_rank.resize(G_FACTOR.low_mask_codes.size());
        high_rank.resize(G_FACTOR.high_all_codes.size());
        low_count.assign(std::size_t(NM) * (L + 2) * CAP_STRIDE, 0);
        high_count.assign(std::size_t(H + 2) * CAP_STRIDE, 0);

        std::vector<std::uint32_t> order;
        for (std::uint32_t mask = 0; mask < NM; ++mask) {
            for (int h = 0; h <= L + 1; ++h) {
                const std::size_t ix = std::size_t(mask) * S + h;
                const std::uint32_t a = G_FACTOR.low_mask_off[ix];
                const std::uint32_t n = factor_count(G_FACTOR.low_mask_off, mask, h);
                if (n > 0xffffu) {
                    std::cerr << "row-depth compact LOW rank exceeds uint16 mask="
                              << mask << " h=" << h << " n=" << n << '\n';
                    std::exit(270);
                }
                order.resize(n);
                for (std::uint32_t r = 0; r < n; ++r) order[r] = r;
                std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x,
                                                                 std::uint32_t y) {
                    const std::uint8_t px = exact.low_peak[a + x];
                    const std::uint8_t py = exact.low_peak[a + y];
                    return px != py ? px < py : x < y;
                });
                for (std::uint32_t q = 0; q < n; ++q)
                    low_rank[a + q] = std::uint16_t(order[q]);
                for (int cap = 0; cap <= FULL_CAP; ++cap) {
                    std::uint32_t count = 0;
                    while (count < n
                           && int(exact.low_peak[a + order[count]]) <= cap)
                        ++count;
                    low_count[low_count_index(mask, h, cap)] =
                        std::uint16_t(count);
                }
            }
        }

        for (int h = 0; h <= H + 1; ++h) {
            const std::uint32_t a = G_FACTOR.high_all_off[h];
            const std::uint32_t z = G_FACTOR.high_all_off[h + 1];
            const std::uint32_t n = z - a;
            order.resize(n);
            for (std::uint32_t r = 0; r < n; ++r) order[r] = r;
            std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x,
                                                             std::uint32_t y) {
                const std::uint8_t px = exact.high_peak[a + x];
                const std::uint8_t py = exact.high_peak[a + y];
                return px != py ? px < py : x < y;
            });
            for (std::uint32_t q = 0; q < n; ++q) high_rank[a + q] = order[q];
            for (int cap = 0; cap <= FULL_CAP; ++cap) {
                std::uint32_t count = 0;
                while (count < n
                       && int(exact.high_peak[a + order[count]]) <= cap)
                    ++count;
                high_count[high_count_index(h, cap)] = count;
            }
        }

        built = true;
        const std::size_t gpu_bytes = low_rank.size() * sizeof(std::uint16_t)
                                    + high_rank.size() * sizeof(std::uint32_t);
        const std::size_t host_count_bytes = low_count.size() * sizeof(std::uint16_t)
                                           + high_count.size() * sizeof(std::uint32_t);
        std::cerr << "row-depth compact orbit metadata low_rank=" << low_rank.size()
                  << " high_rank=" << high_rank.size()
                  << " gpu_mib=" << double(gpu_bytes) / double(1ULL << 20)
                  << " host_count_mib="
                  << double(host_count_bytes) / double(1ULL << 20) << '\n';
    }

    Code make_job_plan(
        std::uint32_t mask,
        int cap,
        std::array<Code, HIGH_LUT_K + 3>& prefix,
        std::array<std::uint16_t, HIGH_LUT_K + 2>& job_low_count
    ) {
        build();
        constexpr int H = HIGH_LUT_K;
        cap = std::max(0, std::min(cap, FULL_CAP));
        prefix.fill(0);
        job_low_count.fill(0);
        for (int h = 0; h <= H + 1; ++h) {
            const std::uint32_t hc = high_count[high_count_index(h, cap)];
            const std::uint16_t lc = low_count[low_count_index(mask, h, cap)];
            job_low_count[size_t(h)] = lc;
            prefix[size_t(h + 1)] = prefix[size_t(h)] + Code(hc) * Code(lc);
        }
        return prefix[size_t(H + 2)];
    }

    void install_current_device() {
        build();
        int dev = -1;
        ck(cudaGetDevice(&dev), "row-depth compact get device");
        if (dev < 0 || dev >= int(installed.size())) {
            std::cerr << "row-depth compact unsupported device id " << dev << '\n';
            std::exit(271);
        }
        if (installed[dev]) return;

        if (!low_rank.empty()) {
            ck(cudaMalloc(&d_low_rank[dev], low_rank.size() * sizeof(std::uint16_t)),
               "row-depth compact alloc LOW ranks");
            ck(cudaMemcpy(d_low_rank[dev], low_rank.data(),
                          low_rank.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "row-depth compact copy LOW ranks");
        }
        if (!high_rank.empty()) {
            ck(cudaMalloc(&d_high_rank[dev], high_rank.size() * sizeof(std::uint32_t)),
               "row-depth compact alloc HIGH ranks");
            ck(cudaMemcpy(d_high_rank[dev], high_rank.data(),
                          high_rank.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "row-depth compact copy HIGH ranks");
        }

        ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_LOW_COMPACT_RANK,
                              &d_low_rank[dev], sizeof(d_low_rank[dev])),
           "row-depth compact LOW rank ptr");
        ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_HIGH_COMPACT_RANK,
                              &d_high_rank[dev], sizeof(d_high_rank[dev])),
           "row-depth compact HIGH rank ptr");
        installed[dev] = true;
    }
};

static MaskShardRowDepthOrbitCompactCache& maskshard_row_depth_orbit_compact_cache() {
    static MaskShardRowDepthOrbitCompactCache cache;
    return cache;
}

static Code maskshard_configure_row_depth_compact_group(
    std::uint32_t mask, int cap
) {
    std::array<Code, HIGH_LUT_K + 3> prefix{};
    std::array<std::uint16_t, HIGH_LUT_K + 2> low_count{};
    const Code total = maskshard_row_depth_orbit_compact_cache().make_job_plan(
        mask, cap, prefix, low_count);
    ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_COMPACT_TASK_PREFIX,
                          prefix.data(), sizeof(prefix)),
       "row-depth compact task prefix");
    ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_COMPACT_JOB_LOW_COUNT,
                          low_count.data(), sizeof(low_count)),
       "row-depth compact job LOW counts");
    return total;
}

// The v0.15 header already redirects report_high_mask_shard_layout() to a setup
// hook that installs exact peak bytes. Replace that macro with one more setup
// layer so compact rank permutations are also built/uploaded before setup_s is
// finalized. Active-count tables remain host-only.
#ifdef report_high_mask_shard_layout
#undef report_high_mask_shard_layout
#endif
static void maskshard_report_high_mask_shard_layout_orbit_compact(
    const MaskShardLayout& s
) {
    maskshard_report_high_mask_shard_layout_exact(s);
    MaskShardRowDepthOrbitCompactCache& cache =
        maskshard_row_depth_orbit_compact_cache();
    cache.build();
    for (int d = 0; d < s.ngpu; ++d) {
        ck(cudaSetDevice(d), "row-depth compact setup device");
        cache.install_current_device();
    }
}
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_orbit_compact

__global__ void maskshard_main_block_highorbit_rowdepth_compact_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    const int nb = D_F_BLOCK_NBLOCKS;
    const Code total = D_MS_ROW_DEPTH_COMPACT_TASK_PREFIX[nb];
    Code task = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const std::uint32_t pi = std::uint32_t((TARGET_W - 1) - p);
    const bool first_high = p == TARGET_W - 1;

    for (; task < total; task += step) {
        int lo = 0, hi = nb + 1;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (D_MS_ROW_DEPTH_COMPACT_TASK_PREFIX[mid] <= task) lo = mid + 1;
            else hi = mid;
        }
        const int dbid = lo - 1;
        const FBlock dx = D_F_BLOCK_BLOCKS[dbid];
        const int h = int(dx.he);
        const Code local = task - D_MS_ROW_DEPTH_COMPACT_TASK_PREFIX[dbid];
        const std::uint16_t lc = D_MS_ROW_DEPTH_COMPACT_JOB_LOW_COUNT[dbid];
        if (!lc) continue;

        const std::uint32_t compact_hr = std::uint32_t(local / Code(lc));
        const std::uint32_t compact_lr =
            std::uint32_t(local - Code(compact_hr) * Code(lc));
        const std::uint32_t hi_base = D_F_HIGH_ALL_OFF[h];
        const std::uint32_t lo_base = D_F_LOW_MASK_OFF[
            std::size_t(D_F_MASK) * S + h];
        const std::uint32_t dhr = D_MS_ROW_DEPTH_HIGH_COMPACT_RANK[
            hi_base + compact_hr];
        const std::uint32_t dlr = std::uint32_t(
            D_MS_ROW_DEPTH_LOW_COMPACT_RANK[lo_base + compact_lr]);
        const Code di = dx.off + Code(dhr) * dx.stride + dlr;

        const std::size_t bdi = std::size_t(pi) * D_HIGHDESC_BLOCK_TOTAL
                              + D_HIGHDESC_BLOCK_BASE[dbid] + dhr;
        const std::uint32_t bdesc = D_HIGHDESC_BLOCK[bdi];
        if (highdesc_kind(bdesc) != HIGHDESC_MAIN) {
            if (first_high) blockv[di] = 0;
            continue;
        }
        const std::uint32_t sbid = highdesc_block(bdesc);
        const std::uint32_t shr = highdesc_rank(bdesc);
        const FBlock sx = D_F_MAIN_BLOCKS[sbid];
        const Code i = sx.off + Code(shr) * sx.stride + dlr;

        const std::size_t sdi = std::size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                              + D_HIGHDESC_MAIN_BASE[sbid] + shr;
#ifdef MASKSHARD_BLOCK_ORBIT_AUX
        const std::uint32_t aux = D_MS_HIGH_ORBIT_AUX[bdi];
#else
        const std::uint32_t aux = D_MS_HIGH_ORBIT_AUX[sdi];
#endif
        const std::uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) {
            if (first_high) blockv[di] = 0;
            continue;
        }

        const Count c = mainv[i];
        const Count d = first_high ? Count(0) : blockv[di];
        if (ak == MS_ORBIT_AUX_NN) {
            const std::uint32_t desc = D_HIGHDESC_MAIN[sdi];
            if (highdesc_kind(desc) != HIGHDESC_MAIN) {
                if (first_high) blockv[di] = 0;
                continue;
            }
            const FBlock y = D_F_MAIN_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + dlr;
            mainv[j] = maskshard_add_mod_plain(mainv[j], c);
            mainv[i] = maskshard_add_mod_plain(c, d);
            blockv[di] = 0;
        } else {
            const FBlock y = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(maskshard_orbit_aux_rank(aux)) * y.stride + dlr;
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
    (void)n;
}

#ifdef maskshard_main_block_highorbit_kernel
#undef maskshard_main_block_highorbit_kernel
#endif
#define maskshard_main_block_highorbit_kernel \
        maskshard_main_block_highorbit_rowdepth_compact_kernel
