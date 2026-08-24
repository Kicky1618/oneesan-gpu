#pragma once

#ifndef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT
#error "maskshard_highclosure_rowdepth_compact.cuh requires MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH
#error "exact HIGH closure task compaction layers on v0.20 row-depth semantics"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_ROWPACK
#error "exact HIGH closure task compaction requires row packing"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_TASK_LAUNCH
#error "exact HIGH closure task compaction requires v0.21 task-sized host launch"
#endif
#ifndef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#error "exact HIGH closure task compaction reuses v0.19 LOW compact ranks"
#endif
static_assert(LOW_LUT_K == HIGH_LUT_K + 1,
              "v0.22 LOW active-count reuse currently assumes LOW=HIGH+1");

// v0.22: compact only the structurally reachable HIGH closure source tasks.
// The existing v0.19 LOW compact-rank permutation gives, for each fixed LOW
// occupancy mask and starting height, a prefix ordered by exact peak.  Add one
// peak-ordered copy of the selected HIGH closure rows and a tiny cumulative
// active-row count table.  Then each source FBlock is exactly
//
//   active selected HIGH rows x active LOW ranks.
//
// The packing policy itself is unchanged from v0.11: whether a block is packed
// is decided from its dense x.stride and the fixed threshold, not from the
// row-dependent active LOW count.  At full depth every state is active, so the
// kernel falls back to the original closure-row order and physical LOW ranks to
// preserve v0.21 locality.
__device__ __constant__ std::uint32_t* D_MS_HC_RD_COMPACT_ROWS;
__device__ __constant__ std::uint32_t* D_MS_HC_RD_ACTIVE_COUNT;

struct MaskShardHighClosureRowDepthCompactCache {
    static constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    static constexpr int CAP_STRIDE = FULL_CAP + 1;
    static constexpr int BLOCK_STRIDE = 65;

    std::vector<std::uint32_t> rows;
    std::vector<std::uint32_t> active_count;
    std::array<std::uint32_t*, 8> d_rows{};
    std::array<std::uint32_t*, 8> d_active_count{};
    std::array<bool, 8> installed{};
    bool built = false;

    static std::size_t count_index(int pi, int bid, int cap) {
        return (std::size_t(pi) * BLOCK_STRIDE + std::size_t(bid)) * CAP_STRIDE
             + std::size_t(cap);
    }

    void build(const HighDescHost& high_desc) {
        if (built) return;
        MaskShardRowDepthExactCache& exact = maskshard_row_depth_exact_cache();
        exact.build();

        const auto blocks = make_factor_main_blocks(true, 0u);
        const int nb = int(blocks.size());
        if (nb <= 0 || nb >= BLOCK_STRIDE) {
            std::cerr << "HIGH closure row-depth compact invalid MAIN block count "
                      << nb << '\n';
            std::exit(280);
        }
        if (high_desc.closure_block_off.size()
            != std::size_t(HIGH_LUT_K) * BLOCK_STRIDE) {
            std::cerr << "HIGH closure row-depth compact block-offset size mismatch\n";
            std::exit(281);
        }

        rows.resize(high_desc.closure_rows.size());
        active_count.assign(
            std::size_t(HIGH_LUT_K) * BLOCK_STRIDE * CAP_STRIDE, 0u);
        std::vector<std::uint32_t> order;

        for (int pi = 0; pi < HIGH_LUT_K; ++pi) {
            for (int bid = 0; bid < nb; ++bid) {
                const std::uint32_t a = high_desc.closure_block_off[
                    std::size_t(pi) * BLOCK_STRIDE + std::size_t(bid)];
                const std::uint32_t z = high_desc.closure_block_off[
                    std::size_t(pi) * BLOCK_STRIDE + std::size_t(bid) + 1];
                if (z < a || z > high_desc.closure_rows.size()) {
                    std::cerr << "HIGH closure row-depth compact invalid row range pi="
                              << pi << " bid=" << bid << '\n';
                    std::exit(282);
                }
                const std::uint32_t n = z - a;
                order.resize(n);
                for (std::uint32_t q = 0; q < n; ++q) order[q] = q;
                const int he = int(blocks[std::size_t(bid)].he);
                const std::uint32_t hi_base = G_FACTOR.high_all_off[he];
                std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x,
                                                                 std::uint32_t y) {
                    const std::uint32_t rx = highdesc_host_rank(
                        high_desc.closure_rows[a + x]);
                    const std::uint32_t ry = highdesc_host_rank(
                        high_desc.closure_rows[a + y]);
                    const std::uint8_t px = exact.high_peak[hi_base + rx];
                    const std::uint8_t py = exact.high_peak[hi_base + ry];
                    return px != py ? px < py : rx < ry;
                });
                for (std::uint32_t q = 0; q < n; ++q) {
                    const std::uint32_t src = high_desc.closure_rows[a + order[q]];
                    if (highdesc_host_block(src) != std::uint32_t(bid)) {
                        std::cerr << "HIGH closure row-depth compact source block mismatch\n";
                        std::exit(283);
                    }
                    rows[a + q] = src;
                }
                for (int cap = 0; cap <= FULL_CAP; ++cap) {
                    std::uint32_t count = 0;
                    while (count < n) {
                        const std::uint32_t hr = highdesc_host_rank(rows[a + count]);
                        if (int(exact.high_peak[hi_base + hr]) > cap) break;
                        ++count;
                    }
                    active_count[count_index(pi, bid, cap)] = count;
                }
                if (active_count[count_index(pi, bid, FULL_CAP)] != n) {
                    std::cerr << "HIGH closure row-depth compact full-cap mismatch pi="
                              << pi << " bid=" << bid << '\n';
                    std::exit(284);
                }
            }
        }

        built = true;
        const std::size_t gpu_bytes = rows.size() * sizeof(std::uint32_t)
                                    + active_count.size() * sizeof(std::uint32_t);
        std::cerr << "HIGH closure row-depth compact metadata rows=" << rows.size()
                  << " count_entries=" << active_count.size()
                  << " gpu_mib=" << double(gpu_bytes) / double(1ULL << 20)
                  << '\n';
    }

    void install_current_device(const HighDescHost& high_desc) {
        build(high_desc);
        int dev = -1;
        ck(cudaGetDevice(&dev), "HIGH closure row-depth compact get device");
        if (dev < 0 || dev >= int(installed.size())) {
            std::cerr << "HIGH closure row-depth compact unsupported device " << dev << '\n';
            std::exit(285);
        }
        if (installed[dev]) return;

        if (!rows.empty()) {
            ck(cudaMalloc(&d_rows[dev], rows.size() * sizeof(std::uint32_t)),
               "HIGH closure row-depth compact rows alloc");
            ck(cudaMemcpy(d_rows[dev], rows.data(), rows.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "HIGH closure row-depth compact rows copy");
        }
        if (!active_count.empty()) {
            ck(cudaMalloc(&d_active_count[dev],
                          active_count.size() * sizeof(std::uint32_t)),
               "HIGH closure row-depth compact count alloc");
            ck(cudaMemcpy(d_active_count[dev], active_count.data(),
                          active_count.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "HIGH closure row-depth compact count copy");
        }
        ck(cudaMemcpyToSymbol(D_MS_HC_RD_COMPACT_ROWS,
                              &d_rows[dev], sizeof(d_rows[dev])),
           "HIGH closure row-depth compact rows ptr");
        ck(cudaMemcpyToSymbol(D_MS_HC_RD_ACTIVE_COUNT,
                              &d_active_count[dev], sizeof(d_active_count[dev])),
           "HIGH closure row-depth compact count ptr");
        installed[dev] = true;
    }

    void release() {
        for (int dev = 0; dev < int(installed.size()); ++dev) {
            if (!installed[dev]) continue;
            cudaSetDevice(dev);
            if (d_rows[dev]) cudaFree(d_rows[dev]);
            if (d_active_count[dev]) cudaFree(d_active_count[dev]);
            d_rows[dev] = nullptr;
            d_active_count[dev] = nullptr;
            installed[dev] = false;
        }
    }
};

static MaskShardHighClosureRowDepthCompactCache&
maskshard_highclosure_rowdepth_compact_cache() {
    static MaskShardHighClosureRowDepthCompactCache cache;
    return cache;
}

static void maskshard_prepare_highclosure_rowdepth_compact(
    const HighDescHost& high_desc, int ngpu
) {
    auto& cache = maskshard_highclosure_rowdepth_compact_cache();
    cache.build(high_desc);
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "HIGH closure row-depth compact setup device");
        cache.install_current_device(high_desc);
    }
}

static void maskshard_release_highclosure_rowdepth_compact() {
    maskshard_highclosure_rowdepth_compact_cache().release();
}

__device__ __forceinline__ void maskshard_highclosure_compact_apply(
    Count* mainv,
    Count* blockv,
    const FBlock& x,
    std::uint32_t hr,
    std::uint32_t desc,
    std::uint32_t lr
) {
    constexpr int S = MAXW + 2;
    constexpr std::uint32_t LR_MASK = (1u << LOW_LUT_K) - 1u;
    const Code i = x.off + Code(hr) * x.stride + lr;
    const Count c = mainv[i];
    if (!c) return;
    const std::uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
        const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
        atomic_add_mod(blockv + j, c);
    } else if (kind == HIGHDESC_CROSS) {
        const std::uint32_t la = D_F_LOW_MASK_OFF[
            std::size_t(D_F_MASK) * S + x.hs];
        const std::uint32_t lc = D_F_LOW_MASK_CODES[la + lr];
        const std::uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
        if (lc2 == 0xffffffffu) return;
        const std::uint32_t lp = D_F_LOW_PACKED_RANK[lc2];
        const std::uint32_t lr2 = lp & LR_MASK;
        const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
        const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr2;
        atomic_add_mod(blockv + j, c);
    }
}

__global__ void maskshard_main_highdesc_closure_rowdepth_compact_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr int CAP_STRIDE = FULL_CAP + 1;
    constexpr int BLOCK_STRIDE = 65;
    __shared__ Code prefix[65];
    const std::uint32_t pi = std::uint32_t((TARGET_W - 1) - p);
    const int nb = D_F_MAIN_NBLOCKS;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;

    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int b = 0; b < nb; ++b) {
            const FBlock x = D_F_MAIN_BLOCKS[b];
            const std::uint32_t a = D_HIGHDESC_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * BLOCK_STRIDE + b];
            const std::uint32_t z = D_HIGHDESC_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * BLOCK_STRIDE + b + 1];
            const Code rows = saturated
                ? Code(z - a)
                : Code(D_MS_HC_RD_ACTIVE_COUNT[
                    (std::size_t(pi) * BLOCK_STRIDE + b) * CAP_STRIDE + cap]);
            const std::uint32_t lc = saturated
                ? x.stride
                : (x.stride ? std::uint32_t(
                    D_MS_ROW_DEPTH_COMPACT_JOB_LOW_COUNT[x.hs]) : 0u);
            Code tasks = 0;
            if (rows && lc) {
                tasks = maskshard_highclosure_pack_block(x)
                    ? (rows * Code(lc) + 31) >> 5
                    : rows;
            }
            prefix[b + 1] = prefix[b] + tasks;
        }
    }
    __syncthreads();

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int(blockDim.x >> 5);
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

        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const std::uint32_t a = D_HIGHDESC_CLOSURE_BLOCK_OFF[
            std::size_t(pi) * BLOCK_STRIDE + bid];
        const std::uint32_t z = D_HIGHDESC_CLOSURE_BLOCK_OFF[
            std::size_t(pi) * BLOCK_STRIDE + bid + 1];
        const std::uint32_t rows = saturated
            ? z - a
            : D_MS_HC_RD_ACTIVE_COUNT[
                (std::size_t(pi) * BLOCK_STRIDE + bid) * CAP_STRIDE + cap];
        const std::uint32_t lc = saturated
            ? x.stride
            : std::uint32_t(D_MS_ROW_DEPTH_COMPACT_JOB_LOW_COUNT[x.hs]);
        const bool pack = maskshard_highclosure_pack_block(x);
        const std::uint32_t lo_base = D_F_LOW_MASK_OFF[
            std::size_t(D_F_MASK) * S + x.hs];

        if (!pack) {
            const std::uint32_t row_local = std::uint32_t(local);
            if (row_local >= rows) continue;
            std::uint32_t source = 0;
            if (lane == 0) {
                source = saturated
                    ? D_HIGHDESC_CLOSURE_ROWS[a + row_local]
                    : D_MS_HC_RD_COMPACT_ROWS[a + row_local];
            }
            source = __shfl_sync(active, source, 0);
            const std::uint32_t hr = highdesc_rank(source);
            std::uint32_t desc = 0;
            if (lane == 0) {
                desc = D_HIGHDESC_MAIN[
                    std::size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                    + D_HIGHDESC_MAIN_BASE[bid] + hr];
            }
            desc = __shfl_sync(active, desc, 0);
            for (std::uint32_t q = std::uint32_t(lane); q < lc; q += 32u) {
                const std::uint32_t lr = saturated
                    ? q
                    : std::uint32_t(D_MS_ROW_DEPTH_LOW_COMPACT_RANK[lo_base + q]);
                maskshard_highclosure_compact_apply(
                    mainv, blockv, x, hr, desc, lr);
            }
            continue;
        }

        const Code items = Code(rows) * Code(lc);
        const Code item = (local << 5) + Code(lane);
        const bool valid = item < items;
        const unsigned valid_mask = __ballot_sync(active, valid);
        if (!valid) continue;
        const std::uint32_t row_local = std::uint32_t(item / Code(lc));
        const std::uint32_t q = std::uint32_t(item - Code(row_local) * Code(lc));
        const std::uint32_t lr = saturated
            ? q
            : std::uint32_t(D_MS_ROW_DEPTH_LOW_COMPACT_RANK[lo_base + q]);
        const unsigned row_mask = __match_any_sync(valid_mask, row_local);
        const int leader = __ffs(int(row_mask)) - 1;

        std::uint32_t source = 0;
        if (lane == leader) {
            source = saturated
                ? D_HIGHDESC_CLOSURE_ROWS[a + row_local]
                : D_MS_HC_RD_COMPACT_ROWS[a + row_local];
        }
        source = __shfl_sync(row_mask, source, leader);
        const std::uint32_t hr = highdesc_rank(source);
        std::uint32_t desc = 0;
        if (lane == leader) {
            desc = D_HIGHDESC_MAIN[
                std::size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                + D_HIGHDESC_MAIN_BASE[bid] + hr];
        }
        desc = __shfl_sync(row_mask, desc, leader);
        maskshard_highclosure_compact_apply(mainv, blockv, x, hr, desc, lr);
    }
    (void)n;
}

#ifdef maskshard_main_highdesc_closure_inplace_kernel
#undef maskshard_main_highdesc_closure_inplace_kernel
#endif
#define maskshard_main_highdesc_closure_inplace_kernel \
        maskshard_main_highdesc_closure_rowdepth_compact_inplace_kernel
