#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_mlp.cuh"

#if !P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#error "directmap table header requires P10DC_RANKFORMULA_NOMETA_DIRECTMAP=1"
#endif
#if !P10DC_RANKFORMULA_NOMETA_GROUP61
#error "directmap table currently requires GROUP61"
#endif
#if P10DC_RANKFORMULA_DIRECTGATHER && !P10DC_RANKFORMULA_GATHER_MLP
#error "DIRECTGATHER requires GATHER_MLP=1"
#endif

// Compact direct descriptor layout:
//   start15 | source_base15 | lcount3 | abstract_off13 = 46 bits.
__device__ __host__ __forceinline__ uint64_t p10dc_rankformula_directmap_pack(
    uint32_t h, const P10DCRankFormulaNometa4Resolved& z
) {
    if (z.n < h || ((z.n - h) & 1u)) return ~uint64_t(0);
    const uint32_t lcount = (z.n - h) >> 1;
    if (z.start >= (1u << 15) || z.source_base >= (1u << 15) ||
        lcount >= (1u << 3) || z.abstract_off >= (1u << 13))
        return ~uint64_t(0);
    return uint64_t(z.start) |
           (uint64_t(z.source_base) << 15) |
           (uint64_t(lcount) << 30) |
           (uint64_t(z.abstract_off) << 33);
}

__global__ void p10dc_rankformula_directmap_fill_kernel(
    uint64_t* out, uint32_t out_base, uint32_t h, uint32_t count,
    uint32_t* error
) {
    for (uint32_t rank = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         rank < count; rank += uint32_t(gridDim.x) * blockDim.x) {
        const auto z = p10dc_low_rankformula_nometa4_resolve(h, rank);
        const uint64_t e = p10dc_rankformula_directmap_pack(h, z);
        if (e == ~uint64_t(0)) {
            atomicCAS(error, 0u, 1u);
        } else {
            out[out_base + rank] = e;
        }
    }
}

struct BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables
    : BucketFusedDirectHighRowsRankFormulaNometa4Tables {
    uint64_t* low_rankformula_direct64 = nullptr;
    size_t low_rankformula_direct64_count = 0;
    size_t low_rankformula_direct64_capacity = 0;
#if P10DC_RANKFORMULA_DIRECTGATHER
    uint4* low_rankformula_directgather4 = nullptr;
    size_t low_rankformula_directgather4_count = 0;
    size_t low_rankformula_directgather4_capacity = 0;
#endif

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4Tables::bind_owner(
            fixed, buckets, slot);
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc directmap invalid bind owner=" << fixed << '\n';
            std::exit(790);
        }

        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::array<uint32_t, MAXW + 2> off{};
        std::array<uint32_t, MAXW + 2> count{};
        std::array<uint32_t, MAXW + 2> begin{};
        size_t total = 0;
        uint32_t max_count = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u]
                : owner_end;
            if (a > b || b > owner_end) {
                std::cerr << "p10dc directmap height range invalid owner=" << fixed
                          << " h=" << h << " a=" << a << " b=" << b
                          << " owner_end=" << owner_end << '\n';
                std::exit(791);
            }
            begin[h] = a;
            off[h] = uint32_t(total);
            count[h] = b - a;
            total += count[h];
            max_count = std::max(max_count, count[h]);
        }
        if (total >= size_t(1u << 32)) std::exit(792);

        low_rankformula_direct64_count = total;
        if (total > low_rankformula_direct64_capacity) {
            if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
            low_rankformula_direct64 = nullptr;
            low_rankformula_direct64_capacity = total;
            if (total)
                ck(cudaMalloc(&low_rankformula_direct64, total * sizeof(uint64_t)),
                   "p10dc directmap alloc");
        }

        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA_DIRECT64,
                              &low_rankformula_direct64,
                              sizeof(low_rankformula_direct64)),
           "p10dc directmap ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA_DIRECT_OFF,
                              off.data(), off.size() * sizeof(uint32_t)),
           "p10dc directmap offsets");

        uint32_t* d_error = nullptr;
        ck(cudaMalloc(&d_error, sizeof(uint32_t)), "p10dc directmap error alloc");
        ck(cudaMemset(d_error, 0, sizeof(uint32_t)), "p10dc directmap error zero");
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            if (!count[h]) continue;
            const uint32_t threads = 256;
            const uint32_t blocks = std::min<uint32_t>(
                256u, (count[h] + threads - 1u) / threads);
            p10dc_rankformula_directmap_fill_kernel<<<blocks, threads>>>(
                low_rankformula_direct64, off[h], h, count[h], d_error);
            ck(cudaGetLastError(), "p10dc directmap fill launch");
        }
        ck(cudaDeviceSynchronize(), "p10dc directmap fill sync");
        uint32_t error = 0;
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
           "p10dc directmap error copy");
        cudaFree(d_error);
        if (error) {
            std::cerr << "p10dc directmap fill descriptor overflow owner="
                      << fixed << '\n';
            std::exit(793);
        }

#if P10DC_RANKFORMULA_DIRECTGATHER
        constexpr uint32_t DEPTHS = P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;
        constexpr uint32_t MASKS = P10DC_RANKFORMULA_NOMETA4_MASKS;
        std::vector<int32_t> first(
            size_t(P10DC_RANKFORMULA_NOMETA4_HEIGHTS) * MASKS, -1);
        auto first_ref = [&](uint32_t h, uint32_t mask) -> int32_t& {
            return first[size_t(h) * MASKS + mask];
        };
        for (uint32_t h = 0; h < P10DC_RANKFORMULA_NOMETA4_HEIGHTS; ++h) {
            for (uint32_t rank = 0; rank < count[h]; ++rank) {
                const uint32_t mask =
                    BucketFusedDirectHighRowsRankFormulaNometa4Tables::code_mask(
                        f.low_codes[begin[h] + rank]);
                int32_t& x = first_ref(h, mask);
                if (x < 0) x = int32_t(rank);
            }
        }

        const size_t gather_count = total * DEPTHS;
        std::vector<uint4> gather(gather_count, uint4{0u, 0u, 0u, 0u});
#if P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
        std::array<uint32_t,
            (MAXW + 2) * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS> depth_off{};
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            const uint64_t hbase = uint64_t(off[h]) * DEPTHS;
            for (uint32_t depth = 1; depth <= DEPTHS; ++depth) {
                const uint64_t z = hbase + uint64_t(depth - 1u) * count[h];
                if (z >= uint64_t(1u) << 32) std::exit(798);
                depth_off[h * DEPTHS + (depth - 1u)] = uint32_t(z);
            }
        }
#endif
        uint64_t selected_total = 0;
        uint32_t max_selected = 0;
        for (uint32_t h = 0; h < P10DC_RANKFORMULA_NOMETA4_HEIGHTS; ++h) {
            for (uint32_t rank = 0; rank < count[h]; ++rank) {
                const uint32_t mask =
                    BucketFusedDirectHighRowsRankFormulaNometa4Tables::code_mask(
                        f.low_codes[begin[h] + rank]);
                const uint32_t n = uint32_t(__builtin_popcount(mask));
                if (n <= h || h + 2u >= P10DC_RANKFORMULA_NOMETA4_HEIGHTS) continue;
                const int32_t start_i = first_ref(h, mask);
                const int32_t source_base_i = first_ref(h + 2u, mask);
                if (start_i < 0 || source_base_i < 0) continue;
                const uint32_t local = rank - uint32_t(start_i);
                const uint32_t lp = p10dc_rankformula_abstract_lpattern_host(
                    int(n), int(h), local);
                if (lp == 0xffffffffu) {
                    std::cerr << "p10dc directgather bad lpattern owner=" << fixed
                              << " h=" << h << " rank=" << rank << '\n';
                    std::exit(794);
                }
                for (uint32_t depth = 1; depth <= DEPTHS; ++depth) {
                    const uint32_t select = uint32_t(
                        p10dc_rankformula_abstract_select_host(
                            lp, int(n), depth));
                    uint16_t rr[8]{};
                    uint32_t li = 0, nr = 0;
                    for (uint32_t ord = 0; ord < n; ++ord) {
                        if (!((lp >> ord) & 1u)) continue;
                        if ((select >> li) & 1u) {
                            if (nr >= 7u) std::exit(795);
                            const uint32_t sr = p10dc_rankformula_abstract_rank_host(
                                int(n), int(h + 2u), lp & ~(1u << ord));
                            if (sr == 0xffffffffu) std::exit(796);
                            const uint32_t absolute = uint32_t(source_base_i) + sr;
                            if (absolute >= (1u << 16)) std::exit(797);
                            rr[nr++] = uint16_t(absolute);
                        }
                        ++li;
                    }
                    rr[7] = uint16_t(nr);
#if P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
                    const size_t gi = size_t(depth_off[h * DEPTHS + (depth - 1u)]) + rank;
#else
                    const size_t gi =
                        (size_t(off[h]) + rank) * DEPTHS + (depth - 1u);
#endif
                    gather[gi] = uint4{
                        uint32_t(rr[0]) | (uint32_t(rr[1]) << 16),
                        uint32_t(rr[2]) | (uint32_t(rr[3]) << 16),
                        uint32_t(rr[4]) | (uint32_t(rr[5]) << 16),
                        uint32_t(rr[6]) | (uint32_t(rr[7]) << 16)};
                    selected_total += nr;
                    max_selected = std::max(max_selected, nr);
                }
            }
        }

        low_rankformula_directgather4_count = gather_count;
        if (gather_count > low_rankformula_directgather4_capacity) {
            if (low_rankformula_directgather4) cudaFree(low_rankformula_directgather4);
            low_rankformula_directgather4 = nullptr;
            low_rankformula_directgather4_capacity = gather_count;
            if (gather_count)
                ck(cudaMalloc(&low_rankformula_directgather4,
                              gather_count * sizeof(uint4)),
                   "p10dc directgather alloc");
        }
        if (!gather.empty())
            ck(cudaMemcpy(low_rankformula_directgather4, gather.data(),
                          gather.size() * sizeof(uint4), cudaMemcpyHostToDevice),
               "p10dc directgather H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER4,
                              &low_rankformula_directgather4,
                              sizeof(low_rankformula_directgather4)),
           "p10dc directgather ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER_OFF,
                              off.data(), off.size() * sizeof(uint32_t)),
           "p10dc directgather offsets");
#if P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER_DEPTH_OFF,
                              depth_off.data(), depth_off.size() * sizeof(uint32_t)),
           "p10dc directgather depth offsets");
#endif
        std::cerr << "p10dc_low_rankformula_directgather fixed_owner=" << fixed
                  << " descriptors=" << gather_count
                  << " bytes=" << gather_count * sizeof(uint4)
                  << " mib=" << double(gather_count * sizeof(uint4)) / double(1 << 20)
                  << " avg_selected="
                  << (gather_count ? double(selected_total) / double(gather_count) : 0.0)
                  << " max_selected=" << max_selected
                  << " depth_major=" << P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
                  << " descriptor_lane_stride_bytes="
                  << (P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR ? sizeof(uint4) : DEPTHS * sizeof(uint4))
                  << " runtime_locator_loads=0"
                  << " runtime_depth_select_loads=0"
                  << " runtime_srcpack_loads=0"
                  << " runtime_gather_descriptor_loads=1\n";
#endif

        std::cerr << "p10dc_low_rankformula_directmap fixed_owner=" << fixed
                  << " entries=" << total
                  << " bytes=" << total * sizeof(uint64_t)
                  << " mib=" << double(total * sizeof(uint64_t)) / double(1 << 20)
                  << " max_height_ranks=" << max_count
                  << " runtime_group_loads=1"
                  << " runtime_block16_loads=0"
                  << " runtime_successor_loops=0"
                  << " runtime_ballots=0"
                  << " runtime_group_shuffles=0\n";
    }

    void release() {
#if P10DC_RANKFORMULA_DIRECTGATHER
        if (low_rankformula_directgather4) cudaFree(low_rankformula_directgather4);
        low_rankformula_directgather4 = nullptr;
        low_rankformula_directgather4_count = 0;
        low_rankformula_directgather4_capacity = 0;
#endif
        if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
        low_rankformula_direct64 = nullptr;
        low_rankformula_direct64_count = 0;
        low_rankformula_direct64_capacity = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4Tables::release();
    }
};
