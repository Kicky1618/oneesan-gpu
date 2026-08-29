#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_warpshare.cuh"

#if !P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#error "directmap table header requires P10DC_RANKFORMULA_NOMETA_DIRECTMAP=1"
#endif
#if !P10DC_RANKFORMULA_NOMETA_GROUP61
#error "directmap table currently requires GROUP61"
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
        if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
        low_rankformula_direct64 = nullptr;
        low_rankformula_direct64_count = 0;
        low_rankformula_direct64_capacity = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4Tables::release();
    }
};
