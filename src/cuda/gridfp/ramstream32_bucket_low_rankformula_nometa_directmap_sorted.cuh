#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_directmap.cuh"

#if !P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#error "sorted directgather requires DIRECTMAP"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER
#error "sorted directgather requires DIRECTGATHER"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER_SORTED
#error "sorted directgather header requires SORTED=1"
#endif

__device__ __forceinline__ uint16_t p10dc_directgather_sorted_get(const uint4& d, uint32_t i) {
    const uint32_t w = i < 2u ? d.x : i < 4u ? d.y : i < 6u ? d.z : d.w;
    return uint16_t((w >> (16u * (i & 1u))) & 0xffffu);
}

__global__ void p10dc_directgather_sort_descriptors_kernel(uint4* table, size_t count) {
    for (size_t ix = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         ix < count; ix += size_t(gridDim.x) * blockDim.x) {
        const uint4 d = table[ix];
        const uint32_t n = d.w >> 16;
        if (n < 2u) continue;
        uint16_t r[7] = {
            p10dc_directgather_sorted_get(d, 0),
            p10dc_directgather_sorted_get(d, 1),
            p10dc_directgather_sorted_get(d, 2),
            p10dc_directgather_sorted_get(d, 3),
            p10dc_directgather_sorted_get(d, 4),
            p10dc_directgather_sorted_get(d, 5),
            p10dc_directgather_sorted_get(d, 6)};
#pragma unroll
        for (uint32_t i = 1; i < 7u; ++i) {
            if (i >= n) break;
            const uint16_t x = r[i];
            uint32_t j = i;
            while (j && x < r[j - 1u]) {
                r[j] = r[j - 1u];
                --j;
            }
            r[j] = x;
        }
        table[ix] = make_uint4(
            uint32_t(r[0]) | (uint32_t(r[1]) << 16),
            uint32_t(r[2]) | (uint32_t(r[3]) << 16),
            uint32_t(r[4]) | (uint32_t(r[5]) << 16),
            uint32_t(r[6]) | (n << 16));
    }
}

struct BucketFusedDirectHighRowsRankFormulaNometa4DirectMapSortedTables
    : BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables {
    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::bind_owner(
            fixed, buckets, slot);
        if (!low_rankformula_directgather4_count) return;
        const uint32_t threads = 256;
        const uint32_t blocks = uint32_t(std::min<size_t>(
            4096u, (low_rankformula_directgather4_count + threads - 1u) / threads));
        p10dc_directgather_sort_descriptors_kernel<<<blocks, threads>>>(
            low_rankformula_directgather4, low_rankformula_directgather4_count);
        ck(cudaGetLastError(), "p10dc directgather sorted launch");
        ck(cudaDeviceSynchronize(), "p10dc directgather sorted sync");
        std::cerr << "p10dc_low_rankformula_directgather_sorted fixed_owner=" << fixed
                  << " descriptors=" << low_rankformula_directgather4_count
                  << " descriptor_order_unchanged=1"
                  << " source_order=ascending"
                  << " semantic_operation=commutative_sum\n";
    }
};
