#pragma once

#include "ramstream32_bucket_low_rankformula_directgather64.cuh"
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#include "ramstream32_bucket_low_rankformula_directgather_sparse64.cuh"
#endif

#if !P10DC_RANKFORMULA_DIRECTGATHER64
#error "directgather64 sorted table requires DIRECTGATHER64=1"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER_SORTED
#error "directgather64 sorted table requires SORTED=1"
#endif

// Source summation is commutative. Sort the compressed source-rank list in
// place so a warp's ordinal-j loads have a better chance to hit nearby source
// addresses. The rare index remains attached to its primary descriptor; only
// the rank payload stored at that rare slot is rewritten.
__global__ void p10dc_directgather64_sort_kernel(
    P10DCDirectGather64Word* primary,
    P10DCDirectGather64Word* rare,
    uint32_t n
) {
    for (uint32_t i = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint32_t(gridDim.x) * blockDim.x) {
        P10DCDirectGather64Word p = primary[i];
        const uint32_t count = uint32_t((p >> 45) & 7u);
        if (count < 2u) continue;

        uint16_t r[7] = {
            uint16_t(p & 0x7fffu),
            uint16_t((p >> 15) & 0x7fffu),
            uint16_t((p >> 30) & 0x7fffu),
            0, 0, 0, 0};
        const uint32_t rare_ix = uint32_t(p >> 48);
        P10DCDirectGather64Word q = 0;
        if (count > 3u) {
            q = rare[rare_ix];
            r[3] = uint16_t(q & 0x7fffu);
            r[4] = uint16_t((q >> 15) & 0x7fffu);
            r[5] = uint16_t((q >> 30) & 0x7fffu);
            r[6] = uint16_t((q >> 45) & 0x7fffu);
        }

#pragma unroll
        for (uint32_t a = 1; a < 7u; ++a) {
            if (a >= count) break;
            const uint16_t x = r[a];
            uint32_t b = a;
            while (b && x < r[b - 1u]) {
                r[b] = r[b - 1u];
                --b;
            }
            r[b] = x;
        }

        primary[i] = P10DCDirectGather64Word(r[0]) |
            (P10DCDirectGather64Word(r[1]) << 15) |
            (P10DCDirectGather64Word(r[2]) << 30) |
            (P10DCDirectGather64Word(count) << 45) |
            (P10DCDirectGather64Word(rare_ix) << 48);
        if (count > 3u) {
            rare[rare_ix] = P10DCDirectGather64Word(r[3]) |
                (P10DCDirectGather64Word(r[4]) << 15) |
                (P10DCDirectGather64Word(r[5]) << 30) |
                (P10DCDirectGather64Word(r[6]) << 45);
        }
    }
}

static inline void p10dc_sort_directgather64_table(
    P10DCDirectGather64Word* primary,
    P10DCDirectGather64Word* rare,
    size_t n,
    const char* what
) {
    if (!n) return;
    const uint32_t threads = 256;
    const uint32_t blocks = std::min<uint32_t>(
        4096u, (uint32_t(n) + threads - 1u) / threads);
    p10dc_directgather64_sort_kernel<<<blocks, threads>>>(
        primary, rare, uint32_t(n));
    ck(cudaGetLastError(), what);
    ck(cudaDeviceSynchronize(), "p10dc directgather64 sorted sync");
}

struct BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64SortedTables
    : BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64Tables {
    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64Tables::bind_owner(
            fixed, buckets, slot);
        p10dc_sort_directgather64_table(
            low_rankformula_directgather64,
            low_rankformula_directgather64_rare,
            low_rankformula_directgather64_count,
            "p10dc directgather64 sorted launch");
        std::cerr << "p10dc_low_rankformula_directgather64_sorted fixed_owner=" << fixed
                  << " primary=" << low_rankformula_directgather64_count
                  << " source_order=ascending semantic_operation=commutative_sum\n";
    }
};

#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
struct BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64SortedTables
    : BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64Tables {
    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64Tables::bind_owner(
            fixed, buckets, slot);
        p10dc_sort_directgather64_table(
            low_rankformula_sparse64_primary,
            low_rankformula_sparse64_rare,
            low_rankformula_sparse64_primary_count,
            "p10dc sparse64 sorted launch");
        std::cerr << "p10dc_low_rankformula_sparse64_sorted fixed_owner=" << fixed
                  << " primary=" << low_rankformula_sparse64_primary_count
                  << " source_order=ascending semantic_operation=commutative_sum\n";
    }
};
#endif
