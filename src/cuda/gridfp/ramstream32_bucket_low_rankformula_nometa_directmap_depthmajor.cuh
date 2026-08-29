#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_directmap.cuh"

#if !P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#error "depth-major directgather requires DIRECTMAP"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER
#error "depth-major directgather requires DIRECTGATHER"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
#error "depth-major directgather header requires DEPTHMAJOR=1"
#endif

__global__ void p10dc_rankformula_directgather_depthmajor_transpose_kernel(
    uint4* dst, const uint4* src, uint32_t rank_base, uint32_t count
) {
    constexpr uint32_t DEPTHS = P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;
    const uint32_t total = count * DEPTHS;
    for (uint32_t q = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         q < total; q += uint32_t(gridDim.x) * blockDim.x) {
        const uint32_t depth = q / count;
        const uint32_t rank = q - depth * count;
        const size_t src_i =
            (size_t(rank_base) + size_t(rank)) * DEPTHS + size_t(depth);
        const size_t dst_i =
            size_t(rank_base) * DEPTHS + size_t(depth) * count + size_t(rank);
        dst[dst_i] = src[src_i];
    }
}

struct BucketFusedDirectHighRowsRankFormulaNometa4DirectMapDepthMajorTables
    : BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables {
    uint4* low_rankformula_directgather_depthmajor4 = nullptr;
    size_t low_rankformula_directgather_depthmajor4_capacity = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::bind_owner(
            fixed, buckets, slot);
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc directgather depthmajor invalid owner=" << fixed << '\n';
            std::exit(798);
        }

        constexpr uint32_t DEPTHS = P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;
        constexpr size_t P = size_t(MAXW + 2);
        const BucketFusedHost& f = *host_fused;
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::array<uint32_t, MAXW + 2> rank_off{};
        std::array<uint32_t, MAXW + 2> count{};
        std::array<uint32_t, (MAXW + 2) * DEPTHS> depth_off{};
        size_t total = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            if (a > b || b > owner_end) std::exit(799);
            rank_off[h] = uint32_t(total);
            count[h] = b - a;
            const uint32_t base = uint32_t(total * DEPTHS);
            for (uint32_t d = 0; d < DEPTHS; ++d)
                depth_off[size_t(h) * DEPTHS + d] = base + d * count[h];
            total += count[h];
        }

        const size_t entries = total * DEPTHS;
        if (entries != low_rankformula_directgather4_count) {
            std::cerr << "p10dc directgather depthmajor size mismatch owner=" << fixed
                      << " parent=" << low_rankformula_directgather4_count
                      << " depthmajor=" << entries << '\n';
            std::exit(800);
        }
        if (entries > low_rankformula_directgather_depthmajor4_capacity) {
            if (low_rankformula_directgather_depthmajor4)
                cudaFree(low_rankformula_directgather_depthmajor4);
            low_rankformula_directgather_depthmajor4 = nullptr;
            low_rankformula_directgather_depthmajor4_capacity = entries;
            if (entries)
                ck(cudaMalloc(&low_rankformula_directgather_depthmajor4,
                              entries * sizeof(uint4)),
                   "p10dc directgather depthmajor alloc");
        }

        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            if (!count[h]) continue;
            const uint32_t q = count[h] * DEPTHS;
            const uint32_t threads = 256;
            const uint32_t blocks = std::min<uint32_t>(1024u, (q + threads - 1u) / threads);
            p10dc_rankformula_directgather_depthmajor_transpose_kernel<<<blocks, threads>>>(
                low_rankformula_directgather_depthmajor4,
                low_rankformula_directgather4,
                rank_off[h], count[h]);
            ck(cudaGetLastError(), "p10dc directgather depthmajor transpose launch");
        }
        ck(cudaDeviceSynchronize(), "p10dc directgather depthmajor transpose sync");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER4,
                              &low_rankformula_directgather_depthmajor4,
                              sizeof(low_rankformula_directgather_depthmajor4)),
           "p10dc directgather depthmajor ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER_DEPTH_OFF,
                              depth_off.data(), depth_off.size() * sizeof(uint32_t)),
           "p10dc directgather depthmajor offsets");

        std::cerr << "p10dc_low_rankformula_directgather_depthmajor fixed_owner=" << fixed
                  << " entries=" << entries
                  << " bytes=" << entries * sizeof(uint4)
                  << " descriptor_bytes=16"
                  << " old_same_depth_lane_stride_bytes=" << (DEPTHS * sizeof(uint4))
                  << " new_same_depth_lane_stride_bytes=" << sizeof(uint4)
                  << " warp_descriptor_span_bytes=" << (32u * sizeof(uint4))
                  << " coalesced_depth_major=1\n";
    }

    void release() {
        if (low_rankformula_directgather_depthmajor4)
            cudaFree(low_rankformula_directgather_depthmajor4);
        low_rankformula_directgather_depthmajor4 = nullptr;
        low_rankformula_directgather_depthmajor4_capacity = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::release();
    }
};
