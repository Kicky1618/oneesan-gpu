#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_directmap.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64.cuh"

#if !P10DC_RANKFORMULA_DIRECTGATHER64
#error "directgather64 table requires P10DC_RANKFORMULA_DIRECTGATHER64=1"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER
#error "directgather64 table requires DIRECTGATHER=1"
#endif

static constexpr uint32_t P10DC_RANKFORMULA_DIRECTGATHER64_RARE_CAP = 65536u;

__global__ void p10dc_rankformula_directgather64_compress_kernel(
    const uint4* in, uint64_t* primary, uint64_t* rare,
    uint32_t n, uint32_t* rare_count, uint32_t* error
) {
    for (uint32_t i = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint32_t(gridDim.x) * blockDim.x) {
        const uint4 d = in[i];
        const uint32_t count = d.w >> 16;
        if (count > 7u) { atomicOr(error, 1u); continue; }
        const uint32_t r0 = d.x & 0xffffu, r1 = d.x >> 16;
        const uint32_t r2 = d.y & 0xffffu, r3 = d.y >> 16;
        const uint32_t r4 = d.z & 0xffffu, r5 = d.z >> 16;
        const uint32_t r6 = d.w & 0xffffu;
        if ((count > 0u && r0 >= 32768u) || (count > 1u && r1 >= 32768u) ||
            (count > 2u && r2 >= 32768u) || (count > 3u && r3 >= 32768u) ||
            (count > 4u && r4 >= 32768u) || (count > 5u && r5 >= 32768u) ||
            (count > 6u && r6 >= 32768u)) {
            atomicOr(error, 2u); continue;
        }
        uint32_t rare_ix = 0;
        if (count > 3u) {
            rare_ix = atomicAdd(rare_count, 1u);
            if (rare_ix >= P10DC_RANKFORMULA_DIRECTGATHER64_RARE_CAP) {
                atomicOr(error, 4u); continue;
            }
            rare[rare_ix] = uint64_t(r3) |
                (uint64_t(r4) << 15) |
                (uint64_t(r5) << 30) |
                (uint64_t(r6) << 45);
        }
        primary[i] = uint64_t(r0) |
            (uint64_t(r1) << 15) |
            (uint64_t(r2) << 30) |
            (uint64_t(count) << 45) |
            (uint64_t(rare_ix) << 48);
    }
}

struct BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64Tables
    : BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables {
    uint64_t* low_rankformula_directgather64 = nullptr;
    uint64_t* low_rankformula_directgather64_rare = nullptr;
    size_t low_rankformula_directgather64_count = 0;
    size_t low_rankformula_directgather64_capacity = 0;
    uint32_t low_rankformula_directgather64_rare_count = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::bind_owner(
            fixed, buckets, slot);
        const size_t n = low_rankformula_directgather4_count;
        if (n >= size_t(1u << 32)) {
            std::cerr << "p10dc directgather64 descriptor count overflow n=" << n << '\n';
            std::exit(810);
        }
        low_rankformula_directgather64_count = n;
        if (n > low_rankformula_directgather64_capacity) {
            if (low_rankformula_directgather64) cudaFree(low_rankformula_directgather64);
            low_rankformula_directgather64 = nullptr;
            low_rankformula_directgather64_capacity = n;
            if (n) ck(cudaMalloc(&low_rankformula_directgather64, n * sizeof(uint64_t)),
                      "p10dc directgather64 primary alloc");
        }
        if (!low_rankformula_directgather64_rare && n) {
            ck(cudaMalloc(&low_rankformula_directgather64_rare,
                          size_t(P10DC_RANKFORMULA_DIRECTGATHER64_RARE_CAP) * sizeof(uint64_t)),
               "p10dc directgather64 rare alloc");
        }

        uint32_t *d_rare_count = nullptr, *d_error = nullptr;
        ck(cudaMalloc(&d_rare_count, sizeof(uint32_t)), "p10dc directgather64 count alloc");
        ck(cudaMalloc(&d_error, sizeof(uint32_t)), "p10dc directgather64 error alloc");
        ck(cudaMemset(d_rare_count, 0, sizeof(uint32_t)), "p10dc directgather64 count zero");
        ck(cudaMemset(d_error, 0, sizeof(uint32_t)), "p10dc directgather64 error zero");
        if (n) {
            const uint32_t threads = 256;
            const uint32_t blocks = std::min<uint32_t>(4096u,
                (uint32_t(n) + threads - 1u) / threads);
            p10dc_rankformula_directgather64_compress_kernel<<<blocks, threads>>>(
                low_rankformula_directgather4,
                low_rankformula_directgather64,
                low_rankformula_directgather64_rare,
                uint32_t(n), d_rare_count, d_error);
            ck(cudaGetLastError(), "p10dc directgather64 compress launch");
            ck(cudaDeviceSynchronize(), "p10dc directgather64 compress sync");
        }
        uint32_t error = 0;
        ck(cudaMemcpy(&low_rankformula_directgather64_rare_count, d_rare_count,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc directgather64 count copy");
        ck(cudaMemcpy(&error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc directgather64 error copy");
        cudaFree(d_rare_count); cudaFree(d_error);
        if (error || low_rankformula_directgather64_rare_count >
                         P10DC_RANKFORMULA_DIRECTGATHER64_RARE_CAP) {
            std::cerr << "p10dc directgather64 compress failure owner=" << fixed
                      << " error=" << error
                      << " rare=" << low_rankformula_directgather64_rare_count << '\n';
            std::exit(811);
        }

        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER64,
                              &low_rankformula_directgather64,
                              sizeof(low_rankformula_directgather64)),
           "p10dc directgather64 primary ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE,
                              &low_rankformula_directgather64_rare,
                              sizeof(low_rankformula_directgather64_rare)),
           "p10dc directgather64 rare ptr");

        // The compressed runtime never reads the parent uint4 table. Free it
        // immediately so descriptor compression actually reduces resident HBM.
        if (low_rankformula_directgather4) cudaFree(low_rankformula_directgather4);
        low_rankformula_directgather4 = nullptr;
        low_rankformula_directgather4_count = 0;
        low_rankformula_directgather4_capacity = 0;
        uint4* null4 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER4,
                              &null4, sizeof(null4)),
           "p10dc directgather64 clear uint4 ptr");

        const uint64_t primary_bytes = uint64_t(n) * sizeof(uint64_t);
        const uint64_t rare_bytes = uint64_t(low_rankformula_directgather64_rare_count) * sizeof(uint64_t);
        std::cerr << "p10dc_low_rankformula_directgather64 fixed_owner=" << fixed
                  << " descriptors=" << n
                  << " primary_bytes=" << primary_bytes
                  << " rare_entries=" << low_rankformula_directgather64_rare_count
                  << " rare_bytes=" << rare_bytes
                  << " resident_bytes=" << (primary_bytes + rare_bytes)
                  << " primary_descriptor_bytes=8"
                  << " source_rank_bits=15 rare_index_bits=16"
                  << " parent_uint4_freed=1\n";
    }

    void release() {
        if (low_rankformula_directgather64) cudaFree(low_rankformula_directgather64);
        if (low_rankformula_directgather64_rare) cudaFree(low_rankformula_directgather64_rare);
        low_rankformula_directgather64 = nullptr;
        low_rankformula_directgather64_rare = nullptr;
        low_rankformula_directgather64_count = 0;
        low_rankformula_directgather64_capacity = 0;
        low_rankformula_directgather64_rare_count = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::release();
    }
};
