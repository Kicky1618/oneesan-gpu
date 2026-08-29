#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_directmap.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64.cuh"

#if !P10DC_RANKFORMULA_DIRECTGATHER64
#error "directgather64 table requires P10DC_RANKFORMULA_DIRECTGATHER64=1"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER
#error "directgather64 table requires DIRECTGATHER=1"
#endif
#ifndef P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS
#define P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS 0
#endif
static_assert(P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS == 0 ||
              P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS == 1,
              "P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS must be 0 or 1");

__global__ void p10dc_rankformula_directgather64_count_rare_kernel(
    const uint4* in, uint32_t n, uint32_t* rare_count, uint32_t* error
) {
    for (uint32_t i = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint32_t(gridDim.x) * blockDim.x) {
        const uint32_t count = in[i].w >> 16;
        if (count > 7u) atomicOr(error, 1u);
        else if (count > 3u) atomicAdd(rare_count, 1u);
    }
}

__global__ void p10dc_rankformula_directgather64_compress_kernel(
    const uint4* in, uint64_t* primary, uint64_t* rare,
    uint32_t n, uint32_t rare_capacity,
    uint32_t* rare_count, uint32_t* error
) {
    for (uint32_t i = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint32_t(gridDim.x) * blockDim.x) {
        const uint4 d = in[i];
        const uint32_t count = d.w >> 16;
        if (count > 7u) { atomicOr(error, 1u); continue; }
        uint32_t r[7] = {
            d.x & 0xffffu, d.x >> 16,
            d.y & 0xffffu, d.y >> 16,
            d.z & 0xffffu, d.z >> 16,
            d.w & 0xffffu
        };
#if P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS
        // Each destination sums these sources modulo p, so source order is
        // algebraically irrelevant.  Canonical ascending order aligns the
        // j-th gather of neighboring destination ranks much better than the
        // original L-endpoint enumeration order and can reduce sector spread.
#pragma unroll
        for (uint32_t a = 1; a < 7u; ++a) {
            if (a >= count) break;
            const uint32_t x = r[a];
            uint32_t b = a;
#pragma unroll
            while (b && r[b - 1u] > x) {
                r[b] = r[b - 1u];
                --b;
            }
            r[b] = x;
        }
#endif
        if ((count > 0u && r[0] >= 32768u) || (count > 1u && r[1] >= 32768u) ||
            (count > 2u && r[2] >= 32768u) || (count > 3u && r[3] >= 32768u) ||
            (count > 4u && r[4] >= 32768u) || (count > 5u && r[5] >= 32768u) ||
            (count > 6u && r[6] >= 32768u)) {
            atomicOr(error, 2u); continue;
        }
        uint32_t rare_ix = 0;
        if (count > 3u) {
            rare_ix = atomicAdd(rare_count, 1u);
            if (rare_ix >= rare_capacity || rare_ix >= 65536u) {
                atomicOr(error, 4u); continue;
            }
            rare[rare_ix] = uint64_t(r[3]) |
                (uint64_t(r[4]) << 15) |
                (uint64_t(r[5]) << 30) |
                (uint64_t(r[6]) << 45);
        }
        primary[i] = uint64_t(r[0]) |
            (uint64_t(r[1]) << 15) |
            (uint64_t(r[2]) << 30) |
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
    size_t low_rankformula_directgather64_rare_capacity = 0;
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

        uint32_t *d_rare_count = nullptr, *d_error = nullptr;
        ck(cudaMalloc(&d_rare_count, sizeof(uint32_t)), "p10dc directgather64 count alloc");
        ck(cudaMalloc(&d_error, sizeof(uint32_t)), "p10dc directgather64 error alloc");
        ck(cudaMemset(d_rare_count, 0, sizeof(uint32_t)), "p10dc directgather64 count zero");
        ck(cudaMemset(d_error, 0, sizeof(uint32_t)), "p10dc directgather64 error zero");

        const uint32_t threads = 256;
        const uint32_t blocks = n ? std::min<uint32_t>(4096u,
            (uint32_t(n) + threads - 1u) / threads) : 0u;
        if (n) {
            p10dc_rankformula_directgather64_count_rare_kernel<<<blocks, threads>>>(
                low_rankformula_directgather4, uint32_t(n), d_rare_count, d_error);
            ck(cudaGetLastError(), "p10dc directgather64 rare count launch");
            ck(cudaDeviceSynchronize(), "p10dc directgather64 rare count sync");
        }
        uint32_t count_error = 0;
        ck(cudaMemcpy(&low_rankformula_directgather64_rare_count, d_rare_count,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc directgather64 rare count copy");
        ck(cudaMemcpy(&count_error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc directgather64 count error copy");
        if (count_error || low_rankformula_directgather64_rare_count >= 65536u) {
            std::cerr << "p10dc directgather64 rare count failure owner=" << fixed
                      << " error=" << count_error
                      << " rare=" << low_rankformula_directgather64_rare_count << '\n';
            std::exit(811);
        }

        const size_t rare_needed = low_rankformula_directgather64_rare_count;
        if (rare_needed > low_rankformula_directgather64_rare_capacity) {
            if (low_rankformula_directgather64_rare) cudaFree(low_rankformula_directgather64_rare);
            low_rankformula_directgather64_rare = nullptr;
            low_rankformula_directgather64_rare_capacity = rare_needed;
            if (rare_needed)
                ck(cudaMalloc(&low_rankformula_directgather64_rare,
                              rare_needed * sizeof(uint64_t)),
                   "p10dc directgather64 exact rare alloc");
        }

        ck(cudaMemset(d_rare_count, 0, sizeof(uint32_t)), "p10dc directgather64 write count zero");
        ck(cudaMemset(d_error, 0, sizeof(uint32_t)), "p10dc directgather64 write error zero");
        if (n) {
            p10dc_rankformula_directgather64_compress_kernel<<<blocks, threads>>>(
                low_rankformula_directgather4,
                low_rankformula_directgather64,
                low_rankformula_directgather64_rare,
                uint32_t(n), uint32_t(rare_needed), d_rare_count, d_error);
            ck(cudaGetLastError(), "p10dc directgather64 compress launch");
            ck(cudaDeviceSynchronize(), "p10dc directgather64 compress sync");
        }
        uint32_t written_rare = 0, error = 0;
        ck(cudaMemcpy(&written_rare, d_rare_count, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc directgather64 written count copy");
        ck(cudaMemcpy(&error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc directgather64 error copy");
        cudaFree(d_rare_count); cudaFree(d_error);
        if (error || written_rare != low_rankformula_directgather64_rare_count) {
            std::cerr << "p10dc directgather64 compress failure owner=" << fixed
                      << " error=" << error
                      << " expected_rare=" << low_rankformula_directgather64_rare_count
                      << " written_rare=" << written_rare << '\n';
            std::exit(812);
        }

        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER64,
                              &low_rankformula_directgather64,
                              sizeof(low_rankformula_directgather64)),
           "p10dc directgather64 primary ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE,
                              &low_rankformula_directgather64_rare,
                              sizeof(low_rankformula_directgather64_rare)),
           "p10dc directgather64 rare ptr");

        // DIRECTGATHER64 is a fully resolved runtime path.  The uint4 parent
        // descriptors, dense direct rank map, and GROUP61 locator tables were
        // construction aids only; retaining them wastes HBM and cache tags.
        const uint64_t freed_uint4_bytes =
            uint64_t(low_rankformula_directgather4_count) * sizeof(uint4);
        const uint64_t freed_directmap_bytes =
            uint64_t(low_rankformula_direct64_count) * sizeof(uint64_t);
        const uint64_t freed_group_bytes =
            uint64_t(low_rankformula_nometa4_group64_count) * sizeof(uint64_t);
        const uint64_t freed_block_bytes =
            uint64_t(low_rankformula_nometa4_block16_count) * sizeof(uint16_t);

        if (low_rankformula_directgather4) cudaFree(low_rankformula_directgather4);
        low_rankformula_directgather4 = nullptr;
        low_rankformula_directgather4_count = 0;
        low_rankformula_directgather4_capacity = 0;
        uint4* null4 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER4,
                              &null4, sizeof(null4)),
           "p10dc directgather64 clear uint4 ptr");

        if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
        low_rankformula_direct64 = nullptr;
        low_rankformula_direct64_count = 0;
        low_rankformula_direct64_capacity = 0;
        uint64_t* null64 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA_DIRECT64,
                              &null64, sizeof(null64)),
           "p10dc directgather64 clear directmap ptr");

        if (low_rankformula_nometa4_group64) cudaFree(low_rankformula_nometa4_group64);
        if (low_rankformula_nometa4_block16) cudaFree(low_rankformula_nometa4_block16);
        low_rankformula_nometa4_group64 = nullptr;
        low_rankformula_nometa4_block16 = nullptr;
        low_rankformula_nometa4_group64_count = 0;
        low_rankformula_nometa4_block16_count = 0;
        low_rankformula_nometa4_group64_capacity = 0;
        low_rankformula_nometa4_block16_capacity = 0;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64,
                              &null64, sizeof(null64)),
           "p10dc directgather64 clear group ptr");
        uint16_t* null16 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16,
                              &null16, sizeof(null16)),
           "p10dc directgather64 clear block ptr");

        const uint64_t primary_bytes = uint64_t(n) * sizeof(uint64_t);
        const uint64_t rare_bytes = uint64_t(low_rankformula_directgather64_rare_count) * sizeof(uint64_t);
        std::cerr << "p10dc_low_rankformula_directgather64 fixed_owner=" << fixed
                  << " descriptors=" << n
                  << " primary_bytes=" << primary_bytes
                  << " rare_entries=" << low_rankformula_directgather64_rare_count
                  << " rare_bytes=" << rare_bytes
                  << " resident_bytes=" << (primary_bytes + rare_bytes)
                  << " rare_capacity_entries=" << low_rankformula_directgather64_rare_capacity
                  << " primary_descriptor_bytes=8"
                  << " source_rank_bits=15 rare_index_bits=16"
                  << " source_ranks_sorted=" << P10DC_RANKFORMULA_DIRECTGATHER_SORT_RANKS
                  << " exact_rare_allocation=1"
                  << " parent_uint4_freed=1"
                  << " directmap_freed=1"
                  << " group_locator_freed=1"
                  << " construction_bytes_freed="
                  << (freed_uint4_bytes + freed_directmap_bytes + freed_group_bytes + freed_block_bytes)
                  << '\n';
    }

    void release() {
        if (low_rankformula_directgather64) cudaFree(low_rankformula_directgather64);
        if (low_rankformula_directgather64_rare) cudaFree(low_rankformula_directgather64_rare);
        low_rankformula_directgather64 = nullptr;
        low_rankformula_directgather64_rare = nullptr;
        low_rankformula_directgather64_count = 0;
        low_rankformula_directgather64_capacity = 0;
        low_rankformula_directgather64_rare_capacity = 0;
        low_rankformula_directgather64_rare_count = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::release();
    }
};
