#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_directmap.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather_sparse64.cuh"

#if !P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#error "sparse64 table requires P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64=1"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
#error "sparse64 currently targets depth-major directgather"
#endif

__global__ void p10dc_rankformula_directgather_sparse64_bitmap_kernel(
    const uint4* in, uint32_t* bits, uint32_t n,
    uint32_t* rare_count, uint32_t* error
) {
    for (uint32_t i = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint32_t(gridDim.x) * blockDim.x) {
        const uint32_t count = in[i].w >> 16;
        if (count > 7u) { atomicOr(error, 1u); continue; }
        if (count) atomicOr(bits + (i >> 5), 1u << (i & 31u));
        if (count > 3u) atomicAdd(rare_count, 1u);
    }
}

__global__ void p10dc_rankformula_directgather_sparse64_compact_kernel(
    const uint4* in, const P10DCDirectGather64Word* index64,
    P10DCDirectGather64Word* primary,
    P10DCDirectGather64Word* rare,
    uint32_t n, uint32_t compact_capacity, uint32_t rare_capacity,
    uint32_t* rare_count, uint32_t* error
) {
    for (uint32_t i = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint32_t(gridDim.x) * blockDim.x) {
        const uint4 d = in[i];
        const uint32_t count = d.w >> 16;
        if (!count) continue;
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

        const uint32_t wi = i >> 5;
        const uint32_t bit = i & 31u;
        const P10DCDirectGather64Word ix = index64[wi];
        const uint32_t mask = uint32_t(ix);
        const uint32_t prefix = uint32_t(ix >> 32);
        const uint32_t lower = bit ? (mask & ((1u << bit) - 1u)) : 0u;
        const uint32_t ci = prefix + uint32_t(__popc(lower));
        if (ci >= compact_capacity) { atomicOr(error, 8u); continue; }

        uint32_t rare_ix = 0u;
        if (count > 3u) {
            rare_ix = atomicAdd(rare_count, 1u);
            if (rare_ix >= rare_capacity || rare_ix >= 65536u) {
                atomicOr(error, 4u); continue;
            }
            rare[rare_ix] = P10DCDirectGather64Word(r3) |
                (P10DCDirectGather64Word(r4) << 15) |
                (P10DCDirectGather64Word(r5) << 30) |
                (P10DCDirectGather64Word(r6) << 45);
        }
        primary[ci] = P10DCDirectGather64Word(r0) |
            (P10DCDirectGather64Word(r1) << 15) |
            (P10DCDirectGather64Word(r2) << 30) |
            (P10DCDirectGather64Word(count) << 45) |
            (P10DCDirectGather64Word(rare_ix) << 48);
    }
}

struct BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64Tables
    : BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables {
    P10DCDirectGather64Word* low_rankformula_sparse64_index = nullptr;
    P10DCDirectGather64Word* low_rankformula_sparse64_primary = nullptr;
    P10DCDirectGather64Word* low_rankformula_sparse64_rare = nullptr;
    size_t low_rankformula_sparse64_index_count = 0;
    size_t low_rankformula_sparse64_primary_count = 0;
    size_t low_rankformula_sparse64_rare_count = 0;
    size_t low_rankformula_sparse64_index_capacity = 0;
    size_t low_rankformula_sparse64_primary_capacity = 0;
    size_t low_rankformula_sparse64_rare_capacity = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::bind_owner(
            fixed, buckets, slot);
        const size_t n = low_rankformula_directgather4_count;
        if (n >= size_t(1u << 32)) std::exit(820);
        const size_t words = (n + 31u) >> 5;

        uint32_t* d_bits = nullptr;
        uint32_t *d_rare_count = nullptr, *d_error = nullptr;
        if (words) {
            ck(cudaMalloc(&d_bits, words * sizeof(uint32_t)),
               "p10dc sparse64 bitmap alloc");
            ck(cudaMemset(d_bits, 0, words * sizeof(uint32_t)),
               "p10dc sparse64 bitmap zero");
        }
        ck(cudaMalloc(&d_rare_count, sizeof(uint32_t)), "p10dc sparse64 rare count alloc");
        ck(cudaMalloc(&d_error, sizeof(uint32_t)), "p10dc sparse64 error alloc");
        ck(cudaMemset(d_rare_count, 0, sizeof(uint32_t)), "p10dc sparse64 rare count zero");
        ck(cudaMemset(d_error, 0, sizeof(uint32_t)), "p10dc sparse64 error zero");

        const uint32_t threads = 256;
        const uint32_t blocks = n ? std::min<uint32_t>(4096u,
            (uint32_t(n) + threads - 1u) / threads) : 0u;
        if (n) {
            p10dc_rankformula_directgather_sparse64_bitmap_kernel<<<blocks, threads>>>(
                low_rankformula_directgather4, d_bits, uint32_t(n),
                d_rare_count, d_error);
            ck(cudaGetLastError(), "p10dc sparse64 bitmap launch");
            ck(cudaDeviceSynchronize(), "p10dc sparse64 bitmap sync");
        }

        uint32_t error = 0, rare_count = 0;
        ck(cudaMemcpy(&rare_count, d_rare_count, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc sparse64 rare count copy");
        ck(cudaMemcpy(&error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc sparse64 bitmap error copy");
        if (error || rare_count >= 65536u) {
            std::cerr << "p10dc sparse64 bitmap failure owner=" << fixed
                      << " error=" << error << " rare=" << rare_count << '\n';
            std::exit(821);
        }

        std::vector<uint32_t> hbits(words);
        if (words)
            ck(cudaMemcpy(hbits.data(), d_bits, words * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               "p10dc sparse64 bitmap D2H");
        if (d_bits) cudaFree(d_bits);

        std::vector<P10DCDirectGather64Word> hindex(words);
        uint64_t nonzero64 = 0;
        for (size_t w = 0; w < words; ++w) {
            if (nonzero64 >= (1ull << 32)) std::exit(822);
            hindex[w] = P10DCDirectGather64Word(hbits[w]) |
                        (P10DCDirectGather64Word(uint32_t(nonzero64)) << 32);
            nonzero64 += uint32_t(__builtin_popcount(hbits[w]));
        }
        if (nonzero64 >= (1ull << 32)) std::exit(823);
        const size_t nonzero = size_t(nonzero64);

        low_rankformula_sparse64_index_count = words;
        low_rankformula_sparse64_primary_count = nonzero;
        low_rankformula_sparse64_rare_count = rare_count;
        if (words > low_rankformula_sparse64_index_capacity) {
            if (low_rankformula_sparse64_index) cudaFree(low_rankformula_sparse64_index);
            low_rankformula_sparse64_index = nullptr;
            low_rankformula_sparse64_index_capacity = words;
            if (words) ck(cudaMalloc(&low_rankformula_sparse64_index,
                                     words * sizeof(P10DCDirectGather64Word)),
                          "p10dc sparse64 index alloc");
        }
        if (nonzero > low_rankformula_sparse64_primary_capacity) {
            if (low_rankformula_sparse64_primary) cudaFree(low_rankformula_sparse64_primary);
            low_rankformula_sparse64_primary = nullptr;
            low_rankformula_sparse64_primary_capacity = nonzero;
            if (nonzero) ck(cudaMalloc(&low_rankformula_sparse64_primary,
                                       nonzero * sizeof(P10DCDirectGather64Word)),
                            "p10dc sparse64 primary alloc");
        }
        if (size_t(rare_count) > low_rankformula_sparse64_rare_capacity) {
            if (low_rankformula_sparse64_rare) cudaFree(low_rankformula_sparse64_rare);
            low_rankformula_sparse64_rare = nullptr;
            low_rankformula_sparse64_rare_capacity = rare_count;
            if (rare_count) ck(cudaMalloc(&low_rankformula_sparse64_rare,
                                          size_t(rare_count) * sizeof(P10DCDirectGather64Word)),
                               "p10dc sparse64 rare alloc");
        }
        if (words)
            ck(cudaMemcpy(low_rankformula_sparse64_index, hindex.data(),
                          words * sizeof(P10DCDirectGather64Word), cudaMemcpyHostToDevice),
               "p10dc sparse64 index H2D");

        ck(cudaMemset(d_rare_count, 0, sizeof(uint32_t)), "p10dc sparse64 write count zero");
        ck(cudaMemset(d_error, 0, sizeof(uint32_t)), "p10dc sparse64 write error zero");
        if (n) {
            p10dc_rankformula_directgather_sparse64_compact_kernel<<<blocks, threads>>>(
                low_rankformula_directgather4,
                low_rankformula_sparse64_index,
                low_rankformula_sparse64_primary,
                low_rankformula_sparse64_rare,
                uint32_t(n), uint32_t(nonzero), rare_count,
                d_rare_count, d_error);
            ck(cudaGetLastError(), "p10dc sparse64 compact launch");
            ck(cudaDeviceSynchronize(), "p10dc sparse64 compact sync");
        }
        uint32_t written_rare = 0;
        ck(cudaMemcpy(&written_rare, d_rare_count, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc sparse64 written rare copy");
        ck(cudaMemcpy(&error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost),
           "p10dc sparse64 compact error copy");
        cudaFree(d_rare_count); cudaFree(d_error);
        if (error || written_rare != rare_count) {
            std::cerr << "p10dc sparse64 compact failure owner=" << fixed
                      << " error=" << error << " expected_rare=" << rare_count
                      << " written_rare=" << written_rare << '\n';
            std::exit(824);
        }

        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX,
                              &low_rankformula_sparse64_index,
                              sizeof(low_rankformula_sparse64_index)),
           "p10dc sparse64 index ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY,
                              &low_rankformula_sparse64_primary,
                              sizeof(low_rankformula_sparse64_primary)),
           "p10dc sparse64 primary ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE,
                              &low_rankformula_sparse64_rare,
                              sizeof(low_rankformula_sparse64_rare)),
           "p10dc sparse64 rare ptr");

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
                              &null4, sizeof(null4)), "p10dc sparse64 clear uint4 ptr");

        if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
        low_rankformula_direct64 = nullptr;
        low_rankformula_direct64_count = 0;
        low_rankformula_direct64_capacity = 0;
        uint64_t* null64 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA_DIRECT64,
                              &null64, sizeof(null64)), "p10dc sparse64 clear directmap ptr");

        if (low_rankformula_nometa4_group64) cudaFree(low_rankformula_nometa4_group64);
        if (low_rankformula_nometa4_block16) cudaFree(low_rankformula_nometa4_block16);
        low_rankformula_nometa4_group64 = nullptr;
        low_rankformula_nometa4_block16 = nullptr;
        low_rankformula_nometa4_group64_count = 0;
        low_rankformula_nometa4_block16_count = 0;
        low_rankformula_nometa4_group64_capacity = 0;
        low_rankformula_nometa4_block16_capacity = 0;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64,
                              &null64, sizeof(null64)), "p10dc sparse64 clear group ptr");
        uint16_t* null16 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16,
                              &null16, sizeof(null16)), "p10dc sparse64 clear block ptr");

        const uint64_t index_bytes = uint64_t(words) * sizeof(P10DCDirectGather64Word);
        const uint64_t primary_bytes = uint64_t(nonzero) * sizeof(P10DCDirectGather64Word);
        const uint64_t rare_bytes = uint64_t(rare_count) * sizeof(P10DCDirectGather64Word);
        std::cerr << "p10dc_low_rankformula_directgather_sparse64 fixed_owner=" << fixed
                  << " descriptors=" << n
                  << " nonzero_descriptors=" << nonzero
                  << " zero_descriptors=" << (n - nonzero)
                  << " nonzero_fraction=" << (n ? double(nonzero) / double(n) : 0.0)
                  << " index_words=" << words
                  << " index_bytes=" << index_bytes
                  << " primary_bytes=" << primary_bytes
                  << " rare_entries=" << rare_count
                  << " rare_bytes=" << rare_bytes
                  << " resident_bytes=" << (index_bytes + primary_bytes + rare_bytes)
                  << " zero_primary_loads=1"
                  << " compact_prefix_popc=1"
                  << " construction_bytes_freed="
                  << (freed_uint4_bytes + freed_directmap_bytes + freed_group_bytes + freed_block_bytes)
                  << '\n';
    }

    void release() {
        if (low_rankformula_sparse64_index) cudaFree(low_rankformula_sparse64_index);
        if (low_rankformula_sparse64_primary) cudaFree(low_rankformula_sparse64_primary);
        if (low_rankformula_sparse64_rare) cudaFree(low_rankformula_sparse64_rare);
        low_rankformula_sparse64_index = nullptr;
        low_rankformula_sparse64_primary = nullptr;
        low_rankformula_sparse64_rare = nullptr;
        low_rankformula_sparse64_index_count = 0;
        low_rankformula_sparse64_primary_count = 0;
        low_rankformula_sparse64_rare_count = 0;
        low_rankformula_sparse64_index_capacity = 0;
        low_rankformula_sparse64_primary_capacity = 0;
        low_rankformula_sparse64_rare_capacity = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables::release();
    }
};
