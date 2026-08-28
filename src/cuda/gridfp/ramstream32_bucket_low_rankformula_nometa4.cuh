#pragma once

#include "ramstream32_bucket_low_prekey.cuh"

static constexpr uint32_t P10DC_RANKFORMULA_NOMETA4_BLOCK = 4u;
static constexpr uint32_t P10DC_RANKFORMULA_NOMETA4_MASKS = 1u << LOW_LUT_K;
static constexpr uint32_t P10DC_RANKFORMULA_NOMETA4_HEIGHTS = LOW_LUT_K + 2u;
static_assert(LOW_LUT_K <= 14, "rankformula nometa4 assumes LOW_LUT_K<=14");
static_assert(P10DC_RANKFORMULA_NOMETA4_HEIGHTS <= uint32_t(MAXW + 2));

// group64: [15:0] local group start, [29:16] support mask,
// [47:32] signed (source_base-dest_base).  One sentinel is appended per height.
__constant__ uint64_t* D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64;
__constant__ uint16_t* D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16;
__constant__ uint32_t D_P10DC_LOW_RANKFORMULA_NOMETA4_GOFF[MAXW + 2];
__constant__ uint32_t D_P10DC_LOW_RANKFORMULA_NOMETA4_BOFF[MAXW + 2];

__device__ __forceinline__ uint32_t p10dc_rankformula_nometa4_group_start(uint64_t x) {
    return uint32_t(x) & 0xffffu;
}
__device__ __forceinline__ uint32_t p10dc_rankformula_nometa4_group_mask(uint64_t x) {
    return (uint32_t(x >> 16) & (P10DC_RANKFORMULA_NOMETA4_MASKS - 1u));
}
__device__ __forceinline__ int p10dc_rankformula_nometa4_group_delta(uint64_t x) {
    return int(int16_t(uint16_t(x >> 32)));
}

struct P10DCRankFormulaNometa4Resolved {
    uint32_t mask = 0;
    uint32_t start = 0;
    int base_delta = 0;
};

__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa4_resolve(uint32_t h, uint32_t rank) {
    const uint32_t bi = D_P10DC_LOW_RANKFORMULA_NOMETA4_BOFF[h] +
                        rank / P10DC_RANKFORMULA_NOMETA4_BLOCK;
    uint32_t gi = uint32_t(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16[bi]);
    uint64_t e = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[gi];
#pragma unroll
    for (int k = 0; k < int(P10DC_RANKFORMULA_NOMETA4_BLOCK - 1u); ++k) {
        const uint64_t n = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[gi + 1u];
        if (rank >= p10dc_rankformula_nometa4_group_start(n)) {
            ++gi;
            e = n;
        }
    }
    return P10DCRankFormulaNometa4Resolved{
        p10dc_rankformula_nometa4_group_mask(e),
        p10dc_rankformula_nometa4_group_start(e),
        p10dc_rankformula_nometa4_group_delta(e)};
}

struct BucketFusedDirectHighRowsRankFormulaNometa4Tables
    : BucketFusedDirectHighRowsPrekeyTables {
    uint64_t* low_rankformula_nometa4_group64 = nullptr;
    uint16_t* low_rankformula_nometa4_block16 = nullptr;
    size_t low_rankformula_nometa4_group64_count = 0;
    size_t low_rankformula_nometa4_block16_count = 0;
    size_t low_rankformula_nometa4_group64_capacity = 0;
    size_t low_rankformula_nometa4_block16_capacity = 0;

    static uint32_t code_mask(uint32_t code) {
        uint32_t mask = 0;
        for (int pos = 0; pos < LOW_LUT_K; ++pos)
            if (((code >> (2 * pos)) & 3u) != 0u) mask |= 1u << pos;
        return mask;
    }

    static uint64_t pack_group(uint32_t start, uint32_t mask, int delta) {
        if (start >= 65536u || mask >= P10DC_RANKFORMULA_NOMETA4_MASKS ||
            delta < -32768 || delta > 32767) std::exit(760);
        return uint64_t(uint16_t(start)) |
               (uint64_t(mask) << 16) |
               (uint64_t(uint16_t(int16_t(delta))) << 32);
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyTables::bind_owner(fixed, buckets, slot);
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc rankformula nometa4 invalid owner=" << fixed << '\n';
            std::exit(761);
        }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        struct G { uint16_t mask, start; };
        std::array<std::vector<G>, P10DC_RANKFORMULA_NOMETA4_HEIGHTS> gh;
        std::vector<int32_t> absbase(
            size_t(P10DC_RANKFORMULA_NOMETA4_HEIGHTS) * P10DC_RANKFORMULA_NOMETA4_MASKS,
            -1);
        auto bref = [&](uint32_t h, uint32_t mask) -> int32_t& {
            return absbase[size_t(h) * P10DC_RANKFORMULA_NOMETA4_MASKS + mask];
        };
        std::array<uint32_t, P10DC_RANKFORMULA_NOMETA4_HEIGHTS> ranks{};

        for (uint32_t h = 0; h < P10DC_RANKFORMULA_NOMETA4_HEIGHTS; ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            uint32_t prev = 0;
            bool have = false;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t mask = code_mask(f.low_codes[i]);
                if (have && mask < prev) {
                    std::cerr << "p10dc rankformula nometa4 mask order failure owner="
                              << fixed << " h=" << h << '\n';
                    std::exit(762);
                }
                if (!have || mask != prev) {
                    const uint32_t start = i - a;
                    if (start >= 65536u) std::exit(763);
                    gh[h].push_back(G{uint16_t(mask), uint16_t(start)});
                    bref(h, mask) = int32_t(start);
                    prev = mask;
                    have = true;
                }
            }
            ranks[h] = b - a;
        }

        std::vector<uint64_t> groups;
        std::vector<uint16_t> blocks;
        std::array<uint32_t, MAXW + 2> goff{}, boff{};
        size_t real_groups = 0;
        int min_delta = 32767, max_delta = -32768;
        size_t delta_rows = 0;
        uint32_t max_groups_per_block = 0, max_locator_steps = 0;

        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            goff[h] = uint32_t(groups.size());
            boff[h] = uint32_t(blocks.size());
            if (h >= P10DC_RANKFORMULA_NOMETA4_HEIGHTS) continue;
            const auto& v = gh[h];
            if (v.empty()) {
                groups.push_back(pack_group(0u, 0u, 0));
                continue;
            }
            const uint32_t gbase = uint32_t(groups.size());
            for (const G& z : v) {
                int delta = 0;
                if (h + 2u < P10DC_RANKFORMULA_NOMETA4_HEIGHTS) {
                    const int32_t a = bref(h, z.mask);
                    const int32_t b = bref(h + 2u, z.mask);
                    if (a >= 0 && b >= 0) {
                        delta = int(b - a);
                        min_delta = std::min(min_delta, delta);
                        max_delta = std::max(max_delta, delta);
                        ++delta_rows;
                    }
                }
                groups.push_back(pack_group(z.start, z.mask, delta));
                ++real_groups;
            }
            // Sentinel makes the unrolled next-group tests safe for the final block.
            groups.push_back(pack_group(ranks[h], 0u, 0));

            uint32_t gi = 0;
            const uint32_t nb = (ranks[h] + P10DC_RANKFORMULA_NOMETA4_BLOCK - 1u) /
                                P10DC_RANKFORMULA_NOMETA4_BLOCK;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t lo = b * P10DC_RANKFORMULA_NOMETA4_BLOCK;
                const uint32_t hi = std::min(ranks[h], lo + P10DC_RANKFORMULA_NOMETA4_BLOCK);
                while (gi + 1u < v.size() && lo >= uint32_t(v[gi + 1u].start)) ++gi;
                const uint32_t global_gi = gbase + gi;
                if (global_gi >= 65536u) {
                    std::cerr << "p10dc rankformula nometa4 group16 overflow owner="
                              << fixed << " group=" << global_gi << '\n';
                    std::exit(764);
                }
                blocks.push_back(uint16_t(global_gi));
                uint32_t last = gi;
                while (last + 1u < v.size() && uint32_t(v[last + 1u].start) < hi) ++last;
                max_groups_per_block = std::max(max_groups_per_block, last - gi + 1u);
                max_locator_steps = std::max(max_locator_steps, last - gi);
            }
        }
        if (max_locator_steps > P10DC_RANKFORMULA_NOMETA4_BLOCK - 1u ||
            max_groups_per_block > P10DC_RANKFORMULA_NOMETA4_BLOCK) {
            std::cerr << "p10dc rankformula nometa4 locator bound failure owner=" << fixed
                      << " max_steps=" << max_locator_steps
                      << " max_groups=" << max_groups_per_block << '\n';
            std::exit(765);
        }

        low_rankformula_nometa4_group64_count = groups.size();
        low_rankformula_nometa4_block16_count = blocks.size();
        if (low_rankformula_nometa4_group64_count > low_rankformula_nometa4_group64_capacity) {
            if (low_rankformula_nometa4_group64) cudaFree(low_rankformula_nometa4_group64);
            low_rankformula_nometa4_group64 = nullptr;
            low_rankformula_nometa4_group64_capacity = low_rankformula_nometa4_group64_count;
            if (low_rankformula_nometa4_group64_capacity)
                ck(cudaMalloc(&low_rankformula_nometa4_group64,
                              low_rankformula_nometa4_group64_capacity * sizeof(uint64_t)),
                   "p10dc rankformula nometa4 group alloc");
        }
        if (low_rankformula_nometa4_block16_count > low_rankformula_nometa4_block16_capacity) {
            if (low_rankformula_nometa4_block16) cudaFree(low_rankformula_nometa4_block16);
            low_rankformula_nometa4_block16 = nullptr;
            low_rankformula_nometa4_block16_capacity = low_rankformula_nometa4_block16_count;
            if (low_rankformula_nometa4_block16_capacity)
                ck(cudaMalloc(&low_rankformula_nometa4_block16,
                              low_rankformula_nometa4_block16_capacity * sizeof(uint16_t)),
                   "p10dc rankformula nometa4 block alloc");
        }
        if (!groups.empty())
            ck(cudaMemcpy(low_rankformula_nometa4_group64, groups.data(),
                          groups.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "p10dc rankformula nometa4 group H2D");
        if (!blocks.empty())
            ck(cudaMemcpy(low_rankformula_nometa4_block16, blocks.data(),
                          blocks.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
               "p10dc rankformula nometa4 block H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64,
                              &low_rankformula_nometa4_group64,
                              sizeof(low_rankformula_nometa4_group64)),
           "p10dc rankformula nometa4 group ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16,
                              &low_rankformula_nometa4_block16,
                              sizeof(low_rankformula_nometa4_block16)),
           "p10dc rankformula nometa4 block ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_GOFF,
                              goff.data(), goff.size() * sizeof(uint32_t)),
           "p10dc rankformula nometa4 goff");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_NOMETA4_BOFF,
                              boff.data(), boff.size() * sizeof(uint32_t)),
           "p10dc rankformula nometa4 boff");

        // Parent prekey is only used to reuse the authoritative fixed-owner bind;
        // no per-code metadata remains resident in this backend.
        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr;
        low_prekey_count = 0;
        low_prekey_capacity = 0;
        uint32_t* null32 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &null32, sizeof(null32)),
           "p10dc rankformula nometa4 null prekey ptr");

        const size_t group_bytes = groups.size() * sizeof(uint64_t);
        const size_t block_bytes = blocks.size() * sizeof(uint16_t);
        std::cerr << "p10dc_low_rankformula_nometa4 fixed_owner=" << fixed
                  << " real_groups=" << real_groups
                  << " group_entries=" << groups.size()
                  << " group_bytes=" << group_bytes
                  << " blocks=" << blocks.size()
                  << " block_bytes=" << block_bytes
                  << " total_bytes=" << (group_bytes + block_bytes)
                  << " per_code_metadata_bytes=0"
                  << " block_size=" << P10DC_RANKFORMULA_NOMETA4_BLOCK
                  << " max_groups_per_block=" << max_groups_per_block
                  << " max_locator_steps=" << max_locator_steps
                  << " delta_rows=" << delta_rows
                  << " min_base_delta=" << (delta_rows ? min_delta : 0)
                  << " max_base_delta=" << (delta_rows ? max_delta : 0)
                  << " group_load_bytes=8 block_load_bytes=2 old_prekey_freed=1\n";
    }

    void release() {
        if (low_rankformula_nometa4_group64) cudaFree(low_rankformula_nometa4_group64);
        if (low_rankformula_nometa4_block16) cudaFree(low_rankformula_nometa4_block16);
        low_rankformula_nometa4_group64 = nullptr;
        low_rankformula_nometa4_block16 = nullptr;
        low_rankformula_nometa4_group64_count = 0;
        low_rankformula_nometa4_block16_count = 0;
        low_rankformula_nometa4_group64_capacity = 0;
        low_rankformula_nometa4_block16_capacity = 0;
        BucketFusedDirectHighRowsPrekeyTables::release();
    }
};
