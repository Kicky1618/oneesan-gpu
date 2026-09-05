#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_warpshare.cuh"

// B300 has enormous HBM bandwidth.  The compact GROUP61 locator saves a tiny
// amount of metadata, but burns integer/control instructions on every HIGH
// element (block16 lookup, subgroup shuffles/ballots and successor scanning).
// This experimental backend expands the already host-resolved groups into one
// packed 64-bit entry per LOW rank.  The hot resolver becomes one coalesced
// read plus bit extraction.
static_assert(P10DC_RANKFORMULA_NOMETA_GROUP61 == 1,
              "direct64 resolver currently requires GROUP61 layout");

__constant__ uint64_t* D_P10DC_LOW_RANKFORMULA_DIRECT64;

__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa_resolve_direct64(uint32_t h, uint32_t rank) {
    const uint64_t e = __ldg(D_P10DC_LOW_RANKFORMULA_DIRECT64 +
                             D_P10DC_LOW_PREKEY_HOFF[h] + rank);
    const uint32_t start = uint32_t(e) & 0x7fffu;
    const uint32_t source_base = uint32_t(e >> 15) & 0x7fffu;
    const uint32_t lcount = uint32_t(e >> 30) & 0x07u;
    const uint32_t abstract_off = uint32_t(e >> 33) & 0x1fffu;
    return P10DCRankFormulaNometa4Resolved{
        h + 2u * lcount, start, int(source_base) - int(start),
        abstract_off, source_base};
}

struct BucketFusedDirectHighRowsRankFormulaNometa4Direct64Tables
    : BucketFusedDirectHighRowsRankFormulaNometa4Tables {
    uint64_t* low_rankformula_direct64 = nullptr;
    size_t low_rankformula_direct64_count = 0;
    size_t low_rankformula_direct64_capacity = 0;

    static uint64_t pack_direct64(uint32_t start, uint32_t source_base,
                                  uint32_t lcount, uint32_t abstract_off) {
        if (start >= (1u << 15) || source_base >= (1u << 15) ||
            lcount >= (1u << 3) || abstract_off >= (1u << 13)) {
            std::cerr << "p10dc direct64 pack overflow start=" << start
                      << " source_base=" << source_base
                      << " lcount=" << lcount
                      << " abstract_off=" << abstract_off << '\n';
            std::exit(781);
        }
        return uint64_t(start) |
               (uint64_t(source_base) << 15) |
               (uint64_t(lcount) << 30) |
               (uint64_t(abstract_off) << 33);
    }

    void bind_owner(uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
                    const std::array<Count*, BUCKET_NGPU>& slot) {
        BucketFusedDirectHighRowsRankFormulaNometa4Tables::bind_owner(
            fixed, buckets, slot);
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc direct64 invalid owner=" << fixed << '\n';
            std::exit(782);
        }

        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_begin = f.low_code_off[owner_base];
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        struct G { uint32_t mask = 0, start = 0, end = 0; };
        std::array<std::vector<G>, P10DC_RANKFORMULA_NOMETA4_HEIGHTS> gh;
        std::vector<int32_t> absbase(
            size_t(P10DC_RANKFORMULA_NOMETA4_HEIGHTS) *
                P10DC_RANKFORMULA_NOMETA4_MASKS,
            -1);
        auto bref = [&](uint32_t h, uint32_t mask) -> int32_t& {
            return absbase[size_t(h) * P10DC_RANKFORMULA_NOMETA4_MASKS + mask];
        };

        for (uint32_t h = 0; h < P10DC_RANKFORMULA_NOMETA4_HEIGHTS; ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            if (a < owner_begin || a > b || b > owner_end) {
                std::cerr << "p10dc direct64 height range invalid owner=" << fixed
                          << " h=" << h << " a=" << a << " b=" << b << '\n';
                std::exit(783);
            }
            uint32_t prev = 0;
            bool have = false;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t mask = code_mask(f.low_codes[i]);
                if (have && mask < prev) {
                    std::cerr << "p10dc direct64 mask order failure owner=" << fixed
                              << " h=" << h << '\n';
                    std::exit(784);
                }
                if (!have || mask != prev) {
                    const uint32_t start = i - a;
                    if (!gh[h].empty()) gh[h].back().end = start;
                    gh[h].push_back(G{mask, start, 0});
                    bref(h, mask) = int32_t(start);
                    prev = mask;
                    have = true;
                }
            }
            if (!gh[h].empty()) gh[h].back().end = b - a;
        }

        std::vector<uint64_t> direct;
        direct.reserve(size_t(owner_end - owner_begin));
        size_t groups = 0;
        uint32_t max_count = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            if (h >= P10DC_RANKFORMULA_NOMETA4_HEIGHTS) continue;
            for (const G& z : gh[h]) {
                if (z.end <= z.start) std::exit(785);
                const uint32_t n = uint32_t(__builtin_popcount(z.mask));
                if (n < h || ((n - h) & 1u)) {
                    std::cerr << "p10dc direct64 parity failure owner=" << fixed
                              << " h=" << h << " n=" << n << '\n';
                    std::exit(786);
                }
                const uint32_t lcount = (n - h) >> 1;
                uint32_t source_base = 0;
                if (lcount) {
                    if (h + 2u >= P10DC_RANKFORMULA_NOMETA4_HEIGHTS ||
                        bref(h + 2u, z.mask) < 0) {
                        std::cerr << "p10dc direct64 missing successor owner=" << fixed
                                  << " h=" << h << " mask=" << z.mask << '\n';
                        std::exit(787);
                    }
                    source_base = uint32_t(bref(h + 2u, z.mask));
                }
                const uint32_t abstract_off =
                    p10dc_rankformula_nometa4_abstract_off_host(n, h);
                const uint64_t e = pack_direct64(
                    z.start, source_base, lcount, abstract_off);
                const uint32_t count = z.end - z.start;
                max_count = std::max(max_count, count);
                direct.insert(direct.end(), count, e);
                ++groups;
            }
        }

        if (direct.size() != size_t(owner_end - owner_begin)) {
            std::cerr << "p10dc direct64 compact size mismatch owner=" << fixed
                      << " got=" << direct.size()
                      << " expected=" << (owner_end - owner_begin) << '\n';
            std::exit(788);
        }
        if (direct.size() != low_prekey_count) {
            std::cerr << "p10dc direct64/prekey size mismatch owner=" << fixed
                      << " direct=" << direct.size()
                      << " prekey=" << low_prekey_count << '\n';
            std::exit(789);
        }

        low_rankformula_direct64_count = direct.size();
        if (low_rankformula_direct64_count > low_rankformula_direct64_capacity) {
            if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
            low_rankformula_direct64 = nullptr;
            low_rankformula_direct64_capacity = low_rankformula_direct64_count;
            if (low_rankformula_direct64_capacity)
                ck(cudaMalloc(&low_rankformula_direct64,
                              low_rankformula_direct64_capacity * sizeof(uint64_t)),
                   "p10dc direct64 alloc");
        }
        if (!direct.empty())
            ck(cudaMemcpy(low_rankformula_direct64, direct.data(),
                          direct.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "p10dc direct64 H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_DIRECT64,
                              &low_rankformula_direct64,
                              sizeof(low_rankformula_direct64)),
           "p10dc direct64 ptr");

        std::cerr << "p10dc_low_rankformula_direct64 fixed_owner=" << fixed
                  << " entries=" << low_rankformula_direct64_count
                  << " groups=" << groups
                  << " bytes=" << low_rankformula_direct64_count * sizeof(uint64_t)
                  << " mib=" << double(low_rankformula_direct64_count * sizeof(uint64_t)) /
                                   double(1 << 20)
                  << " max_group_count=" << max_count
                  << " hot_locator_loads=1 hot_shuffle=0 hot_ballot=0 hot_scan=0\n";
    }

    void release() {
        if (low_rankformula_direct64) cudaFree(low_rankformula_direct64);
        low_rankformula_direct64 = nullptr;
        low_rankformula_direct64_count = 0;
        low_rankformula_direct64_capacity = 0;
        BucketFusedDirectHighRowsRankFormulaNometa4Tables::release();
    }
};
