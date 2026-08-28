#pragma once

#include "ramstream32_bucket_low_rankchunk32.cuh"

#ifndef P10DC_RANKFORMULA_SPARSE_BASE
#define P10DC_RANKFORMULA_SPARSE_BASE 1
#endif
#ifndef P10DC_RANKFORMULA_RAWCODE
#define P10DC_RANKFORMULA_RAWCODE 0
#endif
#ifndef P10DC_RANKFORMULA_BASE_DELTA
#define P10DC_RANKFORMULA_BASE_DELTA 0
#endif
static_assert(P10DC_RANKFORMULA_SPARSE_BASE == 0 || P10DC_RANKFORMULA_SPARSE_BASE == 1,
              "P10DC_RANKFORMULA_SPARSE_BASE must be 0 or 1");
static_assert(P10DC_RANKFORMULA_RAWCODE == 0 || P10DC_RANKFORMULA_RAWCODE == 1,
              "P10DC_RANKFORMULA_RAWCODE must be 0 or 1");
static_assert(P10DC_RANKFORMULA_BASE_DELTA == 0 || P10DC_RANKFORMULA_BASE_DELTA == 1,
              "P10DC_RANKFORMULA_BASE_DELTA must be 0 or 1");

static constexpr uint32_t P10DC_RANKFORMULA_MASKS = 1u << LOW_LUT_K;
static constexpr uint32_t P10DC_RANKFORMULA_HEIGHTS = LOW_LUT_K + 2u;
static constexpr uint16_t P10DC_RANKFORMULA_BASE_INVALID = 0xffffu;
static constexpr uint32_t P10DC_RANKFORMULA_RAWCODE_BITS = 2u * LOW_LUT_K;
static constexpr uint32_t P10DC_RANKFORMULA_RAWCODE_MASK =
    (1u << P10DC_RANKFORMULA_RAWCODE_BITS) - 1u;
static_assert(LOW_LUT_K <= 14, "rankformula assumes LOW_LUT_K<=14");
static_assert(P10DC_RANKFORMULA_RAWCODE_BITS <= 28u,
              "rankformula raw 2-bit metadata must fit uint32");
static_assert(P10DC_RANKCHUNK32_BYTEPACK == 0,
              "rankformula ternary metadata uses compact 23-bit CROSS5 chunks");
static_assert(P10DC_RANKFORMULA_HEIGHTS <= uint32_t(MAXW + 2));

__constant__ uint32_t* D_P10DC_LOW_RANKFORMULA_META32;
// In ordinary mode entries are uint16 owner-local group bases. In base-delta
// mode the same storage holds int16(source_base-dest_base), bit-cast to uint16.
__constant__ uint16_t* D_P10DC_LOW_RANKFORMULA_BASE16;
__constant__ uint16_t* D_P10DC_LOW_RANKFORMULA_MASK_SLOT16;
__constant__ uint32_t D_P10DC_LOW_RANKFORMULA_HOFF[MAXW + 2];

__device__ __forceinline__ uint32_t p10dc_low_rankformula_meta(
    uint32_t h, uint32_t rank
) {
    return D_P10DC_LOW_RANKFORMULA_META32[D_P10DC_LOW_RANKFORMULA_HOFF[h] + rank];
}

__device__ __forceinline__ uint32_t p10dc_low_rankformula_chunks(
    uint32_t h, uint32_t rank
) {
    return p10dc_low_rankformula_meta(h, rank);
}

__device__ __forceinline__ uint32_t p10dc_low_rankformula_base(
    uint32_t h, uint32_t mask
) {
    if (h >= P10DC_RANKFORMULA_HEIGHTS) return 0u;
#if P10DC_RANKFORMULA_SPARSE_BASE
    const uint32_t slot = uint32_t(D_P10DC_LOW_RANKFORMULA_MASK_SLOT16[mask]);
    return uint32_t(D_P10DC_LOW_RANKFORMULA_BASE16[
        size_t(slot) * P10DC_RANKFORMULA_HEIGHTS + h]);
#else
    return uint32_t(D_P10DC_LOW_RANKFORMULA_BASE16[
        size_t(h) * P10DC_RANKFORMULA_MASKS + mask]);
#endif
}

__device__ __forceinline__ void p10dc_low_rankformula_base_pair(
    uint32_t h, uint32_t mask, uint32_t& dest_base, uint32_t& source_base
) {
#if P10DC_RANKFORMULA_SPARSE_BASE
    const uint32_t slot = uint32_t(D_P10DC_LOW_RANKFORMULA_MASK_SLOT16[mask]);
    const uint16_t* row = D_P10DC_LOW_RANKFORMULA_BASE16 +
        size_t(slot) * P10DC_RANKFORMULA_HEIGHTS;
    dest_base = h < P10DC_RANKFORMULA_HEIGHTS ? uint32_t(row[h]) : 0u;
    const uint32_t sh = h + 2u;
    source_base = sh < P10DC_RANKFORMULA_HEIGHTS ? uint32_t(row[sh]) : 0u;
#else
    dest_base = h < P10DC_RANKFORMULA_HEIGHTS
        ? uint32_t(D_P10DC_LOW_RANKFORMULA_BASE16[
              size_t(h) * P10DC_RANKFORMULA_MASKS + mask])
        : 0u;
    const uint32_t sh = h + 2u;
    source_base = sh < P10DC_RANKFORMULA_HEIGHTS
        ? uint32_t(D_P10DC_LOW_RANKFORMULA_BASE16[
              size_t(sh) * P10DC_RANKFORMULA_MASKS + mask])
        : 0u;
#endif
}

__device__ __forceinline__ int p10dc_low_rankformula_base_delta(
    uint32_t h, uint32_t mask
) {
    if (h + 2u >= P10DC_RANKFORMULA_HEIGHTS) return 0;
#if P10DC_RANKFORMULA_SPARSE_BASE
    const uint32_t slot = uint32_t(D_P10DC_LOW_RANKFORMULA_MASK_SLOT16[mask]);
    const uint16_t raw = D_P10DC_LOW_RANKFORMULA_BASE16[
        size_t(slot) * P10DC_RANKFORMULA_HEIGHTS + h];
#else
    const uint16_t raw = D_P10DC_LOW_RANKFORMULA_BASE16[
        size_t(h) * P10DC_RANKFORMULA_MASKS + mask];
#endif
    return int(int16_t(raw));
}

struct BucketFusedDirectHighRowsRankFormulaTables
    : BucketFusedDirectHighRowsPrekeyTables {
    uint32_t* low_rankformula_meta32 = nullptr;
    uint16_t* low_rankformula_base16 = nullptr;
    uint16_t* low_rankformula_mask_slot16 = nullptr;
    size_t low_rankformula_meta32_count = 0;
    size_t low_rankformula_base16_count = 0;
    size_t low_rankformula_mask_slot16_count = 0;
    size_t low_rankformula_meta32_capacity = 0;
    size_t low_rankformula_base16_capacity = 0;
    size_t low_rankformula_mask_slot16_capacity = 0;
    uint32_t low_rankformula_owned_masks = 0;

    static uint32_t code_mask(uint32_t code) {
        uint32_t mask = 0;
        for (int pos = 0; pos < LOW_LUT_K; ++pos)
            if (((code >> (2 * pos)) & 3u) != 0u) mask |= 1u << pos;
        return mask;
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyTables::bind_owner(fixed, buckets, slot);
        if (!host_fused) {
            std::cerr << "p10dc rankformula missing host fused metadata\n";
            std::exit(730);
        }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::array<uint8_t, P10DC_RANKFORMULA_MASKS> owned{};
        for (uint32_t h = 0; h < P10DC_RANKFORMULA_HEIGHTS; ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            for (uint32_t i = a; i < b; ++i) owned[code_mask(f.low_codes[i])] = 1u;
        }
        std::vector<uint16_t> mask_slot(P10DC_RANKFORMULA_MASKS,
                                        P10DC_RANKFORMULA_BASE_INVALID);
        uint32_t owned_masks = 0;
        for (uint32_t mask = 0; mask < P10DC_RANKFORMULA_MASKS; ++mask) {
            if (!owned[mask]) continue;
            if (owned_masks >= P10DC_RANKFORMULA_BASE_INVALID) std::exit(731);
            mask_slot[mask] = uint16_t(owned_masks++);
        }
        if (!owned_masks) {
            std::cerr << "p10dc rankformula owner has no masks owner=" << fixed << '\n';
            std::exit(732);
        }

        std::array<uint32_t, MAXW + 2> hoff{};
        std::vector<uint32_t> meta;
        const size_t base_cols = P10DC_RANKFORMULA_SPARSE_BASE
            ? size_t(owned_masks) : size_t(P10DC_RANKFORMULA_MASKS);
        std::vector<uint16_t> base(
            size_t(P10DC_RANKFORMULA_HEIGHTS) * base_cols,
            P10DC_RANKFORMULA_BASE_INVALID);
        auto base_index = [&](uint32_t h, uint32_t col) -> size_t {
#if P10DC_RANKFORMULA_SPARSE_BASE
            return size_t(col) * P10DC_RANKFORMULA_HEIGHTS + h;
#else
            return size_t(h) * P10DC_RANKFORMULA_MASKS + col;
#endif
        };
        meta.reserve(low_prekey_count);
        size_t actual_codes = 0, nonempty_mask_rows = 0;
        uint32_t max_mask_base = 0;

        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            hoff[h] = uint32_t(meta.size());
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            if (h >= P10DC_RANKFORMULA_HEIGHTS && a != b) {
                std::cerr << "p10dc rankformula unexpected LOW height owner=" << fixed
                          << " h=" << h << " count=" << (b - a)
                          << " height_limit=" << P10DC_RANKFORMULA_HEIGHTS << '\n';
                std::exit(733);
            }
            uint32_t previous_mask = 0;
            bool have_mask = false;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t code = f.low_codes[i];
                const uint32_t mask = code_mask(code);
                if (have_mask && mask < previous_mask) {
                    std::cerr << "p10dc rankformula mask order failure owner=" << fixed
                              << " h=" << h << " prev=" << previous_mask
                              << " mask=" << mask << '\n';
                    std::exit(734);
                }
                if (!have_mask || mask != previous_mask) {
                    const uint32_t local_base = i - a;
                    if (local_base >= P10DC_RANKFORMULA_BASE_INVALID) {
                        std::cerr << "p10dc rankformula base16 overflow owner=" << fixed
                                  << " h=" << h << " mask=" << mask
                                  << " base=" << local_base << '\n';
                        std::exit(735);
                    }
                    const uint32_t base_col = P10DC_RANKFORMULA_SPARSE_BASE
                        ? uint32_t(mask_slot[mask]) : mask;
                    if (base_col >= base_cols) std::exit(736);
                    base[base_index(h, base_col)] = uint16_t(local_base);
                    max_mask_base = std::max(max_mask_base, local_base);
                    ++nonempty_mask_rows;
                    previous_mask = mask;
                    have_mask = true;
                }

#if P10DC_RANKFORMULA_RAWCODE
                if ((code & ~P10DC_RANKFORMULA_RAWCODE_MASK) != 0u) {
                    std::cerr << "p10dc rankformula raw-code overflow owner=" << fixed
                              << " h=" << h << " code=" << code
                              << " bits=" << P10DC_RANKFORMULA_RAWCODE_BITS << '\n';
                    std::exit(737);
                }
                meta.push_back(code);
#else
                const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                const uint32_t chunks = p10dc_rankchunk32_pack_host(key);
                if ((chunks >> 23) != 0u || p10dc_rankchunk32_unpack_host(chunks) != key) {
                    std::cerr << "p10dc rankformula chunk packing failure owner=" << fixed
                              << " h=" << h << " key=" << key
                              << " chunks=" << chunks << '\n';
                    std::exit(737);
                }
                meta.push_back(chunks);
#endif
                ++actual_codes;
            }
        }
        if (actual_codes != low_prekey_count || meta.size() != low_prekey_count) {
            std::cerr << "p10dc rankformula size mismatch owner=" << fixed
                      << " actual=" << actual_codes << '/' << low_prekey_count
                      << " meta=" << meta.size() << '\n';
            std::exit(738);
        }

        int min_base_delta = 32767, max_base_delta = -32768;
        size_t base_delta_rows = 0;
#if P10DC_RANKFORMULA_BASE_DELTA
        for (uint32_t col = 0; col < uint32_t(base_cols); ++col) {
            for (uint32_t h = 0; h < P10DC_RANKFORMULA_HEIGHTS; ++h) {
                const size_t ix = base_index(h, col);
                uint16_t packed_delta = 0u;
                if (h + 2u < P10DC_RANKFORMULA_HEIGHTS) {
                    const uint16_t a = base[ix];
                    const uint16_t b = base[base_index(h + 2u, col)];
                    if (a != P10DC_RANKFORMULA_BASE_INVALID &&
                        b != P10DC_RANKFORMULA_BASE_INVALID) {
                        const int delta = int(b) - int(a);
                        if (delta < -32768 || delta > 32767) {
                            std::cerr << "p10dc rankformula base-delta overflow owner=" << fixed
                                      << " h=" << h << " col=" << col
                                      << " delta=" << delta << '\n';
                            std::exit(739);
                        }
                        min_base_delta = std::min(min_base_delta, delta);
                        max_base_delta = std::max(max_base_delta, delta);
                        ++base_delta_rows;
                        packed_delta = uint16_t(int16_t(delta));
                    }
                }
                base[ix] = packed_delta;
            }
        }
#endif

        low_rankformula_meta32_count = meta.size();
        low_rankformula_base16_count = base.size();
        low_rankformula_owned_masks = owned_masks;
        low_rankformula_mask_slot16_count = P10DC_RANKFORMULA_SPARSE_BASE
            ? mask_slot.size() : 0u;
        if (low_rankformula_meta32_count > low_rankformula_meta32_capacity) {
            if (low_rankformula_meta32) cudaFree(low_rankformula_meta32);
            low_rankformula_meta32 = nullptr;
            low_rankformula_meta32_capacity = low_rankformula_meta32_count;
            if (low_rankformula_meta32_capacity)
                ck(cudaMalloc(&low_rankformula_meta32,
                              low_rankformula_meta32_capacity * sizeof(uint32_t)),
                   "p10dc rankformula meta alloc");
        }
        if (low_rankformula_base16_count > low_rankformula_base16_capacity) {
            if (low_rankformula_base16) cudaFree(low_rankformula_base16);
            low_rankformula_base16 = nullptr;
            low_rankformula_base16_capacity = low_rankformula_base16_count;
            if (low_rankformula_base16_capacity)
                ck(cudaMalloc(&low_rankformula_base16,
                              low_rankformula_base16_capacity * sizeof(uint16_t)),
                   "p10dc rankformula base alloc");
        }
        if (low_rankformula_mask_slot16_count > low_rankformula_mask_slot16_capacity) {
            if (low_rankformula_mask_slot16) cudaFree(low_rankformula_mask_slot16);
            low_rankformula_mask_slot16 = nullptr;
            low_rankformula_mask_slot16_capacity = low_rankformula_mask_slot16_count;
            if (low_rankformula_mask_slot16_capacity)
                ck(cudaMalloc(&low_rankformula_mask_slot16,
                              low_rankformula_mask_slot16_capacity * sizeof(uint16_t)),
                   "p10dc rankformula mask-slot alloc");
        }
        if (!meta.empty())
            ck(cudaMemcpy(low_rankformula_meta32, meta.data(),
                          meta.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "p10dc rankformula meta H2D");
        if (!base.empty())
            ck(cudaMemcpy(low_rankformula_base16, base.data(),
                          base.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
               "p10dc rankformula base H2D");
#if P10DC_RANKFORMULA_SPARSE_BASE
        ck(cudaMemcpy(low_rankformula_mask_slot16, mask_slot.data(),
                      mask_slot.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
           "p10dc rankformula mask-slot H2D");
#endif
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_META32,
                              &low_rankformula_meta32, sizeof(low_rankformula_meta32)),
           "p10dc rankformula meta ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_BASE16,
                              &low_rankformula_base16, sizeof(low_rankformula_base16)),
           "p10dc rankformula base ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_MASK_SLOT16,
                              &low_rankformula_mask_slot16, sizeof(low_rankformula_mask_slot16)),
           "p10dc rankformula mask-slot ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_HOFF,
                              hoff.data(), hoff.size() * sizeof(uint32_t)),
           "p10dc rankformula hoff");

        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr;
        low_prekey_capacity = 0;
        low_prekey_count = 0;
        uint32_t* null32 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &null32, sizeof(null32)),
           "p10dc rankformula null old prekey ptr");

        const size_t meta_bytes = meta.size() * sizeof(uint32_t);
        const size_t base_bytes = base.size() * sizeof(uint16_t);
        const size_t mask_slot_bytes = low_rankformula_mask_slot16_count * sizeof(uint16_t);
        std::cerr << "p10dc_low_rankformula fixed_owner=" << fixed
                  << " codes=" << actual_codes
                  << " meta_bytes=" << meta_bytes
                  << " base_entries=" << base.size()
                  << " base_bytes=" << base_bytes
                  << " base_heights=" << P10DC_RANKFORMULA_HEIGHTS
                  << " owned_masks=" << owned_masks
                  << " mask_slot_bytes=" << mask_slot_bytes
                  << " nonempty_mask_rows=" << nonempty_mask_rows
                  << " max_mask_base=" << max_mask_base
                  << " rankstream_bytes=0"
                  << " sparse_base=" << P10DC_RANKFORMULA_SPARSE_BASE
                  << " base_layout=" << (P10DC_RANKFORMULA_SPARSE_BASE ? "slot-major" : "height-major")
                  << " base_delta=" << P10DC_RANKFORMULA_BASE_DELTA
                  << " base_delta_rows=" << base_delta_rows
                  << " min_base_delta=" << (base_delta_rows ? min_base_delta : 0)
                  << " max_base_delta=" << (base_delta_rows ? max_base_delta : 0)
                  << " base_values_per_lookup=" << (P10DC_RANKFORMULA_BASE_DELTA ? 1 : 2)
                  << " meta_mode=" << (P10DC_RANKFORMULA_RAWCODE ? "raw2bit" : "ternary23")
                  << " meta_bits=" << (P10DC_RANKFORMULA_RAWCODE ? P10DC_RANKFORMULA_RAWCODE_BITS : 23u)
                  << " bytes_per_code_meta=4 old_prekey_freed=1\n";
    }

    void release() {
        if (low_rankformula_meta32) cudaFree(low_rankformula_meta32);
        if (low_rankformula_base16) cudaFree(low_rankformula_base16);
        if (low_rankformula_mask_slot16) cudaFree(low_rankformula_mask_slot16);
        low_rankformula_meta32 = nullptr;
        low_rankformula_base16 = nullptr;
        low_rankformula_mask_slot16 = nullptr;
        low_rankformula_meta32_count = low_rankformula_base16_count = 0;
        low_rankformula_mask_slot16_count = 0;
        low_rankformula_meta32_capacity = low_rankformula_base16_capacity = 0;
        low_rankformula_mask_slot16_capacity = 0;
        low_rankformula_owned_masks = 0;
        BucketFusedDirectHighRowsPrekeyTables::release();
    }
};
