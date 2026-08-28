#pragma once

#include "ramstream32_bucket_low_rankformula.cuh"

#ifndef P10DC_RANKFORMULA_SLOTROW32
#define P10DC_RANKFORMULA_SLOTROW32 0
#endif
static_assert(P10DC_RANKFORMULA_SLOTROW32 == 0 || P10DC_RANKFORMULA_SLOTROW32 == 1,
              "P10DC_RANKFORMULA_SLOTROW32 must be 0 or 1");
static_assert(!P10DC_RANKFORMULA_SLOTROW32 || P10DC_RANKFORMULA_SLOTMETA,
              "rankformula slotrow32 requires slotmeta");
static_assert(!P10DC_RANKFORMULA_SLOTROW32 || P10DC_RANKFORMULA_BASE_DELTA,
              "rankformula slotrow32 requires int16 base deltas");

__constant__ uint32_t* D_P10DC_LOW_RANKFORMULA_SLOTROW32;

__device__ __forceinline__ uint32_t p10dc_low_rankformula_slotrow32(
    uint32_t h, uint32_t slot
) {
    if (h + 2u >= P10DC_RANKFORMULA_HEIGHTS) return 0u;
    return D_P10DC_LOW_RANKFORMULA_SLOTROW32[
        size_t(slot) * P10DC_RANKFORMULA_HEIGHTS + h];
}

struct BucketFusedDirectHighRowsRankFormulaSlotRow32Tables
    : BucketFusedDirectHighRowsRankFormulaTables {
    uint32_t* low_rankformula_slotrow32 = nullptr;
    size_t low_rankformula_slotrow32_count = 0;
    size_t low_rankformula_slotrow32_capacity = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankFormulaTables::bind_owner(fixed, buckets, slot);
#if P10DC_RANKFORMULA_SLOTROW32
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc rankformula slotrow32 invalid owner=" << fixed << '\n';
            std::exit(750);
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
            for (uint32_t i = a; i < b; ++i)
                owned[code_mask(f.low_codes[i])] = 1u;
        }
        std::vector<uint16_t> mask_slot(P10DC_RANKFORMULA_MASKS,
                                        P10DC_RANKFORMULA_BASE_INVALID);
        std::vector<uint16_t> slot_mask;
        for (uint32_t mask = 0; mask < P10DC_RANKFORMULA_MASKS; ++mask) {
            if (!owned[mask]) continue;
            if (slot_mask.size() >= P10DC_RANKFORMULA_BASE_INVALID) std::exit(751);
            mask_slot[mask] = uint16_t(slot_mask.size());
            slot_mask.push_back(uint16_t(mask));
        }
        if (slot_mask.size() != low_rankformula_owned_masks) {
            std::cerr << "p10dc rankformula slotrow32 slot mismatch owner=" << fixed
                      << " rebuilt=" << slot_mask.size()
                      << " parent=" << low_rankformula_owned_masks << '\n';
            std::exit(752);
        }

        const size_t stride = P10DC_RANKFORMULA_HEIGHTS;
        std::vector<int32_t> absbase(slot_mask.size() * stride, -1);
        for (uint32_t h = 0; h < P10DC_RANKFORMULA_HEIGHTS; ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            uint32_t previous_mask = 0;
            bool have = false;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t mask = code_mask(f.low_codes[i]);
                if (!have || mask != previous_mask) {
                    const uint32_t s = uint32_t(mask_slot[mask]);
                    if (s >= slot_mask.size()) std::exit(753);
                    absbase[size_t(s) * stride + h] = int32_t(i - a);
                    previous_mask = mask;
                    have = true;
                }
            }
        }

        std::vector<uint32_t> row32(slot_mask.size() * stride, 0u);
        int min_delta = 32767, max_delta = -32768;
        size_t exact_rows = 0;
        for (uint32_t s = 0; s < uint32_t(slot_mask.size()); ++s) {
            const uint32_t support = uint32_t(slot_mask[s]);
            if (support >= P10DC_RANKFORMULA_MASKS) std::exit(754);
            for (uint32_t h = 0; h < P10DC_RANKFORMULA_HEIGHTS; ++h) {
                int delta = 0;
                const int32_t a = absbase[size_t(s) * stride + h];
                const int32_t b = h + 2u < P10DC_RANKFORMULA_HEIGHTS
                    ? absbase[size_t(s) * stride + h + 2u] : -1;
                if (a >= 0 && b >= 0) {
                    delta = int(b - a);
                    if (delta < -32768 || delta > 32767) std::exit(755);
                    min_delta = std::min(min_delta, delta);
                    max_delta = std::max(max_delta, delta);
                    ++exact_rows;
                }
                const uint32_t packed = uint32_t(uint16_t(int16_t(delta))) |
                                        (support << 16);
                if ((packed >> 30) != 0u) std::exit(756);
                row32[size_t(s) * stride + h] = packed;
            }
        }

        low_rankformula_slotrow32_count = row32.size();
        if (low_rankformula_slotrow32_count > low_rankformula_slotrow32_capacity) {
            if (low_rankformula_slotrow32) cudaFree(low_rankformula_slotrow32);
            low_rankformula_slotrow32 = nullptr;
            low_rankformula_slotrow32_capacity = low_rankformula_slotrow32_count;
            if (low_rankformula_slotrow32_capacity)
                ck(cudaMalloc(&low_rankformula_slotrow32,
                              low_rankformula_slotrow32_capacity * sizeof(uint32_t)),
                   "p10dc rankformula slotrow32 alloc");
        }
        if (!row32.empty())
            ck(cudaMemcpy(low_rankformula_slotrow32, row32.data(),
                          row32.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "p10dc rankformula slotrow32 H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_SLOTROW32,
                              &low_rankformula_slotrow32,
                              sizeof(low_rankformula_slotrow32)),
           "p10dc rankformula slotrow32 ptr");
        std::cerr << "p10dc_low_rankformula_slotrow32 fixed_owner=" << fixed
                  << " entries=" << row32.size()
                  << " bytes=" << row32.size() * sizeof(uint32_t)
                  << " support_bits=" << LOW_LUT_K
                  << " delta_bits=16 packed_bits=" << (LOW_LUT_K + 16)
                  << " loads_per_lookup=1"
                  << " exact_rows=" << exact_rows
                  << " min_base_delta=" << (exact_rows ? min_delta : 0)
                  << " max_base_delta=" << (exact_rows ? max_delta : 0)
                  << " separate_slot_support_hot=0\n";
#endif
    }

    void release() {
        if (low_rankformula_slotrow32) cudaFree(low_rankformula_slotrow32);
        low_rankformula_slotrow32 = nullptr;
        low_rankformula_slotrow32_count = 0;
        low_rankformula_slotrow32_capacity = 0;
        BucketFusedDirectHighRowsRankFormulaTables::release();
    }
};

#if P10DC_RANKFORMULA_SLOTROW32
using BucketFusedDirectHighRowsRankFormulaActiveTables =
    BucketFusedDirectHighRowsRankFormulaSlotRow32Tables;
#else
using BucketFusedDirectHighRowsRankFormulaActiveTables =
    BucketFusedDirectHighRowsRankFormulaTables;
#endif
