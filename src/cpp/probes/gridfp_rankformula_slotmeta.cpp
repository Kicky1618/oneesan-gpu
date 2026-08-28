#define main gridfp_low_rank16_plan_main_unused
#include "gridfp_low_rank16_plan.cpp"
#undef main

#include <limits>

static uint32_t lmask_of(uint32_t code) {
    uint32_t m = 0;
    for (int p = 0; p < L; ++p)
        if (((code >> (2 * p)) & 3u) == LL) m |= 1u << p;
    return m;
}

static uint32_t code_from_masks(uint32_t support, uint32_t lmask) {
    uint32_t code = 0;
    for (int p = 0; p < L; ++p) {
        if (((support >> p) & 1u) == 0u) continue;
        code |= (((lmask >> p) & 1u) ? uint32_t(LL) : uint32_t(R)) << (2 * p);
    }
    return code;
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;

    std::array<std::vector<uint16_t>, NG> slot_mask;
    std::vector<uint16_t> mask_slot(LM, RANK16_INVALID);
    std::array<uint32_t, NG> owned{};
    for (int g = 0; g < NG; ++g) {
        for (uint32_t m = 0; m < LM; ++m) {
            if (owner[m] != g) continue;
            if (owned[g] >= RANK16_INVALID) return 2;
            if (g == owner[m]) mask_slot[m] = uint16_t(owned[g]);
            slot_mask[g].push_back(uint16_t(m));
            ++owned[g];
        }
    }

    uint64_t codes = 0, exact = 0;
    uint32_t max_slot = 0, max_packed = 0;
    for (int h = 0; h <= L + 1; ++h) {
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(owner[m]);
            const uint32_t slot = uint32_t(mask_slot[m]);
            if (slot >= slot_mask[g].size() || slot_mask[g][slot] != m) return 3;
            for (uint32_t code : f.low_mask_h[size_t(m) * S + size_t(h)]) {
                const uint32_t lm = lmask_of(code);
                if (lm & ~m) return 4;
                const uint32_t packed = lm | (slot << L);
                const uint32_t got_lm = packed & (LM - 1u);
                const uint32_t got_slot = packed >> L;
                const uint32_t got_support = uint32_t(slot_mask[g][got_slot]);
                const uint32_t got_code = code_from_masks(got_support, got_lm);
                ++codes;
                if (got_code != code || got_support != m || got_lm != lm) {
                    std::cerr << "rankformula slotmeta mismatch owner=" << g
                              << " h=" << h << " mask=" << m
                              << " slot=" << slot << " code=" << code
                              << " got=" << got_code << '\n';
                    return 5;
                }
                ++exact;
                max_slot = std::max(max_slot, slot);
                max_packed = std::max(max_packed, packed);
            }
        }
    }

    uint32_t max_owned = 0;
    for (uint32_t n : owned) max_owned = std::max(max_owned, n);
    uint32_t slot_bits = 0, cap = 1;
    while (cap < max_owned) { ++slot_bits; cap <<= 1; }
    const uint32_t metadata_bits = uint32_t(L) + slot_bits;
    const uint32_t max_reverse_bytes = max_owned * sizeof(uint16_t);
    const uint32_t old_mask_slot_bytes = LM * sizeof(uint16_t);
    if (codes != 1201917ull || exact != codes || max_owned != 2050u ||
        max_slot != 2049u || metadata_bits != 26u) return 6;

    std::cout << "gridfp-rankformula-slotmeta OK"
              << " codes=" << codes
              << " exact=" << exact
              << " max_owned_masks=" << max_owned
              << " max_slot=" << max_slot
              << " slot_bits=" << slot_bits
              << " lmask_bits=" << L
              << " metadata_bits=" << metadata_bits
              << " metadata_bytes=4"
              << " max_reverse_slot_mask_bytes=" << max_reverse_bytes
              << " old_direct_mask_slot_bytes=" << old_mask_slot_bytes
              << " broadword_support=0"
              << " direct_mask_slot=0\n";
    return 0;
}
