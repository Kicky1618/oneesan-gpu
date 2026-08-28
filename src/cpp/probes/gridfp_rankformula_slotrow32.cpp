#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

#include <limits>

int main() {
    Factors f = build_factors();
    const auto mask_owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;

    std::array<std::vector<uint16_t>, NG> slot_mask;
    std::vector<uint16_t> mask_slot(LM, RANK16_INVALID);
    for (int g = 0; g < NG; ++g) {
        for (uint32_t m = 0; m < LM; ++m) {
            if (mask_owner[m] != g) continue;
            if (slot_mask[g].size() >= RANK16_INVALID) return 2;
            mask_slot[m] = uint16_t(slot_mask[g].size());
            slot_mask[g].push_back(uint16_t(m));
        }
    }

    std::vector<int32_t> base(size_t(NG) * S * LM, -1);
    auto base_ref = [&](int g, int h, uint32_t m) -> int32_t& {
        return base[(size_t(g) * S + size_t(h)) * LM + m];
    };
    uint64_t codes = 0;
    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(mask_owner[m]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            if (!v.empty()) base_ref(g, h, m) = int32_t(next[g]);
            next[g] += uint32_t(v.size());
            codes += v.size();
        }
    }

    uint64_t rows = 0, exact = 0;
    int min_delta = std::numeric_limits<int>::max();
    int max_delta = std::numeric_limits<int>::min();
    uint32_t max_slot = 0, max_support = 0;
    for (int g = 0; g < NG; ++g) {
        for (uint32_t slot = 0; slot < slot_mask[g].size(); ++slot) {
            const uint32_t m = slot_mask[g][slot];
            if (uint32_t(mask_slot[m]) != slot) return 3;
            max_slot = std::max(max_slot, slot);
            max_support = std::max(max_support, m);
            for (int h = 0; h + 2 <= L + 1; ++h) {
                const int a = base_ref(g, h, m);
                const int b = base_ref(g, h + 2, m);
                if (a < 0 || b < 0) continue;
                const int delta = b - a;
                if (delta < std::numeric_limits<int16_t>::min() ||
                    delta > std::numeric_limits<int16_t>::max()) return 4;
                const uint32_t packed = uint32_t(uint16_t(int16_t(delta))) | (m << 16);
                const int got_delta = int(int16_t(packed & 0xffffu));
                const uint32_t got_support = (packed >> 16) & (LM - 1u);
                ++rows;
                if (got_delta != delta || got_support != m || (packed >> 30) != 0u) return 5;
                ++exact;
                min_delta = std::min(min_delta, delta);
                max_delta = std::max(max_delta, delta);
            }
        }
    }

    if (codes != 1201917ull || rows == 0 || exact != rows ||
        max_slot != 2049u || max_support >= LM) return 6;
    std::cout << "gridfp-rankformula-slotrow32 OK"
              << " codes=" << codes
              << " rows=" << rows
              << " exact=" << exact
              << " support_bits=" << L
              << " delta_bits=16"
              << " packed_bits=30"
              << " word_bytes=4"
              << " max_slot=" << max_slot
              << " max_support=" << max_support
              << " min_base_delta=" << min_delta
              << " max_base_delta=" << max_delta
              << " loads_per_lookup=1"
              << " separate_slot_support=0\n";
    return 0;
}
