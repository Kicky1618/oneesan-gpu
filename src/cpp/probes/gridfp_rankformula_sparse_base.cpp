#define main gridfp_low_rank16_plan_main_unused
#include "gridfp_low_rank16_plan.cpp"
#undef main

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    std::array<uint32_t, NG> masks{};
    for (uint8_t g : owner) ++masks[g];
    const uint32_t max_masks = *std::max_element(masks.begin(), masks.end());
    const uint32_t min_masks = *std::min_element(masks.begin(), masks.end());
    const uint64_t dense_bytes = uint64_t(L + 2) * (1u << L) * sizeof(uint16_t);
    const uint64_t slot_bytes = uint64_t(1u << L) * sizeof(uint16_t);
    const uint64_t max_sparse_base_bytes = uint64_t(max_masks) * (L + 2) * sizeof(uint16_t);
    const uint64_t max_sparse_bytes = slot_bytes + max_sparse_base_bytes;
    if (max_masks >= 0xffffu || max_sparse_bytes >= dense_bytes) return 2;
    std::cout << "gridfp-rankformula-sparse-base OK"
              << " W=" << W << " low_k=" << L
              << " owner_masks=";
    for (int g = 0; g < NG; ++g) {
        if (g) std::cout << ',';
        std::cout << masks[g];
    }
    std::cout << " min_owner_masks=" << min_masks
              << " max_owner_masks=" << max_masks
              << " mask_slot_bytes=" << slot_bytes
              << " max_sparse_base_bytes=" << max_sparse_base_bytes
              << " max_sparse_total_bytes=" << max_sparse_bytes
              << " dense_base_bytes=" << dense_bytes
              << " reduction=" << double(dense_bytes) / double(max_sparse_bytes)
              << "x\n";
    return 0;
}
