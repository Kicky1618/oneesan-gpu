#define main gridfp_rankformula_nometa_blocks_main_unused
#include "gridfp_rankformula_nometa_blocks.cpp"
#undef main

static void prove_prefix_active(int B) {
    for (uint32_t stripe = 0; stripe < 128; stripe += 32) {
        for (uint32_t active = 1; active <= 32; ++active) {
            for (uint32_t lane = 0; lane < active; ++lane) {
                const uint32_t src = lane & ~uint32_t(B - 1);
                if (src >= active) std::exit(10);
                const uint32_t rank = stripe + lane;
                const uint32_t srank = stripe + src;
                if (rank / uint32_t(B) != srank / uint32_t(B)) std::exit(11);
            }
        }
    }
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const auto b4 = measure(f, owner, 4);
    const auto b8 = measure(f, owner, 8);
    const auto b16 = measure(f, owner, 16);
    prove_prefix_active(4);
    prove_prefix_active(8);
    prove_prefix_active(16);

    const uint64_t b4_scalar = 2ull * b4.codes + b4.locator_steps;
    const uint64_t b8_scalar = 2ull * b8.codes + b8.locator_steps;
    const uint64_t b16_scalar = 2ull * b16.codes + b16.locator_steps;
    const uint64_t b4_shared = 2ull * b4.blocks + b4.locator_steps;
    const uint64_t b8_shared = 2ull * b8.blocks + b8.locator_steps;
    const uint64_t b16_shared = 2ull * b16.blocks + b16.locator_steps;

    if (b4.locator_steps != 104346ull || b8.locator_steps != 243417ull ||
        b16.locator_steps != 521034ull ||
        b4_scalar != 2508180ull || b8_scalar != 2647251ull ||
        b16_scalar != 2924868ull ||
        b4_shared != 705394ull || b8_shared != 544003ull ||
        b16_shared != 671384ull ||
        !(b8_shared < b4_shared && b8_shared < b16_shared)) return 12;

    auto emit = [](int B, const MeasureResult& z, uint64_t scalar, uint64_t shared) {
        std::cout << "block=" << B
                  << " codes=" << z.codes
                  << " blocks=" << z.blocks
                  << " locator_steps=" << z.locator_steps
                  << " scalar_table_loads=" << scalar
                  << " warpshare_table_loads=" << shared
                  << " warpshare_loads_per_code=" << double(shared) / double(z.codes)
                  << " load_reduction_fraction=" << 1.0 - double(shared) / double(scalar)
                  << '\n';
    };
    emit(4, b4, b4_scalar, b4_shared);
    emit(8, b8, b8_scalar, b8_shared);
    emit(16, b16, b16_scalar, b16_shared);
    std::cout << "gridfp-rankformula-nometa-warpshare OK"
              << " prefix_active_widths=32"
              << " stripe_bases=4"
              << " blocks=4,8,16"
              << " best_block=8"
              << " block8_shared_loads=544003"
              << " block8_loads_per_code=" << double(b8_shared) / double(b8.codes)
              << " representative_lane_active=1 same_block=1\n";
    return 0;
}
