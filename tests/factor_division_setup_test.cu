// Exercise the actual solver's host block construction without a CUDA device.
// Compile with TARGET_W=10, LOW_LUT_K=5, HIGH_LUT_K=4.
#define main oneesan_solver_main
#include "../src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
#undef main

int main() {
    static_assert(TARGET_W == 10 && LOW_LUT_K == 5 && HIGH_LUT_K == 4);
    build_full_dp();
    G_FACTOR = build_factor_tables();
    uint64_t checked = 0;
    for (bool low : {false, true}) {
        for (uint32_t mask = 0; mask < (1u << (low ? LOW_LUT_K : HIGH_LUT_K)); ++mask) {
            for (const auto& blocks : {make_factor_main_blocks(low, mask),
                                      make_factor_block_blocks(low, mask)}) {
                for (const auto& b : blocks) {
                    if (b.reciprocal != oneesan::division_reciprocal(b.stride)) return 1;
                    for (Code r = 0; r < b.end - b.off; ++r) {
                        if (!b.stride) return 2;
                        auto qr = oneesan::invariant_divmod(r, b.stride, b.reciprocal);
                        if (qr.quotient != r / b.stride || qr.remainder != r % b.stride) return 3;
                        ++checked;
                    }
                }
            }
        }
    }
    std::cout << "PASS " << checked << " factor positions, both partition directions\n";
}
