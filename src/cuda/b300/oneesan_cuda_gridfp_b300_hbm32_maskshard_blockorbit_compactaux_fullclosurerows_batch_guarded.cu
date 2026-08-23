// v0.9 experiment: v0.8 plus compact LOW closure columns. LOW closure source
// columns are stored in storage all-rank order; one warp processes up to 32
// selected columns within one HIGH row, avoiding a strided column traversal.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define main oneesan_maskshard_v09_fullclosure_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    std::cerr << "backend_alias=b300-factorized-maskshard-v0.9-blockorbit-compactaux-fullclosure-batch"
              << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
              << " high_closure_rows=1 low_closure_cols=1 guarded_hbm=1\n";
    return oneesan_maskshard_v09_fullclosure_inner_main(argc, argv);
}
