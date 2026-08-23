// v0.8 experiment: v0.7 blocked-domain compact aux plus HIGH closure row
// traversal. HIGH transition kind depends only on HIGH row + center; a warp
// classifies one row and processes the passive LOW columns coalesced.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define main oneesan_maskshard_v08_highclosurerows_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    std::cerr << "backend_alias=b300-factorized-maskshard-v0.8-blockorbit-compactaux-highclosurerows-batch"
              << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
              << " high_closure_rows=1 guarded_hbm=1\n";
    return oneesan_maskshard_v08_highclosurerows_inner_main(argc, argv);
}
