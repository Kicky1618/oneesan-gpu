// v0.7 experiment: retain the exact blocked-domain orbit of v0.6, but store
// the auxiliary orbit target in blocked-coordinate order. HighDesc/LowDesc
// block_desc already maps each blocked coordinate to its representative main
// state, so only NN-vs-pair plus the pair companion requires an extra word.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define main oneesan_maskshard_v07_compact_blockorbit_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    std::cerr << "backend_alias=b300-factorized-maskshard-v0.7-blockorbit-compactaux-batch"
              << " orbit_aux=1 block_orbit=1 compact_block_aux=1 guarded_hbm=1\n";
    return oneesan_maskshard_v07_compact_blockorbit_inner_main(argc, argv);
}
