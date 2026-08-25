// v0.8 experiment: v0.7 blocked-domain compact aux plus HIGH closure row
// traversal. HIGH transition kind depends only on HIGH row + center; one warp
// processes one row and the passive LOW columns coalesced.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define main oneesan_maskshard_v08_highclosurerows_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.8 compact closure requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr << "backend_alias=b300-factorized-maskshard-v0.8-blockorbit-compactaux-highclosurerows-batch"
              << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
              << " high_closure_rows=1 guarded_hbm=1\n";
    return oneesan_maskshard_v08_highclosurerows_inner_main(argc, argv);
}
