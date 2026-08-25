// v0.11 experiment: v0.10-style HIGH row packing only for source FBlocks
// whose fixed LOW-mask width is <16. Wider FBlocks retain the v0.8 one-row-
// per-warp mapping to limit row-list/HighDesc subgroup-load amplification.
// Reuses all v0.9 metadata and adds no persistent HBM tables.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 16
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define main oneesan_maskshard_v11_highrowpack16_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.11 hybrid packed closure requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.11-highrowpack16-fullclosure-batch"
        << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
        << " high_closure_rows=1 high_closure_rowpack=1 high_closure_rowpack_threshold=16"
        << " low_closure_cols=1 guarded_hbm=1\n";
    return oneesan_maskshard_v11_highrowpack16_inner_main(argc, argv);
}
