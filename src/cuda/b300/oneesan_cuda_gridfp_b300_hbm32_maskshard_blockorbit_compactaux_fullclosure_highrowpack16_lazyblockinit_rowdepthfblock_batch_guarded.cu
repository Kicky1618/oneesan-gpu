// v0.14 experiment: v0.13 plus a coarse structural-zero HIGH I/O filter.
// Every state in one FBlock shares HIGH ending height he and LOW starting height
// hs. If either exceeds the current row-depth cap, the state is unreachable and
// its authoritative transfer can be replaced by a local zero (gather) or omitted
// entirely (scatter). This adds no persistent HBM metadata.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 16
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define main oneesan_maskshard_v14_rowdepthfblock_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.14 row-depth FBlock I/O requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.14-highrowpack16-lazyblockinit-rowdepthfblock-batch"
        << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
        << " high_closure_rows=1 high_closure_rowpack=1 high_closure_rowpack_threshold=16"
        << " low_closure_cols=1 skip_zero_block_gather=1 lazy_zero_block_init=1"
        << " row_depth_fblock_io=1 guarded_hbm=1\n";
    return oneesan_maskshard_v14_rowdepthfblock_inner_main(argc, argv);
}
