// v0.15 experiment: v0.14 plus exact per-factor maximum-frontier-height
// metadata. One byte per LOW/HIGH factor code is enough to decide whether a
// composed state is structurally reachable at the current row depth. The HIGH
// compute kernels remain unchanged; only gather/scatter P2P traffic is filtered.
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
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define main oneesan_maskshard_v15_rowdepthexact_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.15 exact row-depth I/O requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.15-highrowpack16-lazyblockinit-rowdepthexact-batch"
        << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
        << " high_closure_rows=1 high_closure_rowpack=1 high_closure_rowpack_threshold=16"
        << " low_closure_cols=1 skip_zero_block_gather=1 lazy_zero_block_init=1"
        << " row_depth_fblock_io=1 row_depth_exact_io=1 guarded_hbm=1\n";
    return oneesan_maskshard_v15_rowdepthexact_inner_main(argc, argv);
}
