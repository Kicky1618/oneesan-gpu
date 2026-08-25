// v0.12 experiment: v0.11 plus elimination of the row-boundary BLOCKED P2P
// gather. A complete previous row ends at p=1, whose output BLOCKED vector is
// identically zero. Initialize the local HIGH BLOCKED scratch with zero stores
// instead of rereading those zeros from authoritative sharded HBM over P2P.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 16
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define main oneesan_maskshard_v12_zeroblockgather_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.12 zero-block-gather requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.12-highrowpack16-zeroblockgather-batch"
        << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
        << " high_closure_rows=1 high_closure_rowpack=1 high_closure_rowpack_threshold=16"
        << " low_closure_cols=1 skip_zero_block_gather=1 guarded_hbm=1\n";
    return oneesan_maskshard_v12_zeroblockgather_inner_main(argc, argv);
}
