// v0.10 experiment: v0.9 plus packed HIGH closure work assignment.
// Reuses the v0.8 compact HIGH row list but flattens selected rows x fixed-LOW
// columns within each FBlock so small LOW-mask groups no longer waste most warp
// lanes. No additional persistent metadata is introduced.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define main oneesan_maskshard_v10_highrowpack_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.10 packed closure requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.10-highrowpack-fullclosure-batch"
        << " orbit_aux=1 block_orbit=1 compact_block_aux=1"
        << " high_closure_rows=1 high_closure_rowpack=1 low_closure_cols=1"
        << " guarded_hbm=1\n";
    return oneesan_maskshard_v10_highrowpack_inner_main(argc, argv);
}
