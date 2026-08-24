// v0.19 experiment: v0.18 plus exact compact BLOCKED task enumeration.
// Peak-sorted HIGH/LOW rank permutations let each row launch only states whose
// full frontier depth is structurally reachable, replacing v0.17's per-state
// peak check with two compact-rank lookups.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH 1
#define MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 16
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define main oneesan_maskshard_v19_exacttasks_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.19 exact task orbit requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.19-highrowpack16-rowdepthexact-exacttasks-batch"
        << " orbit_aux=1 block_orbit=1 compact_block_aux=1 block_orbit_tight_launch=1"
        << " block_orbit_row_cap_launch=1"
        << " high_closure_rows=1 high_closure_rowpack=1 high_closure_rowpack_threshold=16"
        << " low_closure_cols=1 skip_zero_block_gather=1 lazy_zero_block_init=1"
        << " row_depth_fblock_io=1 row_depth_exact_io=1 row_depth_orbit=1"
        << " row_depth_orbit_compact=1 guarded_hbm=1\n";
    return oneesan_maskshard_v19_exacttasks_inner_main(argc, argv);
}
