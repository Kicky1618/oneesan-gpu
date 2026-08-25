// v0.34 experiment: v0.33 plus LOW orbit/closure pair synchronization.
// The first device-wide wait in each LOW position is skipped; stream ordering
// still enforces orbit-before-closure and the closure-side wait remains.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH 1
#define MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 29
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH 1
#define MASKSHARD_HIGH_CLOSURE_TASK_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define MASKSHARD_LOW_CLOSURE_ROW_DEPTH 1
#define MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT 1
#define MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH 1
#define MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_WARP_DECODE 1
#define MASKSHARD_LOW_ORBIT_WARP_DECODE_FULLCAP 1
#define MASKSHARD_LOW_PAIR_SYNC 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define main oneesan_maskshard_v34_lowpairsync_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.34 LOW pair sync requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.34-low-pair-sync-batch"
        << " low_pair_sync=1 low_pair_sync_timing=closure_includes_orbit"
        << " low_orbit_warp_decode=1 low_orbit_warp_decode_fullcap=1"
        << " low_orbit_row_depth_compact_launch=1 guarded_hbm=1\n";
    const int rc = oneesan_maskshard_v34_lowpairsync_inner_main(argc, argv);
    const std::uint64_t skipped =
        G_MS_LOW_PAIR_SYNC_SKIPPED.load(std::memory_order_relaxed);
    const std::uint64_t executed =
        G_MS_LOW_PAIR_SYNC_EXECUTED.load(std::memory_order_relaxed);
    std::cerr << "low_pair_sync skipped=" << skipped
              << " executed=" << executed << '\n';
    if (rc == 0 && skipped != executed) {
        std::cerr << "v0.34 LOW pair sync count mismatch\n";
        return 4;
    }
    return rc;
}
