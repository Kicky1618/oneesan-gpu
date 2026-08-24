// v0.35 experiment: v0.33 compute path plus one device wait per complete LOW
// mask group. All 14 orbit/closure pairs remain ordered on the same CUDA stream;
// only synchronization frequency changes.
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
#define MASKSHARD_LOW_GROUP_SYNC 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define main oneesan_maskshard_v35_lowgroupsync_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.35 LOW group sync requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.35-low-group-sync-batch"
        << " low_group_sync=1 low_group_sync_timing=final_closure_includes_group"
        << " low_orbit_warp_decode=1 low_orbit_warp_decode_fullcap=1"
        << " low_orbit_row_depth_compact_launch=1 guarded_hbm=1\n";
    const int rc = oneesan_maskshard_v35_lowgroupsync_inner_main(argc, argv);
    const std::uint64_t skipped =
        G_MS_LOW_GROUP_SYNC_SKIPPED.load(std::memory_order_relaxed);
    const std::uint64_t executed =
        G_MS_LOW_GROUP_SYNC_EXECUTED.load(std::memory_order_relaxed);
    std::cerr << "low_group_sync skipped=" << skipped
              << " executed=" << executed << '\n';
    const std::uint64_t skips_per_group = 2u * std::uint64_t(LOW_LUT_K) - 1u;
    if (rc == 0 && skipped != executed * skips_per_group) {
        std::cerr << "v0.35 LOW group sync count mismatch\n";
        return 4;
    }
    if constexpr (TARGET_W == 28 && LOW_LUT_K == 14 && HIGH_LUT_K == 13) {
        const std::uint64_t residues = argc > 4 ? std::uint64_t(argc - 4) : 1u;
        const std::uint64_t expected_groups = 229376ULL * residues;
        if (rc == 0 && executed != expected_groups) {
            std::cerr << "v0.35 n=27 LOW group count mismatch got=" << executed
                      << " expected=" << expected_groups << '\n';
            return 5;
        }
    }
    return rc;
}
