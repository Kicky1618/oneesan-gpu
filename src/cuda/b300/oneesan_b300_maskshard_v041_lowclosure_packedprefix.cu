// v0.41 experiment: v0.40 uint32 LOW closure tasks plus exact closure task
// prefixes precomputed once per LOW group.  The closure CTA no longer performs
// serial shared-prefix construction or a CTA-wide barrier before useful work.
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
#define MASKSHARD_LOW_CLOSURE_TASK_U32 1
#define MASKSHARD_LOW_CLOSURE_PACKED_PREFIX 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH 1
#define MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_WARP_DECODE 1
#define MASKSHARD_LOW_ORBIT_WARP_DECODE_FULLCAP 1
#define MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS 1
#define MASKSHARD_LOW_ORBIT_WARP_ROW_U32 1
#define MASKSHARD_LOW_GROUP_SYNC 1
#define MASKSHARD_LOW_GROUP_PACKED_CONFIG 1
#define MASKSHARD_LOW_GROUP_PACKED_CACHE 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define main oneesan_maskshard_v41_lowclosurepackedprefix_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.41 packed LOW closure prefix requires threads to be a multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.41-lowclosure-packedprefix-batch"
        << " low_closure_packed_prefix=1 low_closure_task_u32=1"
        << " low_group_packed_cache=1 low_orbit_warp_row_u32=1"
        << " low_group_sync=1 guarded_hbm=1\n";
    const int rc = oneesan_maskshard_v41_lowclosurepackedprefix_inner_main(argc, argv);
    const std::uint64_t skipped =
        G_MS_LOW_GROUP_SYNC_SKIPPED.load(std::memory_order_relaxed);
    const std::uint64_t executed =
        G_MS_LOW_GROUP_SYNC_EXECUTED.load(std::memory_order_relaxed);
    const std::uint64_t skips_per_group = 2u * std::uint64_t(LOW_LUT_K) - 1u;
    if (rc == 0 && skipped != executed * skips_per_group) {
        std::cerr << "v0.41 LOW group sync count mismatch skipped=" << skipped
                  << " executed=" << executed << '\n';
        return 4;
    }
    return rc;
}
