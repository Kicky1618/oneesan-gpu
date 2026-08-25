// v0.43 compile prototype.  The production row loop is not switched yet; this
// translation unit forces nvcc to instantiate the resident mask-batch config,
// descriptor planning, orbit/closure kernels, and host executor beside the
// cumulative v0.42 backend.
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
#define MASKSHARD_LOW_CLOSURE_PACKED_META 1
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
#define MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define main oneesan_maskshard_v43_proto_legacy_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

// The guarded wrapper undefines cudaMalloc after the legacy translation unit;
// restore the admission guard for resident v0.43 allocations too.
#define cudaMalloc maskshard_guarded_cuda_malloc
#include "maskshard_low_maskbatch_executor.cuh"
#undef cudaMalloc

int main(int argc, char** argv) {
    static_assert(sizeof(MaskShardLowBatchDesc) == 4,
                  "v0.43 host descriptor ABI changed");
    static_assert(sizeof(MaskShardLowBatchDeviceDesc) == 8,
                  "v0.43 device descriptor ABI changed");
    static_assert(sizeof(MaskShardLowGroupPackedBase) == 2320,
                  "v0.43 static base ABI changed");
    std::cerr
        << "backend_alias=b300-factorized-maskshard-v0.43-lowmaskbatch-proto"
        << " production_row_loop=0 compile_only_batch_path=1\n";
    return oneesan_maskshard_v43_proto_legacy_main(argc, argv);
}
