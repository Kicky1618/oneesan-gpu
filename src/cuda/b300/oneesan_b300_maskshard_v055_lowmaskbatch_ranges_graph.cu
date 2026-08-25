// v0.55 experiment: v0.54 compact ranges + 65,536 warp-task CTAs, with
// one lazy CUDA Graph per device/cap.  The graph preserves the exact
// orbit->closure order for every LOW p while reducing steady-state host launch
// submissions from 28 kernel launches to one graph launch per row/device.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 65536
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#define MASKSHARD_LOW_MASKBATCH_COMPACT_RANGES 1
#define MASKSHARD_LOW_MASKBATCH_CUDA_GRAPH 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
