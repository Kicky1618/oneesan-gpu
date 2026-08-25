// v0.54 experiment: v0.53 compact replica ranges plus the v0.52 65,536
// warp-task target.  n=27 exact modeling keeps the compact range table at
// about 1.63 MiB on the worst GPU while reducing LOW CTA count to about
// 44.41M/residue; the largest group needs at most 244 replica CTAs.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 65536
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#define MASKSHARD_LOW_MASKBATCH_COMPACT_RANGES 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
