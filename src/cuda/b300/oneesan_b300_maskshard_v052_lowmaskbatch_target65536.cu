// v0.52 experiment: keep v0.50's uint16/1024-replica backend but raise
// target work per CTA to 65,536.  Exact n=27 modeling gives about
// 44.41M LOW CTAs/residue and needs at most 244 replicas per group.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 65536
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
