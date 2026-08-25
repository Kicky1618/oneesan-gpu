// v0.51 experiment: keep v0.50's uint16/1024-replica backend but double
// the target work per CTA.  Exact n=27 modeling reduces LOW CTA count from
// about 168.41M/residue at 16,384 to about 86.00M/residue at 32,768 while
// requiring at most 487 replicas for the largest mask/stage group.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 32768
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
