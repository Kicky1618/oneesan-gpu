// v0.53 experiment: same 16,384-task / uint16-replica schedule as v0.50,
// but store one prefix-range descriptor per active mask/stage instead of one
// descriptor per replica CTA.  blockIdx.x binary-searches cta_end to recover
// mask/local/replica/replicas on device.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 16384
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#define MASKSHARD_LOW_MASKBATCH_COMPACT_RANGES 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
