// v0.50 experiment: v0.49 fast rebuild setup with a uint16 replica ABI and
// max 1024 replica CTAs per mask/stage.  At n=27 the exact task distribution
// needs at most 974 replicas to keep the configured 16,384 warp-task target.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
