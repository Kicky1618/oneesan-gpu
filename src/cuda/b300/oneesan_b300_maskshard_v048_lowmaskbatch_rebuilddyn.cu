// v0.48 experiment: v0.45 resident row plans plus CTA-local reconstruction of
// all dynamic LOW batch metadata from canonical factor/closure tables.  No
// per-mask/per-cap dynamic config array is resident in HBM.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
