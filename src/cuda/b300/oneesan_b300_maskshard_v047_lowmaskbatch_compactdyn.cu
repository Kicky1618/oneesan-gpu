// v0.47 experiment: v0.46 CTA config cache plus compact resident dynamic
// configs.  mask-independent closure begin/selected arrays are staged from the
// canonical closure tables once per CTA instead of duplicated for every
// mask/cap resident config.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_COMPACT_DYNAMIC 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
