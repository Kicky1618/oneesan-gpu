// v0.46 experiment: v0.45 resident row plans plus one cooperative CTA-local
// stage of the hot per-mask config.  Each CTA pays one barrier, then thousands
// of warp tasks reuse shared FBlocks/prefix/count metadata instead of global
// resident-config loads.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
