// v0.65 experiment: v0.64 transfer dedup plus a pinned HIGH FBlock cache
// indexed by popcount(low_mask). With all LOW positions fixed, only the number
// of occupied positions affects the per-height counts and FBlock geometry.
#define MASKSHARD_HIGH_FBLOCK_CLASS_CACHE 1
#include "oneesan_b300_maskshard_v064_highfblockdedup.cu"
