// v0.76 experiment: retain v0.75 cap-aware HIGH scheduling, but weight HIGH
// closure by the exact scheduled lane iterations. Unpacked one-row-per-warp
// blocks may loop over multiple 32-lane LOW chunks, so v0.75 warp_tasks*32
// undercounts their execution work when the active LOW width exceeds 32.
#define MASKSHARD_HIGH_CAP_LPT_EXACT_CLOSURE_LANES 1
#include "oneesan_b300_maskshard_v075_highcaplpt.cu"
