// v0.74 experiment: replace the per-row dynamic atomic HIGH-job ticket queue
// with one setup-time LPT assignment of the already work-sorted LOW-mask jobs.
// Each HIGH job stays pinned to one GPU for every row. This removes all per-job
// host atomic RMWs and may also reduce redundant per-GPU graph captures.
#define MASKSHARD_HIGH_STATIC_LPT_SCHEDULE 1
#include "oneesan_b300_maskshard_v073_highstreamsync.cu"
