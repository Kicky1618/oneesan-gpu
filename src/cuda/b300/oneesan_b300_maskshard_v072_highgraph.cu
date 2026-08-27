// v0.72 experiment: lazily capture the complete HIGH kernel sequence into one
// reusable CUDA Graph per (LOW-mask popcount, row-depth cap) class and GPU.
// Mask-specific constant updates remain graph-external on the same per-thread
// stream. Phase timers are intentionally not comparable; use wall_s for A/B.
#define MASKSHARD_HIGH_CUDA_GRAPH 1
#include "oneesan_b300_maskshard_v071_highclosureclass.cu"
