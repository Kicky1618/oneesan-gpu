// v0.62 experiment: v0.61 HIGH row batching with the persistent FBlock and
// compact-plan async sources allocated from contiguous pinned host memory.
#define MASKSHARD_HIGH_PINNED_CONFIG 1
#include "oneesan_b300_maskshard_v061_highrowbatchasync.cu"
