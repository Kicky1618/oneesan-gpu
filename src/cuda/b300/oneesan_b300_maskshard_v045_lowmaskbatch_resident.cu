// v0.45 experiment: v0.43 executable LOW mask batching with every distinct
// row/cap descriptor plan built and uploaded once during setup.  The DP loop
// reuses resident descriptor arrays and performs no LOW descriptor H2D copies.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
