// v0.73 experiment: v0.70 puts every HIGH config copy and kernel on the
// worker's cudaStreamPerThread. At the row boundary, synchronize only that
// execution stream instead of the complete device. Graph capture/replay is
// unchanged from v0.72. Use wall_s for A/B.
#define MASKSHARD_HIGH_STREAM_END_SYNC 1
#include "oneesan_b300_maskshard_v072_highgraph.cu"
