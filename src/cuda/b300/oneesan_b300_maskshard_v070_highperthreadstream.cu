// v0.70 experiment: move the complete HIGH asynchronous sequence from the
// legacy null stream to cudaStreamPerThread. This is a semantics-preserving
// prerequisite for stream-capture/CUDA-Graph batching in v0.71.
#define MASKSHARD_HIGH_PERTHREAD_STREAM 1
#include "oneesan_b300_maskshard_v069_highreleasecounts.cu"
