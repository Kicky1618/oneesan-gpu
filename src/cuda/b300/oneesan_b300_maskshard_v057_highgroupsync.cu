// v0.57 experiment: preserve v0.56's LOW compact-range CUDA Graph backend,
// but queue each complete HIGH gather/orbit/closure/scatter chain on the
// worker's default stream and wait only after the final scatter.
#define MASKSHARD_HIGH_GROUP_SYNC 1
#include "oneesan_b300_maskshard_v056_lowmaskbatch_runtime_tune.cu"
