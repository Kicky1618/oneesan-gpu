// v0.71 experiment: quotient the exact HIGH closure launch-count table by
// LOW-mask popcount. This also pins the class+row-cap geometry invariant needed
// for reusable HIGH CUDA Graphs.
#define MASKSHARD_HIGH_CLOSURE_LAUNCH_CLASS_CACHE 1
#include "oneesan_b300_maskshard_v070_highperthreadstream.cu"
