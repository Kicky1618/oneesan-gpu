// v0.77 experiment: preserve v0.76 exact cap-aware HIGH work weights while
// preferring an already-captured LOW popcount class only among GPUs tied at the
// same minimum LPT load. A cap adopts affinity only when graph classes decrease.
#define MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY 1
#include "oneesan_b300_maskshard_v076_highcaplpt_exactclosure.cu"
