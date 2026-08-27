// v0.79 experiment: preserve the v0.78 cap/locality schedule and move the
// per-job LOW-mask update into each captured HIGH graph. A resident uint16 mask
// sequence plus one row cursor reset replaces 2^LOW_LUT_K host symbol copies per
// row while keeping D_F_MASK in constant memory for all existing kernels.
#define MASKSHARD_HIGH_GRAPH_MASK_SEQUENCE 1
#include "oneesan_b300_maskshard_v078_highcaplpt_locality.cu"
