// v0.78 experiment: retain v0.77 capture-affinity scheduling, but evaluate an
// equal-load affinity+locality candidate against exact row-depth authoritative
// peer I/O. Adopt only Pareto-safe caps: graph classes and peer I/O must both be
// no worse than v0.77, with at least one strict improvement.
#define MASKSHARD_HIGH_CAP_LPT_LOCALITY_GUARD 1
#include "oneesan_b300_maskshard_v077_highcaplpt_captureaffinity.cu"
