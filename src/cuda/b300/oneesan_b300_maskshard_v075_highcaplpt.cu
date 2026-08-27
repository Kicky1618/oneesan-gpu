// v0.75 experiment: rebuild the v0.74 static HIGH LPT assignment for each
// row-depth cap using host-only exact orbit/closure task counts. The shared host
// still sees one jobs_by_gpu interface; a tiny proxy selects the prepared cap
// schedule from the row-depth hook. No extra GPU metadata or symbol copies.
#define MASKSHARD_HIGH_CAP_LPT_SCHEDULE 1
#include "oneesan_b300_maskshard_v074_highstaticlpt.cu"
