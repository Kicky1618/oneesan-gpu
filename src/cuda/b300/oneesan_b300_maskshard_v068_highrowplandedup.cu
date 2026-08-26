// v0.68 experiment: v0.67 class-cached compact plans plus row-worker reuse
// of an already queued plan while consecutive HIGH jobs stay in one class.
#define MASKSHARD_HIGH_ROW_PLAN_COPY_DEDUP 1
#include "oneesan_b300_maskshard_v067_highrowplanclass.cu"
