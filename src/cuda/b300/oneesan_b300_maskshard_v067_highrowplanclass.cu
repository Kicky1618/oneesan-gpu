// v0.67 experiment: v0.66 host class caches plus compact HIGH row-depth plans
// indexed by popcount(low_mask). Peak-filtered LOW counts are unchanged by
// inserting forced-N positions, so each cap also has only LOW_LUT_K+1 plans.
#define MASKSHARD_HIGH_ROW_PLAN_CLASS_CACHE 1
#include "oneesan_b300_maskshard_v066_highgroupsizeclass.cu"
