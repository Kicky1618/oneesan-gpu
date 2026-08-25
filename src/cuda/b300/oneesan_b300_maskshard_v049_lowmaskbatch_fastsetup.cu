// v0.49 experiment: v0.48 CTA-rebuilt metadata plus fast setup.  The exact
// mask/cap task tables remain authoritative for descriptor planning, while the
// now-unused legacy packed config is no longer rebuilt solely for a duplicate
// total-count comparison during device-table installation.
#define MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS 1
#define MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE 1
#define MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC 1
#define MASKSHARD_LOW_MASKBATCH_FAST_REBUILD_SETUP 1
#include "oneesan_b300_maskshard_v043_lowmaskbatch.cu"
