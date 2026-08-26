// v0.64 experiment: v0.63 row-local scalar reuse plus bytewise deduplication
// of persistent pinned HIGH FBlock layouts. For n=27 these layouts depend only
// on popcount(low_mask), so long sorted job runs reuse the same constant state.
#define MASKSHARD_HIGH_FBLOCK_LAYOUT_DEDUP 1
#include "oneesan_b300_maskshard_v063_highnblockonce.cu"
