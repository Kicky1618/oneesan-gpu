// v0.69 experiment: after v0.67 materializes every runtime HIGH compact plan,
// release the now-dead host-only cumulative active-count tables. Kernel state,
// device compact ranks, launch geometry and arithmetic are unchanged.
#define MASKSHARD_HIGH_RELEASE_COMPACT_COUNTS 1
#include "oneesan_b300_maskshard_v068_highrowplandedup.cu"
