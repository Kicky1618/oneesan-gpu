#pragma once

#ifdef MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE
#ifdef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
#include "maskshard_low_maskbatch_kernels_rebuild.cuh"
#elif defined(MASKSHARD_LOW_MASKBATCH_COMPACT_DYNAMIC)
#include "maskshard_low_maskbatch_kernels_compact.cuh"
#else
#include "maskshard_low_maskbatch_kernels_cached.cuh"
#endif
#else
#include "maskshard_low_maskbatch_kernels_global.cuh"
#endif
