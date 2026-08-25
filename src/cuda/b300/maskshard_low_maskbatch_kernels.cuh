#pragma once

#ifdef MASKSHARD_LOW_MASKBATCH_CTA_CONFIG_CACHE
#include "maskshard_low_maskbatch_kernels_cached.cuh"
#else
#include "maskshard_low_maskbatch_kernels_global.cuh"
#endif
