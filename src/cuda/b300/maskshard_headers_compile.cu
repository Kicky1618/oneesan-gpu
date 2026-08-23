#include <cuda_runtime.h>

#define main oneesan_factorized_hbm_unused_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../gridfp/ramstream32_factorized_storage.hpp"
#include "../gridfp/ramstream32_highdesc.cuh"
#include "maskshard_layout.hpp"
#include "maskshard_lowlocal.cuh"
#include "maskshard_highio.cuh"

int main() {
    // Compile-time integration probe. Runtime behavior is validated separately
    // on actual P2P-capable GPUs and by host-side exhaustive probes.
    return 0;
}
