#include <cuda_runtime.h>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_low_orbit_device.cuh"
#include "../ramstream32_b300_compact_io.cuh"
#include "../ramstream32_b300_rowkernels.cuh"
#include "../ramstream32_b300_high_warp.cuh"
#include "../ramstream32_b300_sparse_actions.cuh"

int main() { return 0; }
