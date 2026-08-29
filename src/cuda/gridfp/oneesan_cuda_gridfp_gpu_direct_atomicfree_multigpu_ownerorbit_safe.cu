#include "ramstream32_gpu_direct_atomicfree_base.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_staged.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_peerstreams.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_orbitstage.cuh"

#define gdow_low_orbit_kernel gdow_low_orbit_kernel_v06_buggy
#define gdow_high_orbit_kernel gdow_high_orbit_kernel_v06_buggy
#define gdow_enqueue_row gdow_enqueue_row_v06_buggy
#include "ramstream32_gpu_direct_atomicfree_multigpu_ownerorbit.cuh"
#undef gdow_low_orbit_kernel
#undef gdow_high_orbit_kernel
#undef gdow_enqueue_row

#include "ramstream32_gpu_direct_atomicfree_multigpu_ownerorbit_safe.cuh"

#define main gdow_ownerorbit_safe_delegate_main
#include "oneesan_cuda_gridfp_gpu_direct_atomicfree_multigpu_ownerorbit.cu"
#undef main

int main(int argc,char**argv) {
    return gdow_ownerorbit_safe_delegate_main(argc,argv);
}
