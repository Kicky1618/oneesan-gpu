// v0.6 experiment: use the v0.5 descriptor+aux target tables, but iterate the
// blocked-state side of each in-place orbit. For fixed p, inserting N maps each
// blocked state bijectively to one NN/NR/NL main representative, so the orbit
// kernel scans ~35% as many states as the main-domain v0.5 kernel.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define main oneesan_maskshard_v06_blockorbit_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    std::cerr << "backend_alias=b300-factorized-maskshard-v0.6-blockorbit-aux-batch"
              << " orbit_aux=1 block_orbit=1 guarded_hbm=1\n";
    return oneesan_maskshard_v06_blockorbit_inner_main(argc, argv);
}
