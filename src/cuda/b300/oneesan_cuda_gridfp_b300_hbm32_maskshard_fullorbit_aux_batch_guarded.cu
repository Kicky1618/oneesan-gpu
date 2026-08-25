// v0.5 production experiment. Keep the v0.4 implementation byte-for-byte as
// the comparison baseline, enable orbit aux at compile time, and wrap main so
// logs unambiguously identify this binary even though the inner v0.4 batch
// result line retains its historical backend= string.
#define MASKSHARD_ORBIT_AUX 1
#define main oneesan_maskshard_v05_orbitaux_inner_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

int main(int argc, char** argv) {
    std::cerr << "backend_alias=b300-factorized-maskshard-v0.5-orbitaux-batch"
              << " orbit_aux=1 guarded_hbm=1\n";
    return oneesan_maskshard_v05_orbitaux_inner_main(argc, argv);
}
