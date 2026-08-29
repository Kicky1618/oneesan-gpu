#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN

#include "ramstream32_cpu_low_sparse.hpp"
#include "ramstream32_cpu_high.hpp"
#include "ramstream32_cpu_high_direct.hpp"
#include "ramstream32_gpu_direct.cuh"
#include "ramstream32_gpu_direct_gather.cuh"
#include "ramstream32_gpu_direct_gather_cross.cuh"

// Main-free base for derived direct/gather executors.  Keep this header as the
// include boundary instead of recursively renaming `main` across .cu files;
// nested #undef main directives made that pattern fragile.
static bool gdg_has_arg(int argc, char** argv, const char* needle) {
    for (int i=1;i<argc;++i) if (std::strcmp(argv[i],needle)==0) return true;
    return false;
}

static double gdg_seconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
