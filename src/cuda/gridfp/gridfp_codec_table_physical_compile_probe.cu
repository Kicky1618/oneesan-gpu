#include <cuda_runtime.h>

#include "gridfp_reduced_production_device.cuh"

#include <cstdint>

using namespace oneesan::gridfp::reducedprod;

__global__ void codec_table_physical_compile_kernel(Rank64* out) {
    const int lane = int(threadIdx.x & 31);
    const int n = 28 - (lane & 7);
    const int k = n >> 1;
    const int rem = 28 - ((lane & 7) << 1);
    const int h = rem & 2;
    const Rank64 choose = RP_CHOOSE[n][k];
    const Rank64 primitive = RP_PRIMITIVE[rem][h];
    if (lane == 0) {
        out[0] = choose + primitive + Rank64(RP_CODEC_PHYSICAL_TABLE_BYTES);
    }
}

void codec_table_physical_compile_install() {
    Rank64 choose[RP_MAX_W + 1][RP_MAX_W + 1]{};
    Rank64 primitive[RP_MAX_W + 1][RP_MAX_W + 2]{};
#if RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE == 0
    (void)cudaMemcpyToSymbol(RP_CHOOSE, choose, sizeof(choose));
#endif
#if RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE == 0
    (void)cudaMemcpyToSymbol(RP_PRIMITIVE, primitive, sizeof(primitive));
#endif
}

int main() {
    codec_table_physical_compile_install();
    return RP_CODEC_PHYSICAL_TABLE_BYTES <= 0 ? 2 : 0;
}
