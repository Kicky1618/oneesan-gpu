// Host-only inventory: no kernels, peer enabling, or large allocations.
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
  try {
    int count = 0, driver = 0, runtime = 0;
    check(cudaGetDeviceCount(&count));
    check(cudaDriverGetVersion(&driver));
    check(cudaRuntimeGetVersion(&runtime));
    std::cout << "{\"driver\":" << driver << ",\"runtime\":" << runtime << ",\"gpus\":[";
    for (int i = 0; i < count; ++i) {
      cudaDeviceProp p{};
      check(cudaGetDeviceProperties(&p, i));
      check(cudaSetDevice(i));
      size_t free = 0, total = 0;
      check(cudaMemGetInfo(&free, &total));
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = i;
      size_t gran = 0;
      auto vmm = cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      std::ostringstream uuid;
      uuid << "GPU-" << std::hex << std::setfill('0');
      for (int j = 0; j < 16; ++j) {
        if (j == 4 || j == 6 || j == 8 || j == 10) uuid << '-';
        uuid << std::setw(2) << unsigned(static_cast<unsigned char>(p.uuid.bytes[j]));
      }
      if (i) std::cout << ',';
      std::cout << "{\"index\":" << i << ",\"uuid\":" << std::quoted(uuid.str())
                << ",\"name\":" << std::quoted(p.name) << ",\"major\":" << p.major
                << ",\"minor\":" << p.minor << ",\"sms\":" << p.multiProcessorCount
                << ",\"free_bytes\":" << free << ",\"total_bytes\":" << total
                << ",\"vmm_gran_bytes\":" << (vmm == CUDA_SUCCESS ? gran : 0) << '}';
    }
    std::cout << "],\"p2p\":[";
    for (int i = 0; i < count; ++i) {
      if (i) std::cout << ',';
      std::cout << '[';
      for (int j = 0; j < count; ++j) {
        int can = i == j;
        if (i != j) check(cudaDeviceCanAccessPeer(&can, i, j));
        if (j) std::cout << ',';
        std::cout << can;
      }
      std::cout << ']';
    }
    std::cout << "]}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
