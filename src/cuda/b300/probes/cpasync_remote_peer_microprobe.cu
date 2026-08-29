#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

[[noreturn]] void fail(const char* what) {
    std::cerr << "cpasync_remote_peer_microprobe FAIL: " << what << '\n';
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(1);
    }
}

__host__ __device__ inline std::uint32_t pattern(int owner, std::uint32_t i) {
    return (0x9e3779b9u * (i + 1u)) ^ (0x85ebca6bu * std::uint32_t(owner + 1));
}

__device__ __forceinline__ void cp_async_u32(std::uint32_t* dst, const std::uint32_t* src) {
#if __CUDA_ARCH__ >= 800
    const std::uint32_t sdst = static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    const unsigned long long gsrc = reinterpret_cast<unsigned long long>(src);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sdst), "l"(gsrc));
#else
    *dst = *src;
#endif
}

__global__ void cpasync_remote_sum_kernel(
    const std::uint32_t* __restrict__ remote,
    std::uint32_t n,
    unsigned long long* __restrict__ out
) {
    extern __shared__ std::uint32_t scratch[];
    const std::uint32_t tid = std::uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint32_t* mine = scratch + std::uint32_t(threadIdx.x) * 14u;

#pragma unroll
    for (std::uint32_t j = 0; j < 14u; ++j) {
        const std::uint32_t ix = (tid * 37u + j * 521u + (tid >> 3) * 17u) % n;
        cp_async_u32(mine + j, remote + ix);
        if (j == 7u) {
#if __CUDA_ARCH__ >= 800
            asm volatile("cp.async.commit_group;");
#endif
        }
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
#endif

    unsigned long long sum = 0;
#pragma unroll
    for (std::uint32_t j = 0; j < 14u; ++j) sum += mine[j];
    out[tid] = sum;
}

__global__ void direct_remote_sum_kernel(
    const std::uint32_t* __restrict__ remote,
    std::uint32_t n,
    unsigned long long* __restrict__ out
) {
    const std::uint32_t tid = std::uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned long long sum = 0;
#pragma unroll
    for (std::uint32_t j = 0; j < 14u; ++j) {
        const std::uint32_t ix = (tid * 37u + j * 521u + (tid >> 3) * 17u) % n;
        sum += remote[ix];
    }
    out[tid] = sum;
}

void enable_peer(int src, int dst) {
    ck(cudaSetDevice(src), "set peer source");
    int can = 0;
    ck(cudaDeviceCanAccessPeer(&can, src, dst), "cudaDeviceCanAccessPeer");
    if (!can) fail("required peer access unavailable");
    const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        ck(e, "cudaDeviceEnablePeerAccess");
    }
}

} // namespace

int main(int argc, char** argv) {
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    const std::uint32_t n = argc > 2 ? std::uint32_t(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int blocks = argc > 4 ? std::atoi(argv[4]) : 64;
    if (ngpu < 2 || ngpu > 8 || !n || threads <= 0 || threads > 1024 || blocks <= 0)
        return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < ngpu) fail("not enough visible GPUs");

    std::vector<std::uint32_t*> state(std::size_t(ngpu), nullptr);
    std::vector<unsigned long long*> cpout(std::size_t(ngpu), nullptr);
    std::vector<unsigned long long*> ldout(std::size_t(ngpu), nullptr);
    const std::uint32_t total_threads = std::uint32_t(threads * blocks);

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "alloc set device");
        ck(cudaMalloc(&state[std::size_t(g)], std::size_t(n) * sizeof(std::uint32_t)), "alloc state");
        ck(cudaMalloc(&cpout[std::size_t(g)], std::size_t(total_threads) * sizeof(unsigned long long)), "alloc cpout");
        ck(cudaMalloc(&ldout[std::size_t(g)], std::size_t(total_threads) * sizeof(unsigned long long)), "alloc ldout");
        std::vector<std::uint32_t> h(n);
        for (std::uint32_t i = 0; i < n; ++i) h[i] = pattern(g, i);
        ck(cudaMemcpy(state[std::size_t(g)], h.data(), h.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "init state");
    }

    for (int g = 0; g < ngpu; ++g) enable_peer(g, (g + 1) % ngpu);

    const std::size_t smem = std::size_t(threads) * 14u * sizeof(std::uint32_t);
    for (int g = 0; g < ngpu; ++g) {
        const int peer = (g + 1) % ngpu;
        ck(cudaSetDevice(g), "launch set device");
        cpasync_remote_sum_kernel<<<blocks, threads, smem>>>(
            state[std::size_t(peer)], n, cpout[std::size_t(g)]);
        ck(cudaGetLastError(), "cp.async remote launch");
        direct_remote_sum_kernel<<<blocks, threads>>>(
            state[std::size_t(peer)], n, ldout[std::size_t(g)]);
        ck(cudaGetLastError(), "direct remote launch");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "sync set device");
        ck(cudaDeviceSynchronize(), "probe sync");
    }

    std::vector<unsigned long long> hc(total_threads), hd(total_threads);
    for (int g = 0; g < ngpu; ++g) {
        const int peer = (g + 1) % ngpu;
        ck(cudaSetDevice(g), "verify set device");
        ck(cudaMemcpy(hc.data(), cpout[std::size_t(g)], hc.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy cpout");
        ck(cudaMemcpy(hd.data(), ldout[std::size_t(g)], hd.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy ldout");
        for (std::uint32_t tid = 0; tid < total_threads; ++tid) {
            unsigned long long expect = 0;
            for (std::uint32_t j = 0; j < 14u; ++j) {
                const std::uint32_t ix = (tid * 37u + j * 521u + (tid >> 3) * 17u) % n;
                expect += pattern(peer, ix);
            }
            if (hc[tid] != expect || hd[tid] != expect || hc[tid] != hd[tid]) {
                std::cerr << "mismatch gpu=" << g << " peer=" << peer << " tid=" << tid
                          << " cpasync=" << hc[tid] << " direct=" << hd[tid]
                          << " expect=" << expect << '\n';
                return 3;
            }
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "free set device");
        cudaFree(ldout[std::size_t(g)]);
        cudaFree(cpout[std::size_t(g)]);
        cudaFree(state[std::size_t(g)]);
    }

    std::cout << "cpasync_remote_peer_microprobe"
              << " ngpu=" << ngpu
              << " values_per_gpu=" << n
              << " threads_per_block=" << threads
              << " blocks_per_gpu=" << blocks
              << " async_values_per_thread=14"
              << " shared_bytes_per_block=" << smem
              << " peer_ring=1 direct_reference=OK cp_async_remote_peer=OK exact=OK\n";
    return 0;
}
