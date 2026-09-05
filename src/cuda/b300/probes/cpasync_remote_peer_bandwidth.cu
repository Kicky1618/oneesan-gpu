#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr std::uint32_t kLoads = 14u;

[[noreturn]] void fail(const char* what) {
    std::cerr << "cpasync_remote_peer_bandwidth FAIL: " << what << '\n';
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

__device__ __forceinline__ std::uint32_t random_index(
    std::uint32_t tid, std::uint32_t round, std::uint32_t j, std::uint32_t mask
) {
    unsigned long long x =
        (static_cast<unsigned long long>(tid) + 1ull) * 0x9e3779b97f4a7c15ull;
    x ^= (static_cast<unsigned long long>(round) + 1ull) * 0xd1b54a32d192ed03ull;
    x ^= (static_cast<unsigned long long>(j) + 1ull) * 0x94d049bb133111ebull;
    x ^= x >> 29;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 32;
    return std::uint32_t(x) & mask;
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

__global__ void direct_remote_bandwidth_kernel(
    const std::uint32_t* __restrict__ remote,
    std::uint32_t mask,
    std::uint32_t rounds,
    unsigned long long* __restrict__ out
) {
    const std::uint32_t tid = std::uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned long long sum = 0;
    for (std::uint32_t r = 0; r < rounds; ++r) {
        std::uint32_t idx[kLoads];
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j)
            idx[j] = random_index(tid, r, j, mask);
        std::uint32_t v[kLoads];
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j)
            v[j] = __ldg(remote + idx[j]);
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j)
            sum += v[j];
    }
    out[tid] = sum;
}

__global__ void cpasync_remote_bandwidth_kernel(
    const std::uint32_t* __restrict__ remote,
    std::uint32_t mask,
    std::uint32_t rounds,
    unsigned long long* __restrict__ out
) {
    extern __shared__ std::uint32_t scratch[];
    const std::uint32_t tid = std::uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint32_t* mine = scratch + std::uint32_t(threadIdx.x) * kLoads;
    unsigned long long sum = 0;

    for (std::uint32_t r = 0; r < rounds; ++r) {
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j) {
            const std::uint32_t ix = random_index(tid, r, j, mask);
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
#pragma unroll
        for (std::uint32_t j = 0; j < kLoads; ++j)
            sum += mine[j];
    }
    out[tid] = sum;
}

void enable_peer(int src, int dst) {
    ck(cudaSetDevice(src), "set peer source");
    int can = 0;
    ck(cudaDeviceCanAccessPeer(&can, src, dst), "cudaDeviceCanAccessPeer");
    if (!can) fail("required peer access unavailable");
    const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    else ck(e, "cudaDeviceEnablePeerAccess");
}

struct DeviceRun {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

enum class Mode { Direct, CpAsync };

std::vector<float> run_mode(
    Mode mode,
    const std::vector<std::uint32_t*>& state,
    const std::vector<unsigned long long*>& out,
    std::vector<DeviceRun>& run,
    int ngpu,
    std::uint32_t mask,
    int threads,
    int blocks,
    std::uint32_t rounds,
    int launches,
    int warmup
) {
    const std::size_t smem =
        mode == Mode::CpAsync ? std::size_t(threads) * kLoads * sizeof(std::uint32_t) : 0u;

    for (int w = 0; w < warmup; ++w) {
        for (int g = 0; g < ngpu; ++g) {
            const int peer = (g + 1) % ngpu;
            ck(cudaSetDevice(g), "warmup set device");
            if (mode == Mode::CpAsync)
                cpasync_remote_bandwidth_kernel<<<blocks, threads, smem, run[g].stream>>>(
                    state[peer], mask, rounds, out[g]);
            else
                direct_remote_bandwidth_kernel<<<blocks, threads, 0, run[g].stream>>>(
                    state[peer], mask, rounds, out[g]);
            ck(cudaGetLastError(), "warmup launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "warmup sync set device");
            ck(cudaStreamSynchronize(run[g].stream), "warmup stream sync");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "timed start set device");
        ck(cudaEventRecord(run[g].start, run[g].stream), "timed start event");
    }
    for (int launch = 0; launch < launches; ++launch) {
        for (int g = 0; g < ngpu; ++g) {
            const int peer = (g + 1) % ngpu;
            ck(cudaSetDevice(g), "timed launch set device");
            if (mode == Mode::CpAsync)
                cpasync_remote_bandwidth_kernel<<<blocks, threads, smem, run[g].stream>>>(
                    state[peer], mask, rounds, out[g]);
            else
                direct_remote_bandwidth_kernel<<<blocks, threads, 0, run[g].stream>>>(
                    state[peer], mask, rounds, out[g]);
            ck(cudaGetLastError(), "timed launch");
        }
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "timed stop set device");
        ck(cudaEventRecord(run[g].stop, run[g].stream), "timed stop event");
    }

    std::vector<float> ms(std::size_t(ngpu), 0.0f);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "timed sync set device");
        ck(cudaEventSynchronize(run[g].stop), "timed stop sync");
        ck(cudaEventElapsedTime(&ms[std::size_t(g)], run[g].start, run[g].stop),
           "timed elapsed");
    }
    return ms;
}

void report(
    const char* mode,
    const std::vector<float>& ms,
    std::uint64_t bytes_per_gpu,
    int ngpu
) {
    float max_ms = 0.0f;
    double avg_ms = 0.0;
    for (float x : ms) {
        max_ms = std::max(max_ms, x);
        avg_ms += x;
    }
    avg_ms /= double(ms.size());
    const double aggregate_gbs =
        max_ms > 0.0f ? double(bytes_per_gpu) * double(ngpu) / (double(max_ms) * 1.0e6) : 0.0;
    const double per_gpu_gbs =
        max_ms > 0.0f ? double(bytes_per_gpu) / (double(max_ms) * 1.0e6) : 0.0;
    std::cout << "mode=" << mode
              << " max_ms=" << std::fixed << std::setprecision(6) << max_ms
              << " avg_ms=" << avg_ms
              << " bytes_per_gpu=" << bytes_per_gpu
              << " per_gpu_gbs=" << per_gpu_gbs
              << " aggregate_gbs=" << aggregate_gbs
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    const std::uint32_t values = argc > 2
        ? std::uint32_t(std::strtoul(argv[2], nullptr, 10)) : (1u << 26);
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int blocks = argc > 4 ? std::atoi(argv[4]) : 256;
    const std::uint32_t rounds = argc > 5
        ? std::uint32_t(std::strtoul(argv[5], nullptr, 10)) : 32u;
    const int launches = argc > 6 ? std::atoi(argv[6]) : 8;
    const int warmup = argc > 7 ? std::atoi(argv[7]) : 2;

    if (ngpu < 2 || ngpu > 8 || values < 2u || (values & (values - 1u)) != 0u ||
        threads <= 0 || threads > 1024 || blocks <= 0 || !rounds || launches <= 0 || warmup < 0)
        return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < ngpu) fail("not enough visible GPUs");

    const std::uint32_t total_threads = std::uint32_t(threads * blocks);
    const std::uint32_t mask = values - 1u;
    const std::size_t out_count = std::size_t(total_threads);
    std::vector<std::uint32_t*> state(std::size_t(ngpu), nullptr);
    std::vector<unsigned long long*> direct_out(std::size_t(ngpu), nullptr);
    std::vector<unsigned long long*> async_out(std::size_t(ngpu), nullptr);
    std::vector<DeviceRun> run(std::size_t(ngpu));

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "alloc set device");
        ck(cudaMalloc(&state[g], std::size_t(values) * sizeof(std::uint32_t)), "state alloc");
        ck(cudaMalloc(&direct_out[g], out_count * sizeof(unsigned long long)), "direct out alloc");
        ck(cudaMalloc(&async_out[g], out_count * sizeof(unsigned long long)), "async out alloc");
        std::vector<std::uint32_t> host(values);
        for (std::uint32_t i = 0; i < values; ++i) host[i] = pattern(g, i);
        ck(cudaMemcpy(state[g], host.data(), host.size() * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "state init");
        ck(cudaStreamCreateWithFlags(&run[g].stream, cudaStreamNonBlocking), "stream create");
        ck(cudaEventCreate(&run[g].start), "start event create");
        ck(cudaEventCreate(&run[g].stop), "stop event create");
    }
    for (int g = 0; g < ngpu; ++g) enable_peer(g, (g + 1) % ngpu);

    const auto direct_ms = run_mode(
        Mode::Direct, state, direct_out, run, ngpu, mask,
        threads, blocks, rounds, launches, warmup);
    const auto async_ms = run_mode(
        Mode::CpAsync, state, async_out, run, ngpu, mask,
        threads, blocks, rounds, launches, warmup);

    std::vector<unsigned long long> hd(out_count), ha(out_count);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "verify set device");
        ck(cudaMemcpy(hd.data(), direct_out[g], out_count * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "direct verify copy");
        ck(cudaMemcpy(ha.data(), async_out[g], out_count * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "async verify copy");
        if (hd != ha) {
            std::cerr << "direct/cpasync mismatch gpu=" << g << '\n';
            return 3;
        }
    }

    const std::uint64_t bytes_per_gpu =
        std::uint64_t(total_threads) * std::uint64_t(rounds) *
        std::uint64_t(kLoads) * sizeof(std::uint32_t) * std::uint64_t(launches);
    report("direct", direct_ms, bytes_per_gpu, ngpu);
    report("cpasync", async_ms, bytes_per_gpu, ngpu);
    const float direct_max = *std::max_element(direct_ms.begin(), direct_ms.end());
    const float async_max = *std::max_element(async_ms.begin(), async_ms.end());
    std::cout << "cpasync_speedup=" << std::fixed << std::setprecision(6)
              << (async_max > 0.0f ? double(direct_max) / double(async_max) : 0.0)
              << "x exact_direct_vs_cpasync=1"
              << " peer_ring=1"
              << " values_per_gpu=" << values
              << " working_set_mib="
              << (double(values) * sizeof(std::uint32_t) / double(1u << 20))
              << " threads=" << threads
              << " blocks=" << blocks
              << " rounds=" << rounds
              << " launches=" << launches
              << " loads_per_thread_round=" << kLoads
              << '\n';

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "free set device");
        cudaEventDestroy(run[g].stop);
        cudaEventDestroy(run[g].start);
        cudaStreamDestroy(run[g].stream);
        cudaFree(async_out[g]);
        cudaFree(direct_out[g]);
        cudaFree(state[g]);
    }
    return 0;
}
