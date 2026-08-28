#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void ck(cudaError_t st, const char* what) {
    if (st != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
}

void enable_peer(int device, int peer) {
    ck(cudaSetDevice(device), "event probe set peer device");
    int can = 0;
    ck(cudaDeviceCanAccessPeer(&can, device, peer), "event probe can access peer");
    if (!can) throw std::runtime_error("event probe peer access unavailable");
    cudaError_t st = cudaDeviceEnablePeerAccess(peer, 0);
    if (st == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    } else {
        ck(st, "event probe enable peer");
    }
}

__global__ void event_write_kernel(std::uint32_t* dst, std::uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = value;
}

__global__ void event_peer_read_kernel(
    const std::uint32_t* src,
    std::uint32_t* dst
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = *src;
}

void run_direction(int src_gpu, int dst_gpu, std::uint32_t marker) {
    enable_peer(dst_gpu, src_gpu);

    std::uint32_t* src = nullptr;
    std::uint32_t* dst = nullptr;
    cudaStream_t src_stream = nullptr;
    cudaStream_t dst_stream = nullptr;
    cudaEvent_t src_done = nullptr;

    ck(cudaSetDevice(src_gpu), "event probe src device");
    ck(cudaMalloc(&src, sizeof(std::uint32_t)), "event probe alloc src");
    ck(cudaMemset(src, 0, sizeof(std::uint32_t)), "event probe zero src");
    ck(cudaStreamCreateWithFlags(&src_stream, cudaStreamNonBlocking),
       "event probe create src stream");
    ck(cudaEventCreateWithFlags(&src_done, cudaEventDisableTiming),
       "event probe create event");

    ck(cudaSetDevice(dst_gpu), "event probe dst device");
    ck(cudaMalloc(&dst, sizeof(std::uint32_t)), "event probe alloc dst");
    ck(cudaMemset(dst, 0, sizeof(std::uint32_t)), "event probe zero dst");
    ck(cudaStreamCreateWithFlags(&dst_stream, cudaStreamNonBlocking),
       "event probe create dst stream");

    ck(cudaSetDevice(src_gpu), "event probe launch src device");
    event_write_kernel<<<1, 1, 0, src_stream>>>(src, marker);
    ck(cudaGetLastError(), "event probe write launch");
    ck(cudaEventRecord(src_done, src_stream), "event probe record src done");

    ck(cudaSetDevice(dst_gpu), "event probe wait dst device");
    ck(cudaStreamWaitEvent(dst_stream, src_done, 0),
       "event probe cross-device wait");
    event_peer_read_kernel<<<1, 1, 0, dst_stream>>>(src, dst);
    ck(cudaGetLastError(), "event probe peer read launch");
    ck(cudaStreamSynchronize(dst_stream), "event probe dst sync");

    std::uint32_t got = 0;
    ck(cudaMemcpy(&got, dst, sizeof(got), cudaMemcpyDeviceToHost),
       "event probe copy result");
    if (got != marker)
        throw std::runtime_error("cross-device event peer visibility mismatch");

    ck(cudaSetDevice(dst_gpu), "event probe free dst device");
    cudaStreamDestroy(dst_stream);
    cudaFree(dst);
    ck(cudaSetDevice(src_gpu), "event probe free src device");
    cudaEventDestroy(src_done);
    cudaStreamDestroy(src_stream);
    cudaFree(src);
}

} // namespace

int main(int argc, char** argv) {
    const int a = argc > 1 ? std::atoi(argv[1]) : 0;
    const int b = argc > 2 ? std::atoi(argv[2]) : 1;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "event probe device count");
    if (a < 0 || b < 0 || a == b || a >= visible || b >= visible) return 2;

    run_direction(a, b, 0x51a7c0deu);
    run_direction(b, a, 0xc001d00du);
    std::cout << "ALL_OK gridfp_cross_device_event=1"
              << " bidirectional=1"
              << " peer_visibility=1"
              << " cudaStreamWaitEvent_cross_device=1"
              << " gpu_a=" << a
              << " gpu_b=" << b << '\n';
    return 0;
}
