#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t st, const char* what) {
    if (st != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
}

void enable_peer(int device, int peer) {
    if (device == peer) return;
    ck(cudaSetDevice(device), "fanin set peer device");
    int can = 0;
    ck(cudaDeviceCanAccessPeer(&can, device, peer), "fanin can access peer");
    if (!can) throw std::runtime_error("fanin peer access unavailable");
    cudaError_t st = cudaDeviceEnablePeerAccess(peer, 0);
    if (st == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    } else {
        ck(st, "fanin enable peer");
    }
}

__global__ void write_marker(std::uint32_t* dst, std::uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = value;
}

__global__ void sum_peer_markers(
    std::uint32_t* const* src,
    int n,
    std::uint32_t* dst
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint32_t sum = 0;
    for (int i = 0; i < n; ++i) sum += *src[i];
    *dst = sum;
}

__global__ void read_barrier_value(
    const std::uint32_t* barrier_value,
    std::uint32_t* dst
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = *barrier_value;
}

} // namespace

int main(int argc, char** argv) {
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "fanin device count");
    if (ngpu < 2 || ngpu > visible || ngpu > 16) return 2;

    // The production fan-in coordinator is GPU0.  It must read every peer for
    // this stronger visibility test, and every peer must read GPU0 after the
    // fan-out barrier.
    for (int g = 1; g < ngpu; ++g) {
        enable_peer(0, g);
        enable_peer(g, 0);
    }

    std::vector<std::uint32_t*> src(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::uint32_t*> dst(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<cudaStream_t> stream(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<cudaEvent_t> a_done(static_cast<std::size_t>(ngpu), nullptr);

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fanin alloc device");
        ck(cudaMalloc(&src[static_cast<std::size_t>(g)], sizeof(std::uint32_t)),
           "fanin alloc src");
        ck(cudaMalloc(&dst[static_cast<std::size_t>(g)], sizeof(std::uint32_t)),
           "fanin alloc dst");
        ck(cudaMemset(dst[static_cast<std::size_t>(g)], 0, sizeof(std::uint32_t)),
           "fanin zero dst");
        ck(cudaStreamCreateWithFlags(
               &stream[static_cast<std::size_t>(g)], cudaStreamNonBlocking),
           "fanin create data stream");
        ck(cudaEventCreateWithFlags(
               &a_done[static_cast<std::size_t>(g)], cudaEventDisableTiming),
           "fanin create A event");
    }

    cudaStream_t barrier_stream = nullptr;
    cudaEvent_t barrier_done = nullptr;
    std::uint32_t* barrier_value = nullptr;
    std::uint32_t** src_table = nullptr;
    ck(cudaSetDevice(0), "fanin coordinator setup device");
    ck(cudaStreamCreateWithFlags(&barrier_stream, cudaStreamNonBlocking),
       "fanin create barrier stream");
    ck(cudaEventCreateWithFlags(&barrier_done, cudaEventDisableTiming),
       "fanin create barrier event");
    ck(cudaMalloc(&barrier_value, sizeof(std::uint32_t)),
       "fanin alloc barrier value");
    ck(cudaMalloc(&src_table,
                  static_cast<std::size_t>(ngpu) * sizeof(std::uint32_t*)),
       "fanin alloc src table");
    ck(cudaMemcpy(src_table, src.data(),
                  static_cast<std::size_t>(ngpu) * sizeof(std::uint32_t*),
                  cudaMemcpyHostToDevice),
       "fanin copy src table");

    std::uint32_t expected = 0;
    for (int g = 0; g < ngpu; ++g) {
        const std::uint32_t marker = 0x1000u + std::uint32_t(17 * g + 3);
        expected += marker;
        ck(cudaSetDevice(g), "fanin launch A device");
        write_marker<<<1, 1, 0, stream[static_cast<std::size_t>(g)]>>>(
            src[static_cast<std::size_t>(g)], marker);
        ck(cudaGetLastError(), "fanin write launch");
        ck(cudaEventRecord(a_done[static_cast<std::size_t>(g)],
                           stream[static_cast<std::size_t>(g)]),
           "fanin record A event");
    }

    ck(cudaSetDevice(0), "fanin coordinator device");
    for (int g = 0; g < ngpu; ++g)
        ck(cudaStreamWaitEvent(barrier_stream,
                               a_done[static_cast<std::size_t>(g)], 0),
           "fanin wait A event");
    sum_peer_markers<<<1, 1, 0, barrier_stream>>>(src_table, ngpu, barrier_value);
    ck(cudaGetLastError(), "fanin sum launch");
    ck(cudaEventRecord(barrier_done, barrier_stream),
       "fanin record barrier event");

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fanin fanout device");
        ck(cudaStreamWaitEvent(stream[static_cast<std::size_t>(g)], barrier_done, 0),
           "fanin fanout wait");
        read_barrier_value<<<1, 1, 0, stream[static_cast<std::size_t>(g)]>>>(
            barrier_value, dst[static_cast<std::size_t>(g)]);
        ck(cudaGetLastError(), "fanin read barrier launch");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fanin sync device");
        ck(cudaStreamSynchronize(stream[static_cast<std::size_t>(g)]),
           "fanin sync stream");
        std::uint32_t got = 0;
        ck(cudaMemcpy(&got, dst[static_cast<std::size_t>(g)], sizeof(got),
                      cudaMemcpyDeviceToHost),
           "fanin copy result");
        if (got != expected)
            throw std::runtime_error("fanin/fanout visibility mismatch");
    }

    ck(cudaSetDevice(0), "fanin coordinator cleanup device");
    cudaFree(src_table);
    cudaFree(barrier_value);
    cudaEventDestroy(barrier_done);
    cudaStreamDestroy(barrier_stream);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fanin cleanup device");
        cudaEventDestroy(a_done[static_cast<std::size_t>(g)]);
        cudaStreamDestroy(stream[static_cast<std::size_t>(g)]);
        cudaFree(dst[static_cast<std::size_t>(g)]);
        cudaFree(src[static_cast<std::size_t>(g)]);
    }

    std::cout << "ALL_OK gridfp_cross_device_event_fanin=1"
              << " ngpu=" << ngpu
              << " coordinator_gpu=0"
              << " fanin_events=" << ngpu
              << " fanout_waits=" << ngpu
              << " peer_visibility=1"
              << " transitive_event_visibility=1\n";
    return 0;
}
