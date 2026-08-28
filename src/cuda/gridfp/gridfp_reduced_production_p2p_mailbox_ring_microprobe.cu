#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

static constexpr int MAX_GPU = 8;
static constexpr int TOKEN_WORDS = 1024;

struct alignas(16) Mailbox {
    unsigned int ready = 0;
    unsigned int pad[3]{};
    std::uint32_t payload[TOKEN_WORDS]{};
};

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(90);
    }
}

void enable_full_peer_mesh(int ngpu) {
    for (int src = 0; src < ngpu; ++src) {
        ck(cudaSetDevice(src), "mailbox set peer source");
        for (int dst = 0; dst < ngpu; ++dst) {
            if (src == dst) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, src, dst), "mailbox can access peer");
            if (!can) {
                std::cerr << "mailbox peer access unavailable src="
                          << src << " dst=" << dst << '\n';
                std::exit(91);
            }
            int native = 0;
            ck(cudaDeviceGetP2PAttribute(
                   &native, cudaDevP2PAttrNativeAtomicSupported, src, dst),
               "mailbox native atomic");
            if (!native) {
                std::cerr << "mailbox requires full native atomics src="
                          << src << " dst=" << dst << '\n';
                std::exit(92);
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "mailbox enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

__device__ __forceinline__ std::uint32_t initial_word(int i) {
    return 0x9e3779b9u * static_cast<std::uint32_t>(i + 1) + 0x1234567u;
}

__device__ __forceinline__ std::uint32_t add_before_gpu(int gpu) {
    // GPU j adds (j+1), so data entering gpu has additions 1..gpu.
    return static_cast<std::uint32_t>(gpu * (gpu + 1) / 2);
}

__global__ void mailbox_ring_kernel(
    Mailbox* inbox,
    Mailbox* next_inbox,
    int gpu,
    int ngpu,
    int* error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> in_ready(inbox->ready);
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> out_ready(next_inbox->ready);

    // The host seeds GPU0 with ready=1 before any kernel launch. Other GPUs
    // block here until the token arrives from their predecessor.
    while (in_ready.load(cuda::memory_order_acquire) != 1u) {}

    const std::uint32_t expected_add = add_before_gpu(gpu);
    for (int i = 0; i < TOKEN_WORDS; ++i) {
        const std::uint32_t expected = initial_word(i) + expected_add;
        if (inbox->payload[i] != expected) {
            atomicCAS(error, 0, 231);
            return;
        }
    }

    // Open this inbox for the next generation before forwarding. Only GPU0
    // receives a second token in this microprobe, after the entire ring has
    // progressed, so there is no ABA ambiguity here.
    in_ready.store(0u, cuda::memory_order_release);

    // Keep the payload publication and release-doorbell store in the same
    // thread. This deliberately sacrifices copy parallelism in the protocol
    // microprobe so the C++ release sequence directly orders every payload
    // write without relying on cross-thread publication details.
    const std::uint32_t own_add = static_cast<std::uint32_t>(gpu + 1);
    for (int i = 0; i < TOKEN_WORDS; ++i)
        next_inbox->payload[i] = inbox->payload[i] + own_add;
    out_ready.store(1u, cuda::memory_order_release);

    if (gpu != 0) return;

    // GPU0 is both ring origin and final consumer. Its inbox was reset above,
    // so ready=1 here can only be the token returned by GPU ngpu-1.
    while (in_ready.load(cuda::memory_order_acquire) != 1u) {}
    const std::uint32_t final_add =
        static_cast<std::uint32_t>(ngpu * (ngpu + 1) / 2);
    for (int i = 0; i < TOKEN_WORDS; ++i) {
        if (inbox->payload[i] != initial_word(i) + final_add) {
            atomicCAS(error, 0, 232);
            return;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "mailbox device count");
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    if (ngpu < 2 || ngpu > MAX_GPU) return 2;
    if (visible < ngpu) {
        std::cerr << "need " << ngpu << " CUDA devices, visible=" << visible << '\n';
        return 3;
    }

    enable_full_peer_mesh(ngpu);

    std::vector<Mailbox*> mailbox(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<int*> error(static_cast<std::size_t>(ngpu), nullptr);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "mailbox alloc set device");
        ck(cudaMalloc(&mailbox[static_cast<std::size_t>(g)], sizeof(Mailbox)),
           "mailbox alloc");
        ck(cudaMemset(mailbox[static_cast<std::size_t>(g)], 0, sizeof(Mailbox)),
           "mailbox zero");
        ck(cudaMalloc(&error[static_cast<std::size_t>(g)], sizeof(int)),
           "mailbox error alloc");
        ck(cudaMemset(error[static_cast<std::size_t>(g)], 0, sizeof(int)),
           "mailbox error zero");
    }

    Mailbox seed{};
    seed.ready = 1;
    for (int i = 0; i < TOKEN_WORDS; ++i)
        seed.payload[i] =
            0x9e3779b9u * static_cast<std::uint32_t>(i + 1) + 0x1234567u;
    ck(cudaSetDevice(0), "mailbox seed set device");
    ck(cudaMemcpy(mailbox[0], &seed, sizeof(seed), cudaMemcpyHostToDevice),
       "mailbox seed copy");

    // Launch all devices asynchronously. A token written before a successor
    // kernel starts remains in its mailbox, so launch order does not lose a
    // notification.
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "mailbox launch set device");
        mailbox_ring_kernel<<<1, 1>>>(
            mailbox[static_cast<std::size_t>(g)],
            mailbox[static_cast<std::size_t>((g + 1) % ngpu)],
            g, ngpu, error[static_cast<std::size_t>(g)]);
        ck(cudaGetLastError(), "mailbox launch");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "mailbox sync set device");
        ck(cudaDeviceSynchronize(), "mailbox sync");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "mailbox error set device");
        int e = 0;
        ck(cudaMemcpy(&e, error[static_cast<std::size_t>(g)], sizeof(e),
                      cudaMemcpyDeviceToHost), "mailbox copy error");
        if (e) {
            std::cerr << "mailbox GPU " << g << " error=" << e << '\n';
            return 4;
        }
    }

    std::cout << "ALL_OK"
              << " p2p_mailbox_ring=1"
              << " ngpu=" << ngpu
              << " token_bytes=" << TOKEN_WORDS * sizeof(std::uint32_t)
              << " payload_peer_writes=" << ngpu
              << " remote_state_reads=0"
              << " system_scope_release_acquire=1"
              << " native_atomic_mesh_required=1\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "mailbox free set device");
        cudaFree(error[static_cast<std::size_t>(g)]);
        cudaFree(mailbox[static_cast<std::size_t>(g)]);
    }
    return 0;
}
