#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

static constexpr int MAX_GPU = 8;
static constexpr int THREADS = 256;

struct alignas(16) QueueControl {
    unsigned int published = 0;
    unsigned int consumed = 0;
    unsigned int pad[2]{};
};

struct PairQueueDesc {
    QueueControl* out_ctrl = nullptr;      // peer-owned queue g -> peer
    std::uint32_t* out_payload = nullptr;
    QueueControl* in_ctrl = nullptr;       // local queue peer -> g
    std::uint32_t* in_payload = nullptr;
    int peer = -1;
};

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(90);
    }
}

void enable_full_native_peer_mesh(int ngpu) {
    for (int src = 0; src < ngpu; ++src) {
        ck(cudaSetDevice(src), "pair queue set peer source");
        for (int dst = 0; dst < ngpu; ++dst) {
            if (src == dst) continue;
            int can = 0;
            int native = 0;
            ck(cudaDeviceCanAccessPeer(&can, src, dst),
               "pair queue can access peer");
            if (!can) {
                std::cerr << "pair queue peer access unavailable src="
                          << src << " dst=" << dst << '\n';
                std::exit(91);
            }
            ck(cudaDeviceGetP2PAttribute(
                   &native, cudaDevP2PAttrNativeAtomicSupported, src, dst),
               "pair queue native atomic");
            if (!native) {
                std::cerr << "pair queue requires full native atomics src="
                          << src << " dst=" << dst << '\n';
                std::exit(92);
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "pair queue enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

__device__ __forceinline__ std::uint32_t token_word(
    int src, int dst, unsigned int message, int word
) {
    std::uint32_t x = 0x9e3779b9u * static_cast<std::uint32_t>(word + 1);
    x ^= 0x85ebca6bu * static_cast<std::uint32_t>(src + 1);
    x ^= 0xc2b2ae35u * static_cast<std::uint32_t>(dst + 1);
    x ^= 0x27d4eb2fu * (message + 1u);
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ bool wait_at_least(
    unsigned int* ptr,
    unsigned int want,
    int* error,
    int code
) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a(*ptr);
    constexpr unsigned int MAX_SPINS = 200000000u;
    for (unsigned int spin = 0; spin < MAX_SPINS; ++spin) {
        if (a.load(cuda::memory_order_acquire) >= want) return true;
        if ((spin & 255u) == 255u) __nanosleep(64);
    }
    atomicCAS(error, 0, code);
    return false;
}

__global__ void pair_spsc_queue_kernel(
    const PairQueueDesc* __restrict__ desc,
    int npairs,
    int gpu,
    unsigned int depth,
    int token_words,
    unsigned int messages,
    unsigned int batch,
    unsigned long long* __restrict__ sent_messages,
    unsigned long long* __restrict__ received_messages,
    int* error
) {
    if (blockIdx.x >= npairs) return;
    const PairQueueDesc q = desc[blockIdx.x];
    if (q.peer < 0) return;

    cuda::atomic_ref<unsigned int, cuda::thread_scope_system>
        out_published(q.out_ctrl->published);
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system>
        out_consumed(q.out_ctrl->consumed);
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system>
        in_published(q.in_ctrl->published);
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system>
        in_consumed(q.in_ctrl->consumed);

    const unsigned int mask = depth - 1u;
    for (unsigned int base = 0; base < messages; base += batch) {
        const unsigned int count = min(batch, messages - base);
        const unsigned int end = base + count;

        // Credit is metadata only. No remote state/payload read is performed by
        // the producer. Since this is SPSC, a monotone consumed tail is enough
        // to make ring reuse ABA-free.
        if (threadIdx.x == 0) {
            const unsigned int need = end > depth ? end - depth : 0u;
            if (!wait_at_least(&q.out_ctrl->consumed, need, error, 261))
                return;
        }
        __syncthreads();
        if (*error) return;

        for (unsigned int b = 0; b < count; ++b) {
            const unsigned int message = base + b;
            const unsigned int slot = message & mask;
            std::uint32_t* const dst =
                q.out_payload + static_cast<std::size_t>(slot) * token_words;
            for (int word = threadIdx.x; word < token_words; word += blockDim.x)
                dst[word] = token_word(gpu, q.peer, message, word);
        }

        // Every writer makes its peer payload stores system-visible before the
        // single release publication. The block barrier prevents lane 0 from
        // publishing until all writers have completed their system fences.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            out_published.store(end, cuda::memory_order_release);
            atomicAdd(sent_messages, static_cast<unsigned long long>(count));
        }
        __syncthreads();

        // Wait for the matching batch from peer -> gpu. Every thread performs
        // an acquire load after the leader has observed publication, so each
        // payload reader directly participates in the release/acquire edge.
        if (threadIdx.x == 0) {
            if (!wait_at_least(&q.in_ctrl->published, end, error, 262))
                return;
        }
        __syncthreads();
        if (*error) return;
        if (in_published.load(cuda::memory_order_acquire) < end) {
            atomicCAS(error, 0, 263);
            return;
        }
        __syncthreads();

        for (unsigned int b = 0; b < count; ++b) {
            const unsigned int message = base + b;
            const unsigned int slot = message & mask;
            const std::uint32_t* const src =
                q.in_payload + static_cast<std::size_t>(slot) * token_words;
            for (int word = threadIdx.x; word < token_words; word += blockDim.x) {
                const std::uint32_t expected =
                    token_word(q.peer, gpu, message, word);
                if (src[word] != expected) atomicCAS(error, 0, 264);
            }
        }
        __syncthreads();
        if (*error) return;

        if (threadIdx.x == 0) {
            in_consumed.store(end, cuda::memory_order_release);
            atomicAdd(received_messages, static_cast<unsigned long long>(count));
        }
        __syncthreads();
    }
}

struct QueueAlloc {
    QueueControl* ctrl = nullptr;
    std::uint32_t* payload = nullptr;
};

struct GpuCtx {
    PairQueueDesc* desc = nullptr;
    unsigned long long* sent = nullptr;
    unsigned long long* received = nullptr;
    int* error = nullptr;
};

} // namespace

int main(int argc, char** argv) {
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "pair queue device count");
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    const int token_words = argc > 2 ? std::atoi(argv[2]) : 1024;
    const unsigned int messages = argc > 3
        ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const unsigned int batch = argc > 4
        ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 8u;
    const unsigned int depth = argc > 5
        ? static_cast<unsigned int>(std::strtoul(argv[5], nullptr, 10)) : 64u;

    if (ngpu < 2 || ngpu > MAX_GPU || visible < ngpu || token_words <= 0 ||
        messages == 0 || batch == 0 || batch > depth || depth < 2 ||
        (depth & (depth - 1u)) != 0u) return 2;

    enable_full_native_peer_mesh(ngpu);

    const std::size_t queue_count =
        static_cast<std::size_t>(ngpu) * static_cast<std::size_t>(ngpu);
    std::vector<QueueAlloc> queue(queue_count);
    auto qidx = [ngpu](int dst, int src) {
        return static_cast<std::size_t>(dst) * static_cast<std::size_t>(ngpu) +
               static_cast<std::size_t>(src);
    };

    const std::size_t payload_words =
        static_cast<std::size_t>(depth) * static_cast<std::size_t>(token_words);
    for (int dst = 0; dst < ngpu; ++dst) {
        ck(cudaSetDevice(dst), "pair queue alloc destination");
        for (int src = 0; src < ngpu; ++src) {
            if (src == dst) continue;
            QueueAlloc& q = queue[qidx(dst, src)];
            ck(cudaMalloc(&q.ctrl, sizeof(QueueControl)), "pair queue alloc control");
            ck(cudaMemset(q.ctrl, 0, sizeof(QueueControl)), "pair queue zero control");
            ck(cudaMalloc(&q.payload, payload_words * sizeof(std::uint32_t)),
               "pair queue alloc payload");
        }
    }

    std::vector<GpuCtx> ctx(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        std::vector<PairQueueDesc> host_desc;
        host_desc.reserve(static_cast<std::size_t>(ngpu - 1));
        for (int peer = 0; peer < ngpu; ++peer) {
            if (peer == g) continue;
            const QueueAlloc& out = queue[qidx(peer, g)];
            const QueueAlloc& in = queue[qidx(g, peer)];
            host_desc.push_back(PairQueueDesc{
                out.ctrl, out.payload, in.ctrl, in.payload, peer});
        }

        ck(cudaSetDevice(g), "pair queue metadata set device");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].desc,
                      host_desc.size() * sizeof(PairQueueDesc)),
           "pair queue alloc descriptors");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].desc, host_desc.data(),
                      host_desc.size() * sizeof(PairQueueDesc), cudaMemcpyHostToDevice),
           "pair queue copy descriptors");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].sent,
                      sizeof(unsigned long long)), "pair queue alloc sent");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].received,
                      sizeof(unsigned long long)), "pair queue alloc received");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].error,
                      sizeof(int)), "pair queue alloc error");
        ck(cudaMemset(ctx[static_cast<std::size_t>(g)].sent, 0,
                      sizeof(unsigned long long)), "pair queue zero sent");
        ck(cudaMemset(ctx[static_cast<std::size_t>(g)].received, 0,
                      sizeof(unsigned long long)), "pair queue zero received");
        ck(cudaMemset(ctx[static_cast<std::size_t>(g)].error, 0,
                      sizeof(int)), "pair queue zero error");
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pair queue launch set device");
        pair_spsc_queue_kernel<<<ngpu - 1, THREADS>>>(
            ctx[static_cast<std::size_t>(g)].desc, ngpu - 1, g,
            depth, token_words, messages, batch,
            ctx[static_cast<std::size_t>(g)].sent,
            ctx[static_cast<std::size_t>(g)].received,
            ctx[static_cast<std::size_t>(g)].error);
        ck(cudaGetLastError(), "pair queue launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pair queue sync set device");
        ck(cudaDeviceSynchronize(), "pair queue sync");
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    unsigned long long total_sent = 0;
    unsigned long long total_received = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pair queue results set device");
        int error = 0;
        unsigned long long sent = 0, received = 0;
        ck(cudaMemcpy(&error, ctx[static_cast<std::size_t>(g)].error,
                      sizeof(error), cudaMemcpyDeviceToHost),
           "pair queue copy error");
        ck(cudaMemcpy(&sent, ctx[static_cast<std::size_t>(g)].sent,
                      sizeof(sent), cudaMemcpyDeviceToHost),
           "pair queue copy sent");
        ck(cudaMemcpy(&received, ctx[static_cast<std::size_t>(g)].received,
                      sizeof(received), cudaMemcpyDeviceToHost),
           "pair queue copy received");
        if (error) {
            std::cerr << "pair queue GPU " << g << " error=" << error << '\n';
            return 4;
        }
        const unsigned long long expected =
            static_cast<unsigned long long>(ngpu - 1) * messages;
        if (sent != expected || received != expected) {
            std::cerr << "pair queue count mismatch gpu=" << g
                      << " sent=" << sent << " received=" << received
                      << " expected=" << expected << '\n';
            return 5;
        }
        total_sent += sent;
        total_received += received;
    }

    const unsigned long long directed_pairs =
        static_cast<unsigned long long>(ngpu) * (ngpu - 1);
    const unsigned long long expected_total = directed_pairs * messages;
    if (total_sent != expected_total || total_received != expected_total) return 6;

    const double payload_bytes =
        double(expected_total) * double(token_words) * sizeof(std::uint32_t);
    const unsigned long long batches_per_pair =
        (static_cast<unsigned long long>(messages) + batch - 1ULL) / batch;
    const unsigned long long release_doorbells =
        directed_pairs * batches_per_pair;
    const double aggregate_gbs = ms > 0.0 ? payload_bytes / (ms * 1.0e6) : 0.0;
    const double queue_bytes =
        double(directed_pairs) *
        (double(sizeof(QueueControl)) +
         double(payload_words) * sizeof(std::uint32_t));

    std::cout << "ALL_OK"
              << " p2p_pair_spsc_queue=1"
              << " ngpu=" << ngpu
              << " directed_pairs=" << directed_pairs
              << " token_words=" << token_words
              << " token_bytes=" << token_words * sizeof(std::uint32_t)
              << " messages_per_pair=" << messages
              << " batch=" << batch
              << " depth=" << depth
              << " release_doorbells=" << release_doorbells
              << " payload_GiB=" << payload_bytes / double(1ULL << 30)
              << " queue_MiB=" << queue_bytes / double(1ULL << 20)
              << " wall_ms=" << ms
              << " aggregate_payload_GB_s=" << aggregate_gbs
              << " remote_state_reads=0"
              << " remote_credit_reads=1"
              << " destination_owned_payload=1"
              << " system_scope_release_acquire=1"
              << " full_native_atomic_mesh_required=1\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pair queue free metadata set device");
        cudaFree(ctx[static_cast<std::size_t>(g)].error);
        cudaFree(ctx[static_cast<std::size_t>(g)].received);
        cudaFree(ctx[static_cast<std::size_t>(g)].sent);
        cudaFree(ctx[static_cast<std::size_t>(g)].desc);
    }
    for (int dst = 0; dst < ngpu; ++dst) {
        ck(cudaSetDevice(dst), "pair queue free destination");
        for (int src = 0; src < ngpu; ++src) {
            if (src == dst) continue;
            QueueAlloc& q = queue[qidx(dst, src)];
            cudaFree(q.payload);
            cudaFree(q.ctrl);
        }
    }
    return 0;
}
