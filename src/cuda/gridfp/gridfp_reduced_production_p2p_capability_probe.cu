#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(90);
    }
}

struct PairCaps {
    int can_access = 0;
    int performance_rank = -1;
    int native_atomic = 0;
#if CUDART_VERSION >= 13000
    int partial_native_atomic = 0;
#endif
};

PairCaps query_pair(int src, int dst) {
    PairCaps z;
    ck(cudaDeviceCanAccessPeer(&z.can_access, src, dst), "p2p can access");
    ck(cudaDeviceGetP2PAttribute(
           &z.performance_rank, cudaDevP2PAttrPerformanceRank, src, dst),
       "p2p performance rank");
    ck(cudaDeviceGetP2PAttribute(
           &z.native_atomic, cudaDevP2PAttrNativeAtomicSupported, src, dst),
       "p2p native atomic");
#if CUDART_VERSION >= 13000
    ck(cudaDeviceGetP2PAttribute(
           &z.partial_native_atomic,
           cudaDevP2PAttrOnlyPartialNativeAtomicSupported, src, dst),
       "p2p partial native atomic");
#endif
    return z;
}

} // namespace

int main(int argc, char** argv) {
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "p2p capability device count");
    const int requested = argc > 1 ? std::atoi(argv[1]) : 8;
    if (requested < 2 || requested > 8) return 2;
    if (visible < requested) {
        std::cerr << "need " << requested << " CUDA devices, visible=" << visible << '\n';
        return 3;
    }

    for (int g = 0; g < requested; ++g) {
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, g), "p2p capability device properties");
        std::cout << "p2p-device"
                  << " gpu=" << g
                  << " name=\"" << prop.name << "\""
                  << " cc=" << prop.major << '.' << prop.minor
                  << " global_GiB="
                  << double(prop.totalGlobalMem) / double(1ULL << 30)
                  << " pci=" << prop.pciDomainID << ':'
                  << prop.pciBusID << ':' << prop.pciDeviceID
                  << '\n';
    }

    bool full_peer_mesh = true;
    bool full_native_atomic_mesh = true;
#if CUDART_VERSION >= 13000
    bool full_or_partial_atomic_mesh = true;
#endif
    int worst_rank = 0;

    for (int src = 0; src < requested; ++src) {
        for (int dst = 0; dst < requested; ++dst) {
            if (src == dst) continue;
            const PairCaps z = query_pair(src, dst);
            full_peer_mesh = full_peer_mesh && z.can_access;
            full_native_atomic_mesh = full_native_atomic_mesh && z.native_atomic;
#if CUDART_VERSION >= 13000
            full_or_partial_atomic_mesh = full_or_partial_atomic_mesh &&
                (z.native_atomic || z.partial_native_atomic);
#endif
            if (z.performance_rank > worst_rank) worst_rank = z.performance_rank;

            std::cout << "p2p-link"
                      << " src=" << src
                      << " dst=" << dst
                      << " can_access=" << z.can_access
                      << " performance_rank=" << z.performance_rank
                      << " native_atomic=" << z.native_atomic
#if CUDART_VERSION >= 13000
                      << " partial_native_atomic=" << z.partial_native_atomic
#endif
                      << '\n';
        }
    }

    std::cout << "p2p-capability-summary"
              << " requested_gpus=" << requested
              << " visible_gpus=" << visible
              << " full_peer_mesh=" << (full_peer_mesh ? 1 : 0)
              << " full_native_atomic_mesh="
              << (full_native_atomic_mesh ? 1 : 0)
#if CUDART_VERSION >= 13000
              << " full_or_partial_atomic_mesh="
              << (full_or_partial_atomic_mesh ? 1 : 0)
#endif
              << " worst_performance_rank=" << worst_rank
              << " cudart_version=" << CUDART_VERSION
              << '\n';

    // A peer-memory mailbox can be considered only with a full peer mesh.  A
    // native-atomic doorbell fast path additionally requires full native atomic
    // support in both directions for every pair.  Partial support is reported
    // but is not accepted here until the exact doorbell operation is queried.
    if (!full_peer_mesh) return 4;
    std::cout << "ALL_OK"
              << " full_peer_mesh=1"
              << " native_atomic_mailbox_fastpath="
              << (full_native_atomic_mesh ? 1 : 0)
              << '\n';
    return 0;
}
