#pragma once

#include "ramstream32_b300_dual_tile_layout.cuh"
#include "ramstream32_b300_dual_tile_shuffle.cuh"

#include <algorithm>
#include <array>
#include <cstdint>

// Scratch-free transpose of one unordered GPU-pair slot.
// The launching GPU has peer access to both allocations.  Every thread loads
// both old values before writing either destination, so the overlap region is a
// true register swap.  If the directed stream lengths differ, the longer tail
// is copied into the peer's larger pair-slot capacity; the shorter old tail is
// dead after the orientation change and need not be preserved.
__global__ void b300_dt_peer_swap_kernel(
    Count* a, Count* b, Code na, Code nb
) {
    Code n = na > nb ? na : nb;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    Code common = na < nb ? na : nb;
    for (; i < n; i += step) {
        if (i < common) {
            Count va = a[i];
            Count vb = b[i];
            a[i] = vb;
            b[i] = va;
        } else if (i < na) {
            b[i] = a[i];
        } else {
            a[i] = b[i];
        }
    }
}

struct B300DualPeerSwapContext {
    int ngpu = 0;
    bool ready = false;
    // One stream per unordered peer slot from the active endpoint.  The matrix
    // form keeps lookup trivial and allows independent pair kernels to overlap.
    std::array<std::array<cudaStream_t, MAXGPU>, MAXGPU> stream{};

    static int active_endpoint(int a, int b, int ngpu) {
        // Near-regular orientation of K8: each GPU launches only 3 or 4 of the
        // 7 incident pair kernels.  Fallback alternates by parity for other n.
        if (ngpu == 8) {
            int d = (b - a + 8) & 7;
            if (d < 4) return a;
            if (d > 4) return b;
            return (a & 1) ? b : a;
        }
        return ((a + b) & 1) ? a : b;
    }

    void init(int n) {
        if (ready) {
            if (ngpu != n) std::exit(525);
            return;
        }
        ngpu = n;
        for (int a = 0; a < ngpu; ++a) for (int b = a + 1; b < ngpu; ++b) {
            int g = active_endpoint(a, b, ngpu);
            ck(cudaSetDevice(g), "dual peer-swap stream device");
            ck(cudaStreamCreateWithFlags(&stream[a][b], cudaStreamNonBlocking),
               "dual peer-swap stream");
        }
        ready = true;
    }

    void sync(const char* what) {
        if (!ready) return;
        for (int a = 0; a < ngpu; ++a) for (int b = a + 1; b < ngpu; ++b) {
            int g = active_endpoint(a, b, ngpu);
            ck(cudaSetDevice(g), what);
            ck(cudaStreamSynchronize(stream[a][b]), what);
        }
    }

    void release() {
        if (!ready) return;
        sync("dual peer-swap release sync");
        for (int a = 0; a < ngpu; ++a) for (int b = a + 1; b < ngpu; ++b) {
            int g = active_endpoint(a, b, ngpu);
            ck(cudaSetDevice(g), "dual peer-swap destroy device");
            cudaStreamDestroy(stream[a][b]);
            stream[a][b] = nullptr;
        }
        ready = false;
        ngpu = 0;
    }
};

static long double b300_dt_peer_swap_array(
    const B300DualTileHost& z,
    Count** ptrs,
    bool blocked,
    bool low_to_high,
    B300DualPeerSwapContext& ctx,
    B300DualShuffleStats* stats = nullptr
) {
    ctx.init(z.ngpu);
    const auto& sz = blocked ? z.pair_block_size : z.pair_main_size;
    const auto& bs = blocked ? z.block_slot_base : z.main_slot_base;
    long double moved = 0;

    // All 28 K8 pair slots are disjoint, so no edge-coloring barrier is needed
    // for correctness.  Launch them all; the streams/hardware arbitrate HBM and
    // NVLink bandwidth.  A later B300 benchmark can compare this against the
    // 7-round cudaMemcpyPeer engine without changing the physical layout.
    for (int a = 0; a < z.ngpu; ++a) for (int b = a + 1; b < z.ngpu; ++b) {
        Code na, nb;
        if (low_to_high) {
            na = sz[b][a]; // GPU a currently holds stream(b,a)
            nb = sz[a][b]; // GPU b currently holds stream(a,b)
        } else {
            na = sz[a][b];
            nb = sz[b][a];
        }
        moved += (long double)(na + nb) * sizeof(Count);
        Code n = std::max(na, nb);
        if (!n) continue;
        int active = B300DualPeerSwapContext::active_endpoint(a, b, z.ngpu);
        ck(cudaSetDevice(active), "dual peer-swap launch device");
        int threads = 256;
        Code want = (n + threads - 1) / threads;
        unsigned blocks = unsigned(std::min<Code>(want, 8192));
        b300_dt_peer_swap_kernel<<<blocks, threads, 0, ctx.stream[a][b]>>>(
            ptrs[a] + bs[a][b], ptrs[b] + bs[b][a], na, nb);
        ck(cudaGetLastError(), "dual peer-swap launch");
    }
    ctx.sync("dual peer-swap orientation sync");
    if (stats) {
        if (blocked) stats->block_bytes += moved;
        else stats->main_bytes += moved;
        stats->rounds += 1; // one fully asynchronous pair set
    }
    return moved;
}

static void b300_dt_peer_low_to_high(
    const B300DualTileHost& z,
    Count** main_ptrs, Count** block_ptrs,
    B300DualPeerSwapContext& ctx,
    B300DualShuffleStats* stats = nullptr
) {
    b300_dt_peer_swap_array(z, main_ptrs, false, true, ctx, stats);
    b300_dt_peer_swap_array(z, block_ptrs, true, true, ctx, stats);
}

static void b300_dt_peer_high_to_low_main(
    const B300DualTileHost& z,
    Count** main_ptrs,
    B300DualPeerSwapContext& ctx,
    B300DualShuffleStats* stats = nullptr
) {
    b300_dt_peer_swap_array(z, main_ptrs, false, false, ctx, stats);
}
