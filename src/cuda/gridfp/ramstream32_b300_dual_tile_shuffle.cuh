#pragma once

#include "ramstream32_b300_dual_tile_layout.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

struct B300DualShuffleStats {
    long double main_bytes = 0;
    long double block_bytes = 0;
    uint64_t rounds = 0;
    uint64_t chunk_barriers = 0;
};

static void b300_dt_sync_all(int ngpu, const char* what) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), what);
        ck(cudaDeviceSynchronize(), what);
    }
}

// Swap the two directed streams attached to every unordered GPU pair.
// low_to_high=true:
//   GPU a/slot b currently contains stream(b,a), and after the exchange must
//   contain stream(a,b).  The reverse holds on GPU b.
// high_to_low=false is exactly the inverse.
//
// Each GPU owns one scratch chunk.  For a round, all four disjoint pairs first
// stash the current chunk locally.  After a device-wide barrier, the stashed
// chunks are copied to the peer slots.  This is intentionally conservative and
// simple; an event-pipelined version can replace the two barriers later without
// changing the physical layout or transition kernels.
static long double b300_dt_shuffle_array(
    const B300DualTileHost& z,
    Count** ptrs,
    Count** scratch,
    Code chunk_elems,
    bool blocked,
    bool low_to_high,
    B300DualShuffleStats* stats = nullptr
) {
    if (!chunk_elems) std::exit(520);
    const int ngpu = z.ngpu;
    std::array<int, MAXGPU> ring{};
    for (int g = 0; g < ngpu; ++g) ring[g] = g;
    long double moved = 0;

    for (int round = 0; round < ngpu - 1; ++round) {
        struct Pair {
            int a = 0, b = 0;
            Code send_a = 0, send_b = 0;
            Code base_a = 0, base_b = 0;
        };
        std::array<Pair, MAXGPU / 2> pairs{};
        Code max_send = 0;
        for (int k = 0; k < ngpu / 2; ++k) {
            int a = ring[k], b = ring[ngpu - 1 - k];
            Pair x; x.a = a; x.b = b;
            const auto& sz = blocked ? z.pair_block_size : z.pair_main_size;
            const auto& bs = blocked ? z.block_slot_base : z.main_slot_base;
            if (low_to_high) {
                x.send_a = sz[b][a]; // stream(b,a): a -> b
                x.send_b = sz[a][b]; // stream(a,b): b -> a
            } else {
                x.send_a = sz[a][b]; // stream(a,b): a -> b
                x.send_b = sz[b][a]; // stream(b,a): b -> a
            }
            x.base_a = bs[a][b];
            x.base_b = bs[b][a];
            pairs[k] = x;
            max_send = std::max(max_send, std::max(x.send_a, x.send_b));
            moved += (long double)(x.send_a + x.send_b) * sizeof(Count);
        }

        for (Code off = 0; off < max_send; off += chunk_elems) {
            // Phase 1: preserve outgoing bytes before either peer overwrites the
            // pair slot.  Pairs are disjoint, so every GPU uses one scratch.
            for (int k = 0; k < ngpu / 2; ++k) {
                const Pair& x = pairs[k];
                if (off < x.send_a) {
                    Code n = std::min(chunk_elems, x.send_a - off);
                    ck(cudaSetDevice(x.a), "dual stash a device");
                    ck(cudaMemcpyAsync(scratch[x.a], ptrs[x.a] + x.base_a + off,
                                       size_t(n) * sizeof(Count), cudaMemcpyDeviceToDevice),
                       "dual stash a");
                }
                if (off < x.send_b) {
                    Code n = std::min(chunk_elems, x.send_b - off);
                    ck(cudaSetDevice(x.b), "dual stash b device");
                    ck(cudaMemcpyAsync(scratch[x.b], ptrs[x.b] + x.base_b + off,
                                       size_t(n) * sizeof(Count), cudaMemcpyDeviceToDevice),
                       "dual stash b");
                }
            }
            b300_dt_sync_all(ngpu, "dual stash sync");

            // Phase 2: send preserved chunks into the opposite slot.  The slot
            // capacity is max(stream(a,b),stream(b,a)), so either direction fits.
            for (int k = 0; k < ngpu / 2; ++k) {
                const Pair& x = pairs[k];
                if (off < x.send_a) {
                    Code n = std::min(chunk_elems, x.send_a - off);
                    ck(cudaSetDevice(x.b), "dual peer a2b device");
                    ck(cudaMemcpyPeerAsync(ptrs[x.b] + x.base_b + off, x.b,
                                           scratch[x.a], x.a,
                                           size_t(n) * sizeof(Count)),
                       "dual peer a2b");
                }
                if (off < x.send_b) {
                    Code n = std::min(chunk_elems, x.send_b - off);
                    ck(cudaSetDevice(x.a), "dual peer b2a device");
                    ck(cudaMemcpyPeerAsync(ptrs[x.a] + x.base_a + off, x.a,
                                           scratch[x.b], x.b,
                                           size_t(n) * sizeof(Count)),
                       "dual peer b2a");
                }
            }
            b300_dt_sync_all(ngpu, "dual peer sync");
            if (stats) ++stats->chunk_barriers;
        }
        if (stats) ++stats->rounds;

        int last = ring[ngpu - 1];
        for (int k = ngpu - 1; k >= 2; --k) ring[k] = ring[k - 1];
        ring[1] = last;
    }
    if (stats) {
        if (blocked) stats->block_bytes += moved;
        else stats->main_bytes += moved;
    }
    return moved;
}

static void b300_dt_low_to_high(
    const B300DualTileHost& z,
    Count** main_ptrs, Count** block_ptrs, Count** scratch,
    Code chunk_elems, B300DualShuffleStats* stats = nullptr
) {
    b300_dt_shuffle_array(z, main_ptrs, scratch, chunk_elems, false, true, stats);
    b300_dt_shuffle_array(z, block_ptrs, scratch, chunk_elems, true, true, stats);
}

static void b300_dt_high_to_low_main(
    const B300DualTileHost& z,
    Count** main_ptrs, Count** scratch,
    Code chunk_elems, B300DualShuffleStats* stats = nullptr
) {
    b300_dt_shuffle_array(z, main_ptrs, scratch, chunk_elems, false, false, stats);
}

// After p=1 the logical blocked vector is identically zero.  We do not shuffle
// it back to LOW orientation; instead clear the entire pair-slot arena so even
// slot tails (which can be larger than the currently active directed stream)
// are guaranteed zero for the next row's HIGH window.
static void b300_dt_zero_block_arenas(
    const B300DualTileHost& z, Count** block_ptrs
) {
    for (int g = 0; g < z.ngpu; ++g) {
        ck(cudaSetDevice(g), "dual zero block device");
        if (z.block_count[g])
            ck(cudaMemsetAsync(block_ptrs[g], 0, size_t(z.block_count[g]) * sizeof(Count)),
               "dual zero block arena");
    }
    b300_dt_sync_all(z.ngpu, "dual zero block sync");
}
