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
    uint64_t chunk_barriers = 0; // retained name: now counts pipelined chunk steps
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
// One nonblocking stream and one event are used per GPU.  For each chunk the
// outgoing slot is copied to local scratch, then an event is recorded.  The
// peer's stream waits on that cross-device event before overwriting the slot.
// Because the peer copy reading scratch is on the same source stream, the next
// local stash on that stream cannot reuse scratch too early.  CUDA explicitly
// permits cudaStreamWaitEvent() across devices, so the old all-GPU barrier per
// chunk is unnecessary.
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
    std::array<cudaStream_t, MAXGPU> stream{};
    std::array<cudaEvent_t, MAXGPU> stash_done{};
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "dual shuffle stream device");
        ck(cudaStreamCreateWithFlags(&stream[g], cudaStreamNonBlocking), "dual shuffle stream");
        ck(cudaEventCreateWithFlags(&stash_done[g], cudaEventDisableTiming), "dual shuffle event");
    }

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
            // Preserve both outgoing chunks and publish their completion.
            for (int k = 0; k < ngpu / 2; ++k) {
                const Pair& x = pairs[k];
                ck(cudaSetDevice(x.a), "dual pipeline stash a device");
                if (off < x.send_a) {
                    Code n = std::min(chunk_elems, x.send_a - off);
                    ck(cudaMemcpyAsync(scratch[x.a], ptrs[x.a] + x.base_a + off,
                                       size_t(n) * sizeof(Count), cudaMemcpyDeviceToDevice,
                                       stream[x.a]), "dual pipeline stash a");
                }
                ck(cudaEventRecord(stash_done[x.a], stream[x.a]), "dual pipeline event a");

                ck(cudaSetDevice(x.b), "dual pipeline stash b device");
                if (off < x.send_b) {
                    Code n = std::min(chunk_elems, x.send_b - off);
                    ck(cudaMemcpyAsync(scratch[x.b], ptrs[x.b] + x.base_b + off,
                                       size_t(n) * sizeof(Count), cudaMemcpyDeviceToDevice,
                                       stream[x.b]), "dual pipeline stash b");
                }
                ck(cudaEventRecord(stash_done[x.b], stream[x.b]), "dual pipeline event b");
            }

            // Each outbound copy waits until the destination GPU has preserved
            // its old bytes.  Put the copy on the source stream so scratch reuse
            // for the next chunk is automatically ordered after the peer read.
            for (int k = 0; k < ngpu / 2; ++k) {
                const Pair& x = pairs[k];
                ck(cudaSetDevice(x.a), "dual pipeline a2b device");
                ck(cudaStreamWaitEvent(stream[x.a], stash_done[x.b], 0), "dual wait peer b stash");
                if (off < x.send_a) {
                    Code n = std::min(chunk_elems, x.send_a - off);
                    ck(cudaMemcpyPeerAsync(ptrs[x.b] + x.base_b + off, x.b,
                                           scratch[x.a], x.a,
                                           size_t(n) * sizeof(Count), stream[x.a]),
                       "dual pipeline peer a2b");
                }

                ck(cudaSetDevice(x.b), "dual pipeline b2a device");
                ck(cudaStreamWaitEvent(stream[x.b], stash_done[x.a], 0), "dual wait peer a stash");
                if (off < x.send_b) {
                    Code n = std::min(chunk_elems, x.send_b - off);
                    ck(cudaMemcpyPeerAsync(ptrs[x.a] + x.base_a + off, x.a,
                                           scratch[x.b], x.b,
                                           size_t(n) * sizeof(Count), stream[x.b]),
                       "dual pipeline peer b2a");
                }
            }
            if (stats) ++stats->chunk_barriers;
        }
        if (stats) ++stats->rounds;

        int last = ring[ngpu - 1];
        for (int k = ngpu - 1; k >= 2; --k) ring[k] = ring[k - 1];
        ring[1] = last;
    }

    // Synchronizing all source streams also waits for every inbound transfer,
    // because each directed peer copy is present on exactly one of these streams.
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "dual pipeline final device");
        ck(cudaStreamSynchronize(stream[g]), "dual pipeline final sync");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "dual pipeline destroy device");
        cudaEventDestroy(stash_done[g]);
        cudaStreamDestroy(stream[g]);
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
