#pragma once

#include "ramstream32_b300_dual_tile_peer_swap.cuh"
#include "ramstream32_b300_dual_tile_reachability.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

// mode bit0: GPU a's current outgoing stream is structurally active.
// mode bit1: GPU b's current outgoing stream is structurally active.
// The two physical pair slots use the same stream-index coordinate even when
// their block boundaries differ.  Processing runs in that common coordinate
// avoids in-place corruption from block-by-block remapping.
__global__ void b300_dt_pruned_peer_kernel(
    Count* a, Count* b, Code begin, Code n, uint32_t mode
) {
    Code q = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    for (; q < n; q += step) {
        Code i = begin + q;
        if (mode == 3) {
            Count va = a[i];
            Count vb = b[i];
            a[i] = vb;
            b[i] = va;
        } else if (mode == 1) {
            Count va = a[i];
            b[i] = va;
            a[i] = 0;
        } else if (mode == 2) {
            Count vb = b[i];
            a[i] = vb;
            b[i] = 0;
        }
    }
}

struct B300DualPrunedPeerContext {
    int ngpu = 0;
    bool ready = false;
    // A directed stream on every endpoint lets one-sided active intervals be
    // launched from the source GPU.  One-way runs use the copy engine on that
    // stream; two-way runs use the register-swap kernel.
    std::array<std::array<cudaStream_t, MAXGPU>, MAXGPU> stream{};

    void init(int n) {
        if (ready) {
            if (ngpu != n) std::exit(655);
            return;
        }
        ngpu = n;
        for (int g = 0; g < ngpu; ++g) for (int p = 0; p < ngpu; ++p) if (g != p) {
            ck(cudaSetDevice(g), "dual pruned stream device");
            ck(cudaStreamCreateWithFlags(&stream[g][p], cudaStreamNonBlocking),
               "dual pruned stream");
        }
        ready = true;
    }

    void sync(const char* what) {
        if (!ready) return;
        for (int g = 0; g < ngpu; ++g) for (int p = 0; p < ngpu; ++p) if (g != p) {
            ck(cudaSetDevice(g), what);
            ck(cudaStreamSynchronize(stream[g][p]), what);
        }
    }

    void release() {
        if (!ready) return;
        sync("dual pruned release sync");
        for (int g = 0; g < ngpu; ++g) for (int p = 0; p < ngpu; ++p) if (g != p) {
            ck(cudaSetDevice(g), "dual pruned destroy device");
            cudaStreamDestroy(stream[g][p]);
            stream[g][p] = nullptr;
        }
        ready = false;
        ngpu = 0;
    }
};

static inline Code b300_dt_pruned_seg_len(
    const B300DualTileHost& z, const StorageBlock& b, int hi, int lo
) {
    if (!b.valid) return 0;
    return Code(z.high_count[hi][b.he]) * z.low_count[lo][b.hs];
}

static inline Code b300_dt_pruned_seg_off(
    const B300DualTileHost& z, bool blocked, int nblocks,
    int hi, int lo, int bid
) {
    const auto& off = blocked ? z.pair_block_off : z.pair_main_off;
    return off[(size_t(hi) * z.ngpu + lo) * nblocks + bid];
}

static inline bool b300_dt_pruned_stream_active_at(
    const B300DualTileHost& z, const StorageLayout& l,
    const B300DualReachStage& stage, bool blocked,
    int hi, int lo, Code pos
) {
    const int nb = blocked ? int(l.block_blocks.size()) : int(l.main_blocks.size());
    for (int bid = 0; bid < nb; ++bid) {
        const StorageBlock& b = blocked ? l.block_blocks[bid] : l.main_blocks[bid];
        Code n = b300_dt_pruned_seg_len(z, b, hi, lo);
        if (!n) continue;
        Code a = b300_dt_pruned_seg_off(z, blocked, nb, hi, lo, bid);
        if (pos >= a && pos < a + n) {
            if (blocked)
                return b300_dt_reach_block_active(stage, uint32_t(bid), hi, lo, z.ngpu);
            return b300_dt_reach_main_active(stage, uint32_t(bid), hi, lo, z.ngpu);
        }
    }
    return false; // outside this directed stream's logical length.
}

static inline void b300_dt_pruned_boundaries(
    const B300DualTileHost& z, const StorageLayout& l, bool blocked,
    int ahi, int alo, int bhi, int blo,
    std::vector<Code>& out
) {
    const int nb = blocked ? int(l.block_blocks.size()) : int(l.main_blocks.size());
    const auto& sz = blocked ? z.pair_block_size : z.pair_main_size;
    out.clear();
    out.reserve(size_t(nb) * 4 + 4);
    out.push_back(0);
    out.push_back(sz[ahi][alo]);
    out.push_back(sz[bhi][blo]);
    for (int bid = 0; bid < nb; ++bid) {
        const StorageBlock& ba = blocked ? l.block_blocks[bid] : l.main_blocks[bid];
        Code na = b300_dt_pruned_seg_len(z, ba, ahi, alo);
        Code nbv = b300_dt_pruned_seg_len(z, ba, bhi, blo);
        if (na) {
            Code a = b300_dt_pruned_seg_off(z, blocked, nb, ahi, alo, bid);
            out.push_back(a); out.push_back(a + na);
        }
        if (nbv) {
            Code a = b300_dt_pruned_seg_off(z, blocked, nb, bhi, blo, bid);
            out.push_back(a); out.push_back(a + nbv);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

static inline void b300_dt_pruned_launch_run(
    const B300DualTileHost& z, Count** ptrs, bool blocked,
    int a, int b, Code begin, Code n, uint32_t mode,
    B300DualPrunedPeerContext& ctx
) {
    if (!n || !mode) return;
    const auto& bs = blocked ? z.block_slot_base : z.main_slot_base;
    Count* ap = ptrs[a] + bs[a][b] + begin;
    Count* bp = ptrs[b] + bs[b][a] + begin;
    size_t bytes = size_t(n) * sizeof(Count);

    // A one-sided live interval is a plain contiguous migration.  Use the peer
    // copy engine, then zero the old source on the same source-device stream so
    // scratch/register swap work and SM occupancy are both avoided.
    if (mode == 1 || mode == 2) {
        int src = mode == 1 ? a : b;
        int dst = mode == 1 ? b : a;
        Count* sp = mode == 1 ? ap : bp;
        Count* dp = mode == 1 ? bp : ap;
        ck(cudaSetDevice(src), "dual pruned one-way device");
        cudaStream_t s = ctx.stream[src][dst];
        ck(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, s),
           "dual pruned one-way peer copy");
        ck(cudaMemsetAsync(sp, 0, bytes, s), "dual pruned one-way source zero");
        return;
    }

    // Both streams are live, so a true in-place swap is required.  A single
    // endpoint kernel reads both old values before either write.
    int active = B300DualPeerSwapContext::active_endpoint(a, b, z.ngpu);
    ck(cudaSetDevice(active), "dual pruned two-way device");
    int threads = 256;
    Code want = (n + threads - 1) / threads;
    unsigned blocks = unsigned(std::min<Code>(want, 8192));
    b300_dt_pruned_peer_kernel<<<blocks, threads, 0,
        ctx.stream[active][active == a ? b : a]>>>(
        ptrs[a] + bs[a][b], ptrs[b] + bs[b][a], begin, n, mode);
    ck(cudaGetLastError(), "dual pruned two-way launch");
}

static long double b300_dt_pruned_peer_array(
    const B300DualTileHost& z, const StorageLayout& l,
    Count** ptrs, bool blocked, bool low_to_high,
    const B300DualReachStage& stage,
    B300DualPrunedPeerContext& ctx,
    B300DualShuffleStats* stats = nullptr
) {
    if (!ctx.ready) ctx.init(z.ngpu);
    const auto& sz = blocked ? z.pair_block_size : z.pair_main_size;
    long double moved = 0;
    std::vector<Code> bounds;

    for (int a = 0; a < z.ngpu; ++a) for (int b = a + 1; b < z.ngpu; ++b) {
        // Logical outgoing stream currently held by each physical endpoint.
        int ahi, alo, bhi, blo;
        if (low_to_high) {
            ahi = b; alo = a; // GPU a holds stream(b,a)
            bhi = a; blo = b; // GPU b holds stream(a,b)
        } else {
            ahi = a; alo = b;
            bhi = b; blo = a;
        }
        b300_dt_pruned_boundaries(z, l, blocked, ahi, alo, bhi, blo, bounds);
        Code limit = std::max(sz[ahi][alo], sz[bhi][blo]);
        if (!limit) continue;

        Code run_begin = 0;
        uint32_t run_mode = 0;
        bool have_run = false;
        auto flush = [&](Code end) {
            if (!have_run || end <= run_begin) return;
            Code n = end - run_begin;
            b300_dt_pruned_launch_run(z, ptrs, blocked, a, b,
                                     run_begin, n, run_mode, ctx);
            moved += (long double)n * sizeof(Count)
                   * ((run_mode & 1u) != 0u) +
                     (long double)n * sizeof(Count)
                   * ((run_mode & 2u) != 0u);
        };

        for (size_t bi = 0; bi + 1 < bounds.size(); ++bi) {
            Code x = bounds[bi], y = bounds[bi + 1];
            if (x >= limit || y <= x) continue;
            y = std::min(y, limit);
            bool aa = x < sz[ahi][alo]
                && b300_dt_pruned_stream_active_at(z, l, stage, blocked, ahi, alo, x);
            bool bb = x < sz[bhi][blo]
                && b300_dt_pruned_stream_active_at(z, l, stage, blocked, bhi, blo, x);
            uint32_t mode = (aa ? 1u : 0u) | (bb ? 2u : 0u);
            if (!have_run) {
                run_begin = x; run_mode = mode; have_run = true;
            } else if (mode != run_mode) {
                flush(x);
                run_begin = x; run_mode = mode;
            }
        }
        flush(limit);
    }

    ctx.sync("dual pruned orientation sync");
    if (stats) {
        if (blocked) stats->block_bytes += moved;
        else stats->main_bytes += moved;
        stats->rounds += 1;
    }
    return moved;
}

static inline void b300_dt_pruned_low_to_high(
    const B300DualTileHost& z, const StorageLayout& l,
    Count** main_ptrs, Count** block_ptrs,
    const B300DualReachStage& stage,
    B300DualPrunedPeerContext& ctx,
    B300DualShuffleStats* stats = nullptr
) {
    b300_dt_pruned_peer_array(z, l, main_ptrs, false, true, stage, ctx, stats);
    b300_dt_pruned_peer_array(z, l, block_ptrs, true, true, stage, ctx, stats);
}

static inline void b300_dt_pruned_high_to_low_main(
    const B300DualTileHost& z, const StorageLayout& l,
    Count** main_ptrs, const B300DualReachStage& stage,
    B300DualPrunedPeerContext& ctx,
    B300DualShuffleStats* stats = nullptr
) {
    b300_dt_pruned_peer_array(z, l, main_ptrs, false, false, stage, ctx, stats);
}
