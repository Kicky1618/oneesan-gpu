#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

// v0.4: keep destination-owned gather on one compute stream per GPU, but move
// bulk peer staging to one copy stream per (destination, source) GPU pair.
// Blocks from different peers can therefore traverse different NVLink paths at
// the same time instead of being serialized on the destination compute stream.

struct GdmpPeerStats {
    unsigned long long copy_ops_per_row = 0;
    unsigned long long max_peer_pair_bytes_per_row = 0;
    int active_peer_pairs = 0;
};

static GdmpPeerStats gdmp_peer_stats(
    const GdmsStagePlan& plan,
    const StorageLayout& layout,
    const GdmShardHost& shard,
    int ngpu
) {
    GdmpPeerStats out;
    std::array<std::array<unsigned long long,GDM_MAX_GPU>,GDM_MAX_GPU> bytes{};
    auto add = [&](const std::array<std::vector<std::uint8_t>,GDM_MAX_GPU>& srcs) {
        for (int d=0; d<ngpu; ++d) {
            for (std::uint8_t sbid8 : srcs[d]) {
                std::uint32_t sbid = sbid8;
                int s = shard.main_blocks[sbid].owner;
                if (s == d) continue;
                const auto& b = layout.main_blocks[sbid];
                unsigned long long z = static_cast<unsigned long long>(b.rows) * b.cols * sizeof(Count);
                bytes[d][s] += z;
                ++out.copy_ops_per_row;
            }
        }
    };
    for (const auto& ph : plan.high) { add(ph.source); add(ph.cross_refresh); }
    for (const auto& ph : plan.low)  { add(ph.source); add(ph.cross_refresh); }
    for (int d=0; d<ngpu; ++d) for (int s=0; s<ngpu; ++s) if (d != s && bytes[d][s]) {
        ++out.active_peer_pairs;
        out.max_peer_pair_bytes_per_row = std::max(out.max_peer_pair_bytes_per_row, bytes[d][s]);
    }
    return out;
}

struct GdmpStagePipeline {
    int ngpu = 0;
    int slots = 0;
    Code stage_elems = 0;
    std::array<cudaStream_t,GDM_MAX_GPU> compute{};
    std::array<std::array<cudaStream_t,GDM_MAX_GPU>,GDM_MAX_GPU> copy{};
    std::array<Count*,GDM_MAX_GPU> stage{};
    std::vector<std::array<cudaEvent_t,GDM_MAX_GPU>> fence_event;
    std::vector<std::array<std::array<cudaEvent_t,GDM_MAX_GPU>,GDM_MAX_GPU>> copy_done;

    void init(int n, const StorageLayout& layout) {
        ngpu = n;
        slots = 2 * (LOW_LUT_K + HIGH_LUT_K) + 1;
        stage_elems = layout.main_size;
        fence_event.resize(std::size_t(slots));
        copy_done.resize(std::size_t(slots));

        std::array<Code,GPU_DIRECT_MAX_MAIN_BLOCKS> global_off{};
        for (std::size_t i=0; i<layout.main_blocks.size(); ++i)
            global_off[i] = layout.main_blocks[i].off;

        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp set device");
            ck(cudaStreamCreateWithFlags(&compute[d], cudaStreamNonBlocking), "gdmp compute stream");
            for (int s=0; s<ngpu; ++s) if (s != d)
                ck(cudaStreamCreateWithFlags(&copy[d][s], cudaStreamNonBlocking), "gdmp peer copy stream");
            if (stage_elems)
                ck(cudaMalloc(&stage[d], std::size_t(stage_elems) * sizeof(Count)), "gdmp stage");
            ck(cudaMemcpyToSymbol(D_GDMS_STAGE_MAIN, &stage[d], sizeof(stage[d])), "gdmp stage ptr");
            ck(cudaMemcpyToSymbol(D_GDMS_MAIN_GLOBAL_OFF, global_off.data(),
                                  sizeof(Code) * global_off.size()), "gdmp global off");
        }

        for (int k=0; k<slots; ++k) {
            for (int d=0; d<ngpu; ++d) {
                ck(cudaSetDevice(d), "gdmp event device");
                ck(cudaEventCreateWithFlags(&fence_event[std::size_t(k)][d], cudaEventDisableTiming),
                   "gdmp fence event");
                for (int s=0; s<ngpu; ++s) if (s != d)
                    ck(cudaEventCreateWithFlags(&copy_done[std::size_t(k)][d][s], cudaEventDisableTiming),
                       "gdmp copy event");
            }
        }
    }

    // Preserve the v0.3 all-GPU phase ordering. The expensive host-side
    // cudaDeviceSynchronize remains absent; all waits are queued on-device.
    void fence(int slot) {
        if (slot < 0 || slot >= slots) std::exit(181);
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp fence record device");
            ck(cudaEventRecord(fence_event[std::size_t(slot)][d], compute[d]), "gdmp fence record");
        }
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp fence wait device");
            for (int s=0; s<ngpu; ++s) if (s != d)
                ck(cudaStreamWaitEvent(compute[d], fence_event[std::size_t(slot)][s], 0),
                   "gdmp fence wait");
        }
    }

    void sync_row() {
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp sync device");
            ck(cudaStreamSynchronize(compute[d]), "gdmp row sync");
        }
    }

    void destroy() {
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp destroy device");
            for (int k=0; k<slots; ++k) {
                if (fence_event[std::size_t(k)][d])
                    cudaEventDestroy(fence_event[std::size_t(k)][d]);
                for (int s=0; s<ngpu; ++s) if (s != d && copy_done[std::size_t(k)][d][s])
                    cudaEventDestroy(copy_done[std::size_t(k)][d][s]);
            }
            for (int s=0; s<ngpu; ++s) if (s != d && copy[d][s])
                cudaStreamDestroy(copy[d][s]);
            if (compute[d]) cudaStreamDestroy(compute[d]);
            if (stage[d]) cudaFree(stage[d]);
            compute[d] = nullptr;
            stage[d] = nullptr;
        }
        fence_event.clear();
        copy_done.clear();
    }
};

static void gdmp_stage_sources(
    GdmpStagePipeline& pipe,
    int ready_slot,
    const std::array<std::vector<std::uint8_t>,GDM_MAX_GPU>& sources,
    const StorageLayout& layout,
    const GdmShardHost& shard,
    Count* const* main_ptr
) {
    if (ready_slot < 0 || ready_slot >= pipe.slots) std::exit(182);
    for (int d=0; d<pipe.ngpu; ++d) {
        std::array<bool,GDM_MAX_GPU> used{};
        ck(cudaSetDevice(d), "gdmp copy device");
        for (std::uint8_t sbid8 : sources[d]) {
            std::uint32_t sbid = sbid8;
            const auto& logical = layout.main_blocks[sbid];
            const auto& physical = shard.main_blocks[sbid];
            int s = physical.owner;
            if (s == d) continue;
            cudaStream_t cs = pipe.copy[d][s];
            if (!used[s]) {
                // Source event protects the authoritative source bytes. The
                // destination event prevents this phase from overwriting a
                // mirror range while the previous destination phase uses it.
                ck(cudaStreamWaitEvent(cs, pipe.fence_event[std::size_t(ready_slot)][s], 0),
                   "gdmp copy wait source");
                ck(cudaStreamWaitEvent(cs, pipe.fence_event[std::size_t(ready_slot)][d], 0),
                   "gdmp copy wait destination");
                used[s] = true;
            }
            std::size_t bytes = std::size_t(logical.rows) * logical.cols * sizeof(Count);
            if (!bytes) continue;
            Count* dst = pipe.stage[d] + logical.off;
            const Count* src = main_ptr[s] + physical.off;
            ck(cudaMemcpyPeerAsync(dst, d, src, s, bytes, cs), "gdmp peer bulk copy");
        }
        for (int s=0; s<pipe.ngpu; ++s) if (used[s]) {
            cudaEvent_t done = pipe.copy_done[std::size_t(ready_slot)][d][s];
            ck(cudaEventRecord(done, pipe.copy[d][s]), "gdmp copy done");
            ck(cudaStreamWaitEvent(pipe.compute[d], done, 0), "gdmp compute wait copy");
        }
    }
}

static void gdmp_enqueue_high(
    GdmpStagePipeline& pipe, const GdmsStagePlan& plan,
    const StorageLayout& layout, const GdmShardHost& shard,
    Count* const* main_ptr, int threads, int grid_x, int grid_y, int& slot
) {
    dim3 block(threads);
    for (int p=TARGET_W-1; p>=LOW_LUT_K+1; --p) {
        std::uint32_t pi = std::uint32_t((TARGET_W-1)-p);
        for (int d=0; d<pipe.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp high orbit device");
            dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));
            gdm_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(), "gdmp high orbit");
        }
        int ready = slot;
        pipe.fence(slot++);
        gdmp_stage_sources(pipe, ready, plan.high[pi].source, layout, shard, main_ptr);
        for (int d=0; d<pipe.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp high gather device");
            dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size()));
            gdms_high_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            gdms_high_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(), "gdmp high gather");
        }
        pipe.fence(slot++);
    }
}

static void gdmp_enqueue_low(
    GdmpStagePipeline& pipe, const GdmsStagePlan& plan,
    const StorageLayout& layout, const GdmShardHost& shard,
    Count* const* main_ptr, int threads, int grid_x, int grid_y, int& slot
) {
    dim3 block(threads);
    for (int p=LOW_LUT_K; p>=1; --p) {
        std::uint32_t pi = std::uint32_t(LOW_LUT_K-p);
        for (int d=0; d<pipe.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp low orbit device");
            dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));
            gdm_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(), "gdmp low orbit");
        }
        int ready = slot;
        pipe.fence(slot++);
        gdmp_stage_sources(pipe, ready, plan.low[pi].source, layout, shard, main_ptr);
        unsigned nt = p==1 ? unsigned(layout.main_blocks.size())
                           : unsigned(layout.block_blocks.size());
        for (int d=0; d<pipe.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp low local device");
            dim3 g(grid_x,grid_y,nt);
            gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(), "gdmp low local");
        }
        if (p == 1) {
            ready = slot;
            pipe.fence(slot++);
            gdmp_stage_sources(pipe, ready, plan.low[pi].cross_refresh, layout, shard, main_ptr);
        }
        for (int d=0; d<pipe.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdmp low cross device");
            dim3 g(grid_x,grid_y,nt);
            gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(), "gdmp low cross");
        }
        pipe.fence(slot++);
    }
}
