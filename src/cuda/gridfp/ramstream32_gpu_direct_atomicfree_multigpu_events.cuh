#pragma once

#include <cuda_runtime.h>

#include <array>
#include <vector>

// Device-side all-GPU phase fences for the block-sharded atomic-free backend.
// Every logical fence gets its own event slot within a row, so no event is
// re-recorded until the previous row has been fully synchronized.
struct GdmEventPipeline {
    int ngpu = 0;
    int slots = 0;
    std::array<cudaStream_t,GDM_MAX_GPU> stream{};
    std::vector<std::array<cudaEvent_t,GDM_MAX_GPU>> event;

    void init(int n) {
        ngpu = n;
        slots = 3 * (LOW_LUT_K + HIGH_LUT_K);
        event.resize(size_t(slots));
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event set device");
            ck(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking),
               "gdm event stream");
        }
        for (int s=0; s<slots; ++s) {
            for (int d=0; d<ngpu; ++d) {
                ck(cudaSetDevice(d), "gdm event create device");
                ck(cudaEventCreateWithFlags(&event[size_t(s)][d], cudaEventDisableTiming),
                   "gdm event create");
            }
        }
    }

    void fence(int slot) {
        if (slot < 0 || slot >= slots) std::exit(175);
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event record device");
            ck(cudaEventRecord(event[size_t(slot)][d], stream[d]),
               "gdm event record");
        }
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event wait device");
            for (int src=0; src<ngpu; ++src) {
                if (src == d) continue;
                ck(cudaStreamWaitEvent(stream[d], event[size_t(slot)][src], 0),
                   "gdm cross-device event wait");
            }
        }
    }

    void sync_row() {
        // One logical host barrier point per row. Synchronizing all streams also
        // surfaces asynchronous launch/runtime errors on their owning devices.
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm row sync device");
            ck(cudaStreamSynchronize(stream[d]), "gdm row sync");
        }
    }

    void destroy() {
        for (int s=0; s<slots; ++s) {
            for (int d=0; d<ngpu; ++d) {
                if (!event[size_t(s)][d]) continue;
                cudaSetDevice(d);
                cudaEventDestroy(event[size_t(s)][d]);
                event[size_t(s)][d] = nullptr;
            }
        }
        for (int d=0; d<ngpu; ++d) {
            if (!stream[d]) continue;
            cudaSetDevice(d);
            cudaStreamDestroy(stream[d]);
            stream[d] = nullptr;
        }
        event.clear();
        ngpu = 0;
        slots = 0;
    }
};

static void gdm_enqueue_high_events(
    GdmEventPipeline& q, const StorageLayout& layout,
    int threads, int grid_x, int grid_y, int& slot
) {
    dim3 block(threads);
    for (int p=TARGET_W-1; p>=LOW_LUT_K+1; --p) {
        for (int d=0; d<q.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event high orbit device");
            dim3 g(grid_x, grid_y, unsigned(layout.main_blocks.size()));
            gdm_high_orbit_kernel<<<g,block,0,q.stream[d]>>>(p);
            ck(cudaGetLastError(), "gdm event high orbit");
        }
        q.fence(slot++);

        for (int d=0; d<q.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event high local device");
            dim3 g(grid_x, grid_y, unsigned(layout.block_blocks.size()));
            gdm_high_local_gather_kernel<<<g,block,0,q.stream[d]>>>(p);
            ck(cudaGetLastError(), "gdm event high local");
        }
        q.fence(slot++);

        for (int d=0; d<q.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event high cross device");
            dim3 g(grid_x, grid_y, unsigned(layout.block_blocks.size()));
            gdm_high_cross_gather_kernel<<<g,block,0,q.stream[d]>>>(p);
            ck(cudaGetLastError(), "gdm event high cross");
        }
        q.fence(slot++);
    }
}

static void gdm_enqueue_low_events(
    GdmEventPipeline& q, const StorageLayout& layout,
    int threads, int grid_x, int grid_y, int& slot
) {
    dim3 block(threads);
    for (int p=LOW_LUT_K; p>=1; --p) {
        for (int d=0; d<q.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event low orbit device");
            dim3 g(grid_x, grid_y, unsigned(layout.main_blocks.size()));
            gdm_low_orbit_kernel<<<g,block,0,q.stream[d]>>>(p);
            ck(cudaGetLastError(), "gdm event low orbit");
        }
        q.fence(slot++);

        unsigned nt = p==1
            ? unsigned(layout.main_blocks.size())
            : unsigned(layout.block_blocks.size());
        for (int d=0; d<q.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event low local device");
            dim3 g(grid_x, grid_y, nt);
            gdm_low_local_gather_kernel<<<g,block,0,q.stream[d]>>>(p);
            ck(cudaGetLastError(), "gdm event low local");
        }
        q.fence(slot++);

        for (int d=0; d<q.ngpu; ++d) {
            ck(cudaSetDevice(d), "gdm event low cross device");
            dim3 g(grid_x, grid_y, nt);
            gdm_low_cross_gather_kernel<<<g,block,0,q.stream[d]>>>(p);
            ck(cudaGetLastError(), "gdm event low cross");
        }
        q.fence(slot++);
    }
}
