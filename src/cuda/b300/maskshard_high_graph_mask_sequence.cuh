#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#ifndef MASKSHARD_HIGH_GRAPH_MASK_SEQUENCE
#error "HIGH graph mask sequence requires MASKSHARD_HIGH_GRAPH_MASK_SEQUENCE"
#endif
#ifndef MASKSHARD_HIGH_CUDA_GRAPH
#error "HIGH graph mask sequence requires HIGH CUDA Graph"
#endif
#ifndef MASKSHARD_HIGH_PERTHREAD_STREAM
#error "HIGH graph mask sequence requires per-thread HIGH stream"
#endif
#ifndef MASKSHARD_HIGH_DEAD_SYMBOL_COPIES
#error "HIGH graph mask sequence layers on HIGH config-copy filtering"
#endif
#ifndef MASKSHARD_HIGH_CAP_LPT_SCHEDULE
#error "HIGH graph mask sequence requires cap-aware HIGH scheduling"
#endif
static_assert(LOW_LUT_K <= 16,
              "HIGH graph mask sequence stores LOW masks as uint16");

struct MaskShardHighGraphMaskRow {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    std::uint32_t cursor = 0;
};

__device__ __constant__ const std::uint16_t* D_MS_HIGH_GRAPH_MASK_SEQUENCE;
__device__ MaskShardHighGraphMaskRow D_MS_HIGH_GRAPH_MASK_ROW;
__device__ std::uint32_t D_MS_HIGH_GRAPH_MASK_STAGE;

__global__ void maskshard_high_graph_next_mask_kernel() {
    if (blockIdx.x || threadIdx.x) return;
    const std::uint32_t q = D_MS_HIGH_GRAPH_MASK_ROW.cursor++;
    if (D_MS_HIGH_GRAPH_MASK_ROW.begin + q >= D_MS_HIGH_GRAPH_MASK_ROW.end) {
        asm("trap;");
        return;
    }
    D_MS_HIGH_GRAPH_MASK_STAGE =
        std::uint32_t(D_MS_HIGH_GRAPH_MASK_SEQUENCE[
            D_MS_HIGH_GRAPH_MASK_ROW.begin + q]);
}

struct MaskShardHighGraphMaskSequenceRuntime {
    static constexpr int CAPS = TARGET_W / 2;
    std::array<std::uint16_t*, 8> d_sequence{};
    std::array<std::array<std::uint32_t, CAPS + 1>, 8> cap_off{};
    std::array<void*, 8> stage_ptr{};
    std::array<bool, 8> installed{};
    std::uint64_t total_entries = 0;

    template<class State>
    void install(const State& state) {
        if (!state.jobs || state.ngpu < 1 || state.ngpu > 8
            || state.by_cap.size() != std::size_t(CAPS)) {
            std::cerr << "HIGH graph mask sequence invalid schedule state\n";
            std::exit(418);
        }
        if (total_entries) return;

        const auto& jobs = *state.jobs;
        std::uint64_t sum_entries = 0;
        for (int d = 0; d < state.ngpu; ++d) {
            std::vector<std::uint16_t> flat;
            for (int cap = 1; cap <= CAPS; ++cap) {
                cap_off[std::size_t(d)][std::size_t(cap - 1)] =
                    std::uint32_t(flat.size());
                const auto& js = state.by_cap[std::size_t(cap - 1)]
                                    .jobs_by_gpu[std::size_t(d)];
                for (std::size_t q : js) {
                    if (q >= jobs.size()
                        || jobs[q].low_mask >= (1u << LOW_LUT_K)) {
                        std::cerr << "HIGH graph mask sequence invalid job q="
                                  << q << " gpu=" << d << " cap=" << cap << '\n';
                        std::exit(419);
                    }
                    flat.push_back(std::uint16_t(jobs[q].low_mask));
                }
            }
            cap_off[std::size_t(d)][CAPS] = std::uint32_t(flat.size());
            sum_entries += flat.size();

            ck(cudaSetDevice(d), "HIGH graph mask sequence set device");
            if (!flat.empty()) {
                ck(cudaMalloc(&d_sequence[std::size_t(d)],
                              flat.size() * sizeof(std::uint16_t)),
                   "HIGH graph mask sequence alloc");
                ck(cudaMemcpy(d_sequence[std::size_t(d)], flat.data(),
                              flat.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "HIGH graph mask sequence copy");
            }
            const std::uint16_t* ptr = d_sequence[std::size_t(d)];
            ck(cudaMemcpyToSymbol(D_MS_HIGH_GRAPH_MASK_SEQUENCE,
                                  &ptr, sizeof(ptr)),
               "HIGH graph mask sequence ptr");
            ck(cudaGetSymbolAddress(&stage_ptr[std::size_t(d)],
                                    D_MS_HIGH_GRAPH_MASK_STAGE),
               "HIGH graph mask stage address");
            installed[std::size_t(d)] = true;
        }

        const std::uint64_t expected =
            std::uint64_t(1u << LOW_LUT_K) * std::uint64_t(CAPS);
        if (sum_entries != expected) {
            std::cerr << "HIGH graph mask sequence coverage mismatch got="
                      << sum_entries << " expected=" << expected << '\n';
            std::exit(420);
        }
        total_entries = sum_entries;
        std::cerr << "fullorbit-batch HIGH graph mask sequence entries="
                  << total_entries
                  << " bytes=" << total_entries * sizeof(std::uint16_t)
                  << " host_mask_copies_per_residue_old="
                  << std::uint64_t(1u << LOW_LUT_K) * TARGET_W
                  << " row_resets_per_residue="
                  << std::uint64_t(state.ngpu) * TARGET_W << '\n';
    }

    void set_row_current_device(int zero_based_row) {
        int dev = -1;
        ck(cudaGetDevice(&dev), "HIGH graph mask row get device");
        if (dev < 0 || dev >= 8 || !installed[std::size_t(dev)]) {
            std::cerr << "HIGH graph mask row used before install dev=" << dev << '\n';
            std::exit(421);
        }
        const int cap = std::min(zero_based_row + 1, CAPS);
        MaskShardHighGraphMaskRow row{};
        row.begin = cap_off[std::size_t(dev)][std::size_t(cap - 1)];
        row.end = cap_off[std::size_t(dev)][std::size_t(cap)];
        ck(cudaMemcpyToSymbol(D_MS_HIGH_GRAPH_MASK_ROW, &row, sizeof(row)),
           "HIGH graph mask row reset");
    }
};

static MaskShardHighGraphMaskSequenceRuntime&
maskshard_high_graph_mask_sequence_runtime() {
    static MaskShardHighGraphMaskSequenceRuntime runtime;
    return runtime;
}

template<class State>
static void maskshard_high_graph_mask_sequence_install(const State& state) {
    maskshard_high_graph_mask_sequence_runtime().install(state);
}

static void maskshard_high_graph_mask_sequence_set_row(int zero_based_row) {
    maskshard_high_graph_mask_sequence_runtime().set_row_current_device(
        zero_based_row);
}

static cudaError_t maskshard_high_graph_mask_sequence_begin_capture(
    cudaStream_t stream, cudaStreamCaptureMode mode
) {
    cudaError_t e = (cudaStreamBeginCapture)(stream, mode);
    if (e != cudaSuccess) return e;

    // LOW graph capture uses a different stream. Inject only into the dedicated
    // HIGH per-thread stream so every captured HIGH graph starts by loading the
    // next resident LOW mask and copying it device-to-device into D_F_MASK.
    if (stream != maskshard_high_execution_stream()) return cudaSuccess;

    int dev = -1;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    auto& runtime = maskshard_high_graph_mask_sequence_runtime();
    if (dev < 0 || dev >= 8 || !runtime.installed[std::size_t(dev)]
        || !runtime.stage_ptr[std::size_t(dev)])
        return cudaErrorInvalidDevice;

    maskshard_high_graph_next_mask_kernel<<<1, 1, 0, stream>>>();
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return cudaMemcpyToSymbolAsync(
        D_F_MASK,
        runtime.stage_ptr[std::size_t(dev)],
        sizeof(std::uint32_t),
        0,
        cudaMemcpyDeviceToDevice,
        stream);
}

// v0.58's wrapper remains the single implementation for all normal symbol
// copies. Suppress only worker-side D_F_MASK: v0.79 restores it inside the graph
// from the resident sequence. Main-thread setup and LOW config still use the
// original constant symbol path.
template<class Symbol>
static cudaError_t maskshard_high_graph_mask_sequence_memcpy_to_symbol(
    const char* name,
    const Symbol& symbol,
    const void* src,
    std::size_t count,
    std::size_t offset = 0,
    cudaMemcpyKind kind = cudaMemcpyHostToDevice
) {
    const bool worker =
        std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD;
    if (worker && std::strcmp(name, "D_F_MASK") == 0)
        return cudaSuccess;
    return maskshard_high_filtered_memcpy_to_symbol(
        name, symbol, src, count, offset, kind);
}

#undef cudaMemcpyToSymbol
#define cudaMemcpyToSymbol(symbol, src, count, ...) \
    maskshard_high_graph_mask_sequence_memcpy_to_symbol( \
        #symbol, symbol, src, count, ##__VA_ARGS__)

#define cudaStreamBeginCapture(stream, mode) \
    maskshard_high_graph_mask_sequence_begin_capture((stream), (mode))
