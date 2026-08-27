#pragma once

#include "ramstream32_bucket_orbit_closure_graph.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpctx_stream.cuh"

struct BucketPattern10DepthCodeWarpCtxGraphs {
    cudaStream_t stream = nullptr;
    std::array<cudaGraphExec_t, BKOC_GRAPH_COUNT> exec{};

    template<class F>
    void capture_one(BucketOnePassGraphKind kind, F&& enqueue, const char* what) {
        cudaGraph_t graph = nullptr;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), what);
        enqueue();
        ck(cudaStreamEndCapture(stream, &graph), what);
        if (!graph) {
            std::cerr << "pattern10 depthcode warpctx graph capture returned null graph "
                      << what << '\n';
            std::exit(603);
        }
        ck(cudaGraphInstantiate(&exec[size_t(kind)], graph, nullptr, nullptr, 0), what);
        ck(cudaGraphDestroy(graph), what);
    }

    void init(const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8) {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
           "pattern10 depthcode warpctx graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW, [&] {
            bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpctx(
                layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode warpctx forward LOW graph");
        capture_one(BKOC_GRAPH_FORWARD_HIGH, [&] {
            bucket_enqueue_high_orbit_closure_pattern10_depthcode_warpctx(
                layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode warpctx forward HIGH graph");
        capture_one(BKOC_GRAPH_REVERSE_LOW, [&] {
            bucket_enqueue_reverse_low_pattern10_depthcode_warpctx(
                layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode warpctx reverse LOW graph");
        capture_one(BKOC_GRAPH_REVERSE_HIGH, [&] {
            bucket_enqueue_reverse_high_pattern10_depthcode_warpctx(
                layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode warpctx reverse HIGH graph");
    }

    void launch(BucketOnePassGraphKind kind) {
        ck(cudaGraphLaunch(exec[size_t(kind)], stream),
           "pattern10 depthcode warpctx graph launch");
    }
    void synchronize() {
        if (stream) ck(cudaStreamSynchronize(stream),
                       "pattern10 depthcode warpctx graph sync");
    }
    void release() {
        for (auto& e : exec) {
            if (e) cudaGraphExecDestroy(e);
            e = nullptr;
        }
        if (stream) cudaStreamDestroy(stream);
        stream = nullptr;
    }
};

static void bucket_pattern10_depthcode_warpctx_graph_sync_devices(
    std::array<BucketPattern10DepthCodeWarpCtxGraphs, BUCKET_NGPU>& graphs,
    int ngpu
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pattern10 depthcode warpctx graph sync set");
        graphs[g].synchronize();
    }
}
