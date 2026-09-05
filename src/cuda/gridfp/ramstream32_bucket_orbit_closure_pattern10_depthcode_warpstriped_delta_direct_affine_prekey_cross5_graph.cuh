#pragma once

#include "ramstream32_bucket_orbit_closure_graph.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_cross5_stream.cuh"

struct BucketPattern10DepthCodeWarpStripedDeltaDirectAffinePrekeyCross5Graphs {
    cudaStream_t stream = nullptr;
    std::array<cudaGraphExec_t, BKOC_GRAPH_COUNT> exec{};

    template<class F>
    void capture_one(BucketOnePassGraphKind kind, F&& enqueue, const char* what) {
        cudaGraph_t graph = nullptr;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), what);
        enqueue();
        ck(cudaStreamEndCapture(stream, &graph), what);
        if (!graph) {
            std::cerr << "pattern10 depthcode affine-prekey-cross5 graph capture returned null graph " << what << '\n';
            std::exit(621);
        }
        ck(cudaGraphInstantiate(&exec[size_t(kind)], graph, nullptr, nullptr, 0), what);
        ck(cudaGraphDestroy(graph), what);
    }

    void init(const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8) {
        p10dc_warpstriped_delta_direct_affine_prekey_cross5_require_threads(threads);
        p10dc_install_cross5_lut();
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
           "pattern10 depthcode affine-prekey-cross5 graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW, [&] {
            bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_cross5(layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode affine-prekey-cross5 forward LOW graph");
        capture_one(BKOC_GRAPH_FORWARD_HIGH, [&] {
            bucket_enqueue_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_cross5(layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode affine-prekey-cross5 forward HIGH graph");
        capture_one(BKOC_GRAPH_REVERSE_LOW, [&] {
            bucket_enqueue_reverse_low_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_cross5(layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode affine-prekey-cross5 reverse LOW graph");
        capture_one(BKOC_GRAPH_REVERSE_HIGH, [&] {
            bucket_enqueue_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_cross5(layout, stream, threads, gx, gy);
        }, "capture pattern10 depthcode affine-prekey-cross5 reverse HIGH graph");
    }

    void launch(BucketOnePassGraphKind kind) {
        ck(cudaGraphLaunch(exec[size_t(kind)], stream), "pattern10 depthcode affine-prekey-cross5 graph launch");
    }
    void synchronize() {
        if (stream) ck(cudaStreamSynchronize(stream), "pattern10 depthcode affine-prekey-cross5 graph sync");
    }
    void release() {
        for (auto& e : exec) { if (e) cudaGraphExecDestroy(e); e = nullptr; }
        if (stream) cudaStreamDestroy(stream);
        stream = nullptr;
    }
};

static void bucket_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_cross5_graph_sync_devices(
    std::array<BucketPattern10DepthCodeWarpStripedDeltaDirectAffinePrekeyCross5Graphs, BUCKET_NGPU>& graphs,
    int ngpu
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pattern10 depthcode affine-prekey-cross5 graph sync set");
        graphs[g].synchronize();
    }
}
