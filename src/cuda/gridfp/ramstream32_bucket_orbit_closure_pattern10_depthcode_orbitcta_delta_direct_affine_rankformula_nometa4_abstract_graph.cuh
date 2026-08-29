#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_delta_direct_affine_rankformula_nometa4_abstract.cuh"

static inline int p10dc_orbitcta_grid_env(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    const int v = std::atoi(s);
    if (v <= 0) {
        std::cerr << name << " must be positive, got " << s << '\n';
        std::exit(811);
    }
    return v;
}

static void bucket_enqueue_high_orbit_closure_pattern10_depthcode_orbitcta_rankformula_nometa4_abstract(
    const StorageLayout& layout, cudaStream_t stream, int threads, int gy
) {
    p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
    const dim3 block(unsigned(threads));
    const dim3 grid(1u, unsigned(gy), unsigned(layout.main_blocks.size()));
    const size_t smem = sizeof(P10DCDirectHighResolvedCtx);
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        bucket_high_orbit_closure_pattern10_depthcode_orbitcta_kernel<<<grid, block, smem, stream>>>(p);
        ck(cudaGetLastError(), "bucket high orbitcta rankformula-nometa4-abstract stream");
    }
}

static void bucket_enqueue_reverse_high_pattern10_depthcode_orbitcta_rankformula_nometa4_abstract(
    const StorageLayout& layout, cudaStream_t stream, int threads, int gy
) {
    p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
    const dim3 block(unsigned(threads));
    const dim3 grid(1u, unsigned(gy), unsigned(layout.main_blocks.size()));
    const size_t smem = sizeof(P10DCDirectHighResolvedCtx);
    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        bucket_reverse_high_pattern10_depthcode_orbitcta_kernel<<<grid, block, smem, stream>>>(p);
        ck(cudaGetLastError(), "bucket reverse high orbitcta rankformula-nometa4-abstract stream");
    }
}

struct BucketPattern10DepthCodeOrbitCtaDirectAffineRankFormulaNometa4AbstractGraphs {
    cudaStream_t stream = nullptr;
    std::array<cudaGraphExec_t, BKOC_GRAPH_COUNT> exec{};

    template<class F>
    void capture_one(BucketOnePassGraphKind kind, F&& enqueue, const char* what) {
        cudaGraph_t graph = nullptr;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), what);
        enqueue();
        ck(cudaStreamEndCapture(stream, &graph), what);
        if (!graph) {
            std::cerr << "orbitcta graph capture returned null graph " << what << '\n';
            std::exit(812);
        }
        ck(cudaGraphInstantiate(&exec[size_t(kind)], graph, nullptr, nullptr, 0), what);
        ck(cudaGraphDestroy(graph), what);
    }

    void init(const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8) {
        (void)gx;
        p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
        p10dc_install_rankformula_abstract_lut();
        const int low_gx = p10dc_rankformula_grid_env("BUCKET_LOW_GRID_X", 16);
        const int low_gy = p10dc_rankformula_grid_env("BUCKET_LOW_GRID_Y", 8);
        const int orbit_gy = p10dc_orbitcta_grid_env(
            "BUCKET_ORBITCTA_GRID_Y",
            p10dc_rankformula_grid_env("BUCKET_HIGH_GRID_Y", std::max(gy, 64)));
        std::cerr << "rankformula_orbitcta_grid threads=" << threads
                  << " low_gx=" << low_gx << " low_gy=" << low_gy
                  << " high_gx=1 high_gy=" << orbit_gy
                  << " orbit_context_builds_per_orbit=1"
                  << " context_smem_bytes=" << sizeof(P10DCDirectHighResolvedCtx) << '\n';
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "orbitcta graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW, [&] {
            bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(
                layout, stream, threads, low_gx, low_gy);
        }, "capture orbitcta forward LOW");
        capture_one(BKOC_GRAPH_FORWARD_HIGH, [&] {
            bucket_enqueue_high_orbit_closure_pattern10_depthcode_orbitcta_rankformula_nometa4_abstract(
                layout, stream, threads, orbit_gy);
        }, "capture orbitcta forward HIGH");
        capture_one(BKOC_GRAPH_REVERSE_LOW, [&] {
            bucket_enqueue_reverse_low_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(
                layout, stream, threads, low_gx, low_gy);
        }, "capture orbitcta reverse LOW");
        capture_one(BKOC_GRAPH_REVERSE_HIGH, [&] {
            bucket_enqueue_reverse_high_pattern10_depthcode_orbitcta_rankformula_nometa4_abstract(
                layout, stream, threads, orbit_gy);
        }, "capture orbitcta reverse HIGH");
    }

    void launch(BucketOnePassGraphKind kind) {
        ck(cudaGraphLaunch(exec[size_t(kind)], stream), "orbitcta graph launch");
    }
    void synchronize() {
        if (stream) ck(cudaStreamSynchronize(stream), "orbitcta graph sync");
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

static void bucket_pattern10_depthcode_orbitcta_rankformula_nometa4_abstract_graph_sync_devices(
    std::array<BucketPattern10DepthCodeOrbitCtaDirectAffineRankFormulaNometa4AbstractGraphs, BUCKET_NGPU>& graphs,
    int ngpu
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "orbitcta graph sync set");
        graphs[g].synchronize();
    }
}
