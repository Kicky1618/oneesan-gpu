#pragma once

// Reuse the proven LOW enqueue path, thread validation, environment parser and
// shared-memory sizing from the ordinary orbit-CTA graph.  HIGH is replaced by
// the global flat persistent pool below.
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_delta_direct_affine_rankformula_nometa4_abstract_graph.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_delta_direct_affine_rankformula_nometa4_abstract.cuh"

static inline int p10dc_orbitcta_flat_blocks_env(int fallback) {
    const char* s = std::getenv("BUCKET_ORBITCTA_FLAT_BLOCKS");
    if (!s || !*s) return fallback;
    const int v = std::atoi(s);
    if (v <= 0) {
        std::cerr << "BUCKET_ORBITCTA_FLAT_BLOCKS must be positive, got " << s << '\n';
        std::exit(831);
    }
    return v;
}

static inline int p10dc_orbitcta_flat_blocks_per_sm_env(int fallback) {
    const char* s = std::getenv("BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM");
    if (!s || !*s) return fallback;
    const int v = std::atoi(s);
    if (v <= 0) {
        std::cerr << "BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM must be positive, got " << s << '\n';
        std::exit(832);
    }
    return v;
}

static void p10dc_orbitcta_flat_report_high_occupancy(int threads) {
    const size_t smem = p10dc_orbitcta_high_smem_bytes(threads);
    int device = 0;
    ck(cudaGetDevice(&device), "flat orbitcta occupancy get device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "flat orbitcta occupancy device props");
    int fb = 0, rb = 0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &fb, bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_kernel,
           threads, smem),
       "flat orbitcta forward occupancy");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &rb, bucket_reverse_high_pattern10_depthcode_orbitcta_flat_kernel,
           threads, smem),
       "flat orbitcta reverse occupancy");
    cudaFuncAttributes fa{}, ra{};
    ck(cudaFuncGetAttributes(
           &fa, bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_kernel),
       "flat orbitcta forward attributes");
    ck(cudaFuncGetAttributes(
           &ra, bucket_reverse_high_pattern10_depthcode_orbitcta_flat_kernel),
       "flat orbitcta reverse attributes");
    const int cap = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    const int fw = fb * (threads / prop.warpSize);
    const int rw = rb * (threads / prop.warpSize);
    std::cerr << "rankformula_orbitcta_flat_occupancy device=" << device
              << " sms=" << prop.multiProcessorCount
              << " threads=" << threads
              << " dynamic_smem_bytes=" << smem
              << " cpasync_pair=" << P10DC_RANKFORMULA_CPASYNC_PAIR
              << " forward_regs=" << fa.numRegs
              << " reverse_regs=" << ra.numRegs
              << " forward_blocks_per_sm=" << fb
              << " reverse_blocks_per_sm=" << rb
              << " forward_warps_per_sm=" << fw
              << " reverse_warps_per_sm=" << rw
              << " warp_cap=" << cap
              << " forward_warp_occupancy_pct="
              << (cap ? 100.0 * double(fw) / double(cap) : 0.0)
              << " reverse_warp_occupancy_pct="
              << (cap ? 100.0 * double(rw) / double(cap) : 0.0)
              << '\n';
}

static void bucket_enqueue_high_orbit_closure_pattern10_depthcode_orbitcta_flat_rankformula_nometa4_abstract(
    cudaStream_t stream, int threads, int blocks
) {
    p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
    const dim3 block(unsigned(threads));
    const dim3 grid(unsigned(blocks));
    const size_t smem = p10dc_orbitcta_high_smem_bytes(threads);
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_kernel<<<grid, block, smem, stream>>>(p);
        ck(cudaGetLastError(), "bucket high flat orbitcta rankformula stream");
    }
}

static void bucket_enqueue_reverse_high_pattern10_depthcode_orbitcta_flat_rankformula_nometa4_abstract(
    cudaStream_t stream, int threads, int blocks
) {
    p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
    const dim3 block(unsigned(threads));
    const dim3 grid(unsigned(blocks));
    const size_t smem = p10dc_orbitcta_high_smem_bytes(threads);
    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        bucket_reverse_high_pattern10_depthcode_orbitcta_flat_kernel<<<grid, block, smem, stream>>>(p);
        ck(cudaGetLastError(), "bucket reverse high flat orbitcta rankformula stream");
    }
}

struct BucketPattern10DepthCodeFlatOrbitCtaDirectAffineRankFormulaNometa4AbstractGraphs {
    cudaStream_t stream = nullptr;
    std::array<cudaGraphExec_t, BKOC_GRAPH_COUNT> exec{};

    template<class F>
    void capture_one(BucketOnePassGraphKind kind, F&& enqueue, const char* what) {
        cudaGraph_t graph = nullptr;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), what);
        enqueue();
        ck(cudaStreamEndCapture(stream, &graph), what);
        if (!graph) {
            std::cerr << "flat orbitcta graph capture returned null graph " << what << '\n';
            std::exit(833);
        }
        ck(cudaGraphInstantiate(&exec[size_t(kind)], graph, nullptr, nullptr, 0), what);
        ck(cudaGraphDestroy(graph), what);
    }

    void init(const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8) {
        (void)gx; (void)gy;
        p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);
        p10dc_install_rankformula_abstract_lut();
        p10dc_orbitcta_flat_report_high_occupancy(threads);

        int device = 0;
        ck(cudaGetDevice(&device), "flat orbitcta grid get device");
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device), "flat orbitcta grid device props");
        const int low_gx = p10dc_rankformula_grid_env("BUCKET_LOW_GRID_X", 16);
        const int low_gy = p10dc_rankformula_grid_env("BUCKET_LOW_GRID_Y", 8);
        const int per_sm = p10dc_orbitcta_flat_blocks_per_sm_env(8);
        const int auto_blocks = std::max(1, prop.multiProcessorCount * per_sm);
        const int flat_blocks = p10dc_orbitcta_flat_blocks_env(auto_blocks);

        std::cerr << "rankformula_orbitcta_flat_grid device=" << device
                  << " sms=" << prop.multiProcessorCount
                  << " threads=" << threads
                  << " low_gx=" << low_gx << " low_gy=" << low_gy
                  << " flat_blocks=" << flat_blocks
                  << " blocks_per_sm_request=" << per_sm
                  << " scheduler=persistent_global_orbit_pool"
                  << " bid_binary_search=1"
                  << " high_grid_y=1 high_grid_z=1"
                  << " context_smem_bytes=" << sizeof(P10DCDirectHighResolvedCtx)
                  << " launch_smem_bytes=" << p10dc_orbitcta_high_smem_bytes(threads)
                  << '\n';

        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
           "flat orbitcta graph stream");
        capture_one(BKOC_GRAPH_FORWARD_LOW, [&] {
            bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(
                layout, stream, threads, low_gx, low_gy);
        }, "capture flat orbitcta forward LOW");
        capture_one(BKOC_GRAPH_FORWARD_HIGH, [&] {
            bucket_enqueue_high_orbit_closure_pattern10_depthcode_orbitcta_flat_rankformula_nometa4_abstract(
                stream, threads, flat_blocks);
        }, "capture flat orbitcta forward HIGH");
        capture_one(BKOC_GRAPH_REVERSE_LOW, [&] {
            bucket_enqueue_reverse_low_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(
                layout, stream, threads, low_gx, low_gy);
        }, "capture flat orbitcta reverse LOW");
        capture_one(BKOC_GRAPH_REVERSE_HIGH, [&] {
            bucket_enqueue_reverse_high_pattern10_depthcode_orbitcta_flat_rankformula_nometa4_abstract(
                stream, threads, flat_blocks);
        }, "capture flat orbitcta reverse HIGH");
    }

    void launch(BucketOnePassGraphKind kind) {
        ck(cudaGraphLaunch(exec[size_t(kind)], stream), "flat orbitcta graph launch");
    }
    void synchronize() {
        if (stream) ck(cudaStreamSynchronize(stream), "flat orbitcta graph sync");
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

static void bucket_pattern10_depthcode_flat_orbitcta_rankformula_nometa4_abstract_graph_sync_devices(
    std::array<BucketPattern10DepthCodeFlatOrbitCtaDirectAffineRankFormulaNometa4AbstractGraphs,
               BUCKET_NGPU>& graphs,
    int ngpu
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "flat orbitcta graph sync set");
        graphs[g].synchronize();
    }
}
