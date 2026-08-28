#include <cuda_runtime.h>

#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_factorized_probe_main_unused
#include "../../cpp/probes/gridfp_reduced_production_factorized_probe.cpp"
#pragma pop_macro("main")

#include "gridfp_reduced_production_device.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace oneesan::gridfp;
using namespace oneesan::gridfp::reducedprod;

namespace {

static constexpr int WARP = 32;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int THREADS = WARP * WARPS_PER_BLOCK;
static constexpr int MAX_PAIRS = 32;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(90);
    }
}

void build_motzkin_table(Rank64 out[RP_MAX_W + 1][RP_MAX_W + 2]) {
    for (int r = 0; r <= RP_MAX_W; ++r)
        for (int h = 0; h <= RP_MAX_W + 1; ++h) out[r][h] = 0;
    out[0][0] = 1;
    for (int rem = 1; rem <= RP_MAX_W; ++rem) {
        for (int h = 0; h <= RP_MAX_W; ++h) {
            Rank64 z = out[rem - 1][h];
            if (h > 0) z += out[rem - 1][h - 1];
            z += out[rem - 1][h + 1];
            out[rem][h] = z;
        }
    }
}

void install_tables(const ProductionFactorTables& t) {
    Rank64 choose[RP_MAX_W + 1][RP_MAX_W + 1]{};
    Rank64 primitive[RP_MAX_W + 1][RP_MAX_W + 2]{};
    Rank64 motzkin[RP_MAX_W + 1][RP_MAX_W + 2]{};
    Rank64 sector_offset[RP_MAX_SECTORS + 1]{};
    Rank64 sector_main[RP_MAX_SECTORS]{};
    Rank64 sector_primitive[RP_MAX_SECTORS]{};

    build_motzkin_table(motzkin);
    for (int n = 0; n <= t.W; ++n)
        for (int k = 0; k <= t.W; ++k)
            choose[n][k] = t.choose[static_cast<std::size_t>(n)][static_cast<std::size_t>(k)];
    for (int rem = 0; rem <= t.W; ++rem)
        for (int h = 0; h <= t.W + 1; ++h)
            primitive[rem][h] = t.primitive[static_cast<std::size_t>(rem)][static_cast<std::size_t>(h)];
    for (std::size_t i = 0; i < t.sector_offset.size(); ++i) sector_offset[i] = t.sector_offset[i];
    for (std::size_t i = 0; i < t.sector_main.size(); ++i) {
        sector_main[i] = t.sector_main[i];
        sector_primitive[i] = t.sector_primitive[i];
    }

    ck(cudaMemcpyToSymbol(RP_CHOOSE, choose, sizeof(choose)), "copy choose");
    ck(cudaMemcpyToSymbol(RP_PRIMITIVE, primitive, sizeof(primitive)), "copy primitive");
    ck(cudaMemcpyToSymbol(RP_MOTZKIN, motzkin, sizeof(motzkin)), "copy motzkin");
    ck(cudaMemcpyToSymbol(RP_SECTOR_OFFSET, sector_offset, sizeof(sector_offset)), "copy sector offsets");
    ck(cudaMemcpyToSymbol(RP_SECTOR_MAIN, sector_main, sizeof(sector_main)), "copy sector main");
    ck(cudaMemcpyToSymbol(RP_SECTOR_PRIMITIVE, sector_primitive, sizeof(sector_primitive)), "copy sector primitive");
}

Rank64 motzkin_count(int len) {
    Rank64 table[RP_MAX_W + 1][RP_MAX_W + 2]{};
    build_motzkin_table(table);
    return table[len][1];
}

__device__ __forceinline__ void set_error(int* error, int code) {
    atomicCAS(error, 0, code);
}

__device__ __forceinline__ int find_key(const DeviceKey* a, int n, DeviceKey k) {
    for (int i = 0; i < n; ++i) if (key_equal(a[i], k)) return i;
    return -1;
}

__global__ void component_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    Rank64 raw_labels,
    Rank64 state_count,
    int W,
    int p,
    std::uint32_t mod,
    unsigned long long* component_count,
    int* error
) {
    __shared__ DeviceKey sh_src[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ DeviceKey sh_dst[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 label_rank = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);

    if (lane == 0) {
        sh_ns[warp] = 0;
        sh_nd[warp] = 0;
        if (label_rank < raw_labels) {
            const MateID label = motzkin_unrank_device(W - 1, label_rank);
            bool eligible = false;
            const DeviceKey seed = forward_component_seed(label, W, p, eligible);
            if (eligible) {
                atomicAdd(component_count, 1ULL);
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    DeviceTerm edge[RP_MAX_TERMS]{};
                    const int ne = reduced_step_forward(sh_src[warp][cursor++], W, p, edge);
                    if (ne < 0) {
                        set_error(error, 1);
                        break;
                    }
                    for (int ei = 0; ei < ne; ++ei) {
                        if (!edge[ei].coef) continue;
                        const DeviceKey d = edge[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 2);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;

                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_reduced_forward(d, W, p, pre);
                        if (np < 0) {
                            set_error(error, 3);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 4);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 5);
            }
        }
    }
    __syncwarp();

    const int ns = sh_ns[warp];
    const int nd = sh_nd[warp];
    if (lane < ns) {
        const Rank64 r = factor_rank_device(sh_src[warp][lane], W, p - 1);
        if (r >= state_count) {
            set_error(error, 6);
            sh_value[warp][lane] = 0;
        } else {
            sh_value[warp][lane] = input[r];
        }
    }
    __syncwarp();

    if (lane < nd) {
        DeviceTerm pre[RP_MAX_TERMS]{};
        const int np = inverse_reduced_forward(sh_dst[warp][lane], W, p, pre);
        if (np < 0) {
            set_error(error, 7);
            return;
        }
        long long acc = 0;
        for (int i = 0; i < np; ++i) {
            if (!pre[i].coef) continue;
            const int si = find_key(sh_src[warp], ns, pre[i].key);
            if (si < 0) {
                set_error(error, 8);
                continue;
            }
            acc += static_cast<long long>(pre[i].coef) * static_cast<long long>(sh_value[warp][si]);
        }
        long long z = acc % static_cast<long long>(mod);
        if (z < 0) z += mod;
        const Rank64 dr = factor_rank_device(sh_dst[warp][lane], W, p - 2);
        if (dr >= state_count) {
            set_error(error, 9);
            return;
        }
        const unsigned long long empty = ~0ULL;
        const unsigned long long prev = atomicCAS(owner + dr, empty, static_cast<unsigned long long>(label_rank));
        if (prev != empty && prev != static_cast<unsigned long long>(label_rank)) set_error(error, 10);
        output[dr] = static_cast<std::uint32_t>(z);
    }
}

void add_mod_signed(std::uint32_t& dst, std::uint32_t value, int coef, std::uint32_t mod) {
    long long z = static_cast<long long>(dst) + static_cast<long long>(coef) * value;
    z %= static_cast<long long>(mod);
    if (z < 0) z += mod;
    dst = static_cast<std::uint32_t>(z);
}

bool has_arg(int argc, char** argv, const char* needle) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == needle) return true;
    return false;
}

void run_position(int W, int p, const ProductionFactorTables& tables, std::uint32_t mod) {
    ProductionFactorCodec src(tables, p - 1);
    ProductionFactorCodec dst(tables, p - 2);
    const Rank64 states = tables.size();
    const Rank64 raw_labels = motzkin_count(W - 1);
    const Rank64 expected_components = raw_labels - motzkin_count(W - 3);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank64 r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>((1 + (r * 2654435761ULL) % (mod - 1ULL)) % mod);

    for (Rank64 s = 0; s < states; ++s) {
        const Key k = src.unrank(s);
        const std::uint32_t v = input[static_cast<std::size_t>(s)];
        for (const auto& [d, c] : reduced_step_basis(k, W, p, false))
            add_mod_signed(reference[static_cast<std::size_t>(dst.rank(d))], v, int(c), mod);
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_components = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "alloc output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "alloc owner");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "alloc components");
    ck(cudaMalloc(&d_error, sizeof(int)), "alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "clear owner");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "zero components");
    ck(cudaMemset(d_error, 0, sizeof(int)), "zero error");

    const unsigned blocks = static_cast<unsigned>((raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "event a");
    ck(cudaEventCreate(&b), "event b");
    ck(cudaEventRecord(a), "record a");
    component_kernel<<<blocks, THREADS>>>(
        d_input, d_output, d_owner, raw_labels, states, W, p, mod, d_components, d_error);
    ck(cudaGetLastError(), "launch component microprobe");
    ck(cudaEventRecord(b), "record b");
    ck(cudaEventSynchronize(b), "sync b");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long components = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy owner");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost), "copy components");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "copy error");

    if (error) {
        std::cerr << "FAIL device error=" << error << " W=" << W << " p=" << p << '\n';
        std::exit(91);
    }
    if (components != expected_components) {
        std::cerr << "FAIL component count got=" << components << " want=" << expected_components << '\n';
        std::exit(92);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max()) {
            std::cerr << "FAIL uncovered destination rank=" << r << '\n';
            std::exit(93);
        }
        if (output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL arithmetic W=" << W << " p=" << p << " rank=" << r
                      << " gpu=" << output[static_cast<std::size_t>(r)]
                      << " cpu=" << reference[static_cast<std::size_t>(r)] << '\n';
            std::exit(94);
        }
    }

    const double values_per_s = ms > 0 ? double(states) / (double(ms) * 1e-3) : 0.0;
    std::cout << "gridfp-reduced-component-microprobe"
              << " W=" << W << " p=" << p
              << " states=" << states
              << " raw_label_warps=" << raw_labels
              << " components=" << components
              << " skipped_warps=" << (raw_labels - components)
              << " kernel_ms=" << ms
              << " state_pairs_per_s=" << values_per_s
              << " source_load_once=1 destination_store_once=1"
              << " component_table_bytes=0 inverse_table_bytes=0"
              << " arithmetic=OK\n";

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_owner);
    cudaFree(d_components);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int requested_p = argc > 2 ? std::atoi(argv[2]) : -1;
    const std::uint32_t mod = argc > 3 ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > RP_MAX_W || mod < 3) return 2;
    if (requested_p >= 0 && (requested_p < 3 || requested_p > W - 1)) return 3;

    ProductionFactorTables tables(W);
    const Rank64 raw_labels = motzkin_count(W - 1);
    const Rank64 components = raw_labels - motzkin_count(W - 3);
    const Rank64 states = tables.size();
    if (plan_only) {
        std::cout << "gridfp-reduced-component-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " raw_label_warps=" << raw_labels
                  << " components=" << components
                  << " skipped_fraction=" << double(raw_labels - components) / double(raw_labels)
                  << " blocks_per_position=" << ((raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK)
                  << " threads=" << THREADS
                  << " max_pairs_storage=" << MAX_PAIRS
                  << " component_table_bytes=0 inverse_table_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 4;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "device count");
    if (visible < 1) return 5;
    ck(cudaSetDevice(0), "set device");
    install_tables(tables);

    if (requested_p >= 0) {
        run_position(W, requested_p, tables, mod);
    } else {
        for (int p = W - 1; p >= 3; --p) run_position(W, p, tables, mod);
    }
    std::cout << "ALL_OK gridfp_reduced_production_cuda_component=1 W=" << W << '\n';
    return 0;
}
