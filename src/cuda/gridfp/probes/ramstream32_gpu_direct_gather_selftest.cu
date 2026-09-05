#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_sparse.hpp"
#include "../ramstream32_cpu_high.hpp"
#include "../ramstream32_cpu_high_direct.hpp"
#include "../ramstream32_gpu_direct.cuh"
#include "../ramstream32_gpu_direct_gather.cuh"

static void gdg_enum_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    gdg_enum_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) gdg_enum_rec(pos - 1, h - 1, mset(m, pos, R), out);
    gdg_enum_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> gdg_enum_states(int width) {
    std::vector<MateID> out;
    gdg_enum_rec(width - 1, 1, 0, out);
    return out;
}
static Count gdg_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static std::pair<std::vector<Count>, std::vector<Count>> gdg_reference_window(
    int W, int p_hi, int p_lo, Count mod,
    const std::vector<MateID>& main_states,
    const std::vector<MateID>& block_states,
    const std::unordered_map<MateID, size_t>& mi,
    const std::unordered_map<MateID, size_t>& di,
    const std::vector<Count>& init_m,
    const std::vector<Count>& init_d
) {
    std::vector<Count> rm = init_m, rd = init_d;
    for (int p = p_hi; p >= p_lo; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nd(rd.size(), 0);
        for (size_t i = 0; i < main_states.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(main_states[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = di.find(z.mate); if (it == di.end()) std::exit(3);
                nd[it->second] = gdg_add(nd[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate); if (it == mi.end()) std::exit(4);
                nm[it->second] = gdg_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < block_states.size(); ++i) {
            Count c = rd[i];
            MateID z = oneesan::gridfp::blocked_exclude(block_states[i], p);
            auto it = mi.find(z); if (it == mi.end()) std::exit(5);
            nm[it->second] = gdg_add(nm[it->second], c, mod);
        }
        rm.swap(nm); rd.swap(nd);
    }
    return {std::move(rm), std::move(rd)};
}

static void gdg_fill(
    RamCounts& ma, RamCounts& ba,
    const std::vector<MateID>& ms, const std::vector<MateID>& bs,
    const std::vector<Count>& mv, const std::vector<Count>& bv,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    std::memset(ma.ptr, 0, ma.bytes);
    std::memset(ba.ptr, 0, ba.bytes);
    for (size_t i = 0; i < ms.size(); ++i)
        ma.ptr[storage_rank_main_host(ms[i], storage, layout)] = mv[i];
    for (size_t i = 0; i < bs.size(); ++i)
        ba.ptr[storage_rank_block_host(bs[i], storage, layout)] = bv[i];
}

static bool gdg_compare(
    const char* tag, const RamCounts& ma, const RamCounts& ba,
    const std::vector<MateID>& ms, const std::vector<MateID>& bs,
    const std::vector<Count>& rm, const std::vector<Count>& rb,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < ms.size(); ++i) {
        Count got = ma.ptr[storage_rank_main_host(ms[i], storage, layout)];
        if (got != rm[i]) {
            std::cerr << "FAIL " << tag << " main i=" << i
                      << " got=" << got << " want=" << rm[i] << '\n';
            return false;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        Count got = ba.ptr[storage_rank_block_host(bs[i], storage, layout)];
        if (got != rb[i]) {
            std::cerr << "FAIL " << tag << " block i=" << i
                      << " got=" << got << " want=" << rb[i] << '\n';
            return false;
        }
    }
    return true;
}

static void gdg_to_device(Count* dm, Count* db, const RamCounts& ma, const RamCounts& ba) {
    ck(cudaMemcpy(dm, ma.ptr, ma.bytes, cudaMemcpyHostToDevice), "gdg main H2D");
    ck(cudaMemcpy(db, ba.ptr, ba.bytes, cudaMemcpyHostToDevice), "gdg block H2D");
}
static void gdg_from_device(RamCounts& ma, RamCounts& ba, const Count* dm, const Count* db) {
    ck(cudaMemcpy(ma.ptr, dm, ma.bytes, cudaMemcpyDeviceToHost), "gdg main D2H");
    ck(cudaMemcpy(ba.ptr, db, ba.bytes, cudaMemcpyDeviceToHost), "gdg block D2H");
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "gather selftest intentionally uses small width");

    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "gpu-direct-gather-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "gdg set device");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectCrossHost cross = build_gpu_direct_cross(storage);
    GpuDirectGatherHost gather = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect);

    auto ms = gdg_enum_states(W);
    auto bs = gdg_enum_states(W - 1);
    if (ms.size() != layout.main_size || bs.size() != layout.block_size) return 2;
    std::unordered_map<MateID,size_t> mi, di;
    for (size_t i=0;i<ms.size();++i) mi.emplace(ms[i],i);
    for (size_t i=0;i<bs.size();++i) di.emplace(bs[i],i);

    std::vector<Count> init_m(ms.size()), init_b(bs.size());
    std::mt19937_64 rng(1618);
    for (auto& x:init_m) x=Count(rng()%mod);
    for (auto& x:init_b) x=Count(rng()%mod);

    auto [low_m, low_b] = gdg_reference_window(
        W, LOW_LUT_K, 1, mod, ms, bs, mi, di, init_m, init_b);
    auto [high_m, high_b] = gdg_reference_window(
        W, W-1, LOW_LUT_K+1, mod, ms, bs, mi, di, init_m, init_b);
    auto [row_m, row_b] = gdg_reference_window(
        W, LOW_LUT_K, 1, mod, ms, bs, mi, di, high_m, high_b);

    RamCounts ma, ba;
    ma.alloc(layout.main_size, "gdg main");
    ba.alloc(layout.block_size, "gdg block");
    Count *dm=nullptr,*db=nullptr;
    ck(cudaMalloc(&dm, ma.bytes), "gdg alloc main");
    ck(cudaMalloc(&db, ba.bytes), "gdg alloc block");
    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "gdg modulus");

    GpuDirectDeviceTables base;
    base.install(storage, layout, lowdesc, loworbit, highdirect, cross);
    GpuDirectGatherDeviceTables gt;
    gt.install(gather);
    gpu_direct_gather_drop_redundant(base);

    gdg_fill(ma, ba, ms, bs, init_m, init_b, storage, layout);
    gdg_to_device(dm, db, ma, ba);
    gpu_direct_run_low_gather(dm, db, layout, 256, 4, 4);
    gdg_from_device(ma, ba, dm, db);
    if (!gdg_compare("gather-low", ma, ba, ms, bs, low_m, low_b, storage, layout)) return 10;

    gdg_fill(ma, ba, ms, bs, init_m, init_b, storage, layout);
    gdg_to_device(dm, db, ma, ba);
    gpu_direct_run_high_gather(dm, db, layout, 256, 4, 4);
    gdg_from_device(ma, ba, dm, db);
    if (!gdg_compare("gather-high", ma, ba, ms, bs, high_m, high_b, storage, layout)) return 11;

    gdg_fill(ma, ba, ms, bs, init_m, init_b, storage, layout);
    gdg_to_device(dm, db, ma, ba);
    gpu_direct_run_high_gather(dm, db, layout, 256, 4, 4);
    gpu_direct_run_low_gather(dm, db, layout, 256, 4, 4);
    gdg_from_device(ma, ba, dm, db);
    if (!gdg_compare("gather-row", ma, ba, ms, bs, row_m, row_b, storage, layout)) return 12;

    std::cout << "gpu-direct-gather-selftest OK W=" << W
              << " main=" << ms.size() << " block=" << bs.size()
              << " gather_mib=" << double(gather.bytes())/double(1<<20)
              << " low_edges=" << gather.low_src.size()
              << " low_cross=" << gather.low_cross.size()
              << " low_max_indegree=" << gather.low_max_indegree
              << " high_edges=" << gather.high_src.size()
              << " high_max_indegree=" << gather.high_max_indegree
              << " low_launches=" << (3*LOW_LUT_K)
              << " high_launches=" << (3*HIGH_LUT_K)
              << " scratch_bytes=0\n";

    gt.release();
    base.release();
    cudaFree(dm); cudaFree(db);
    ma.release(); ba.release();
    return 0;
}
