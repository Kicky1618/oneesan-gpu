#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_sparse.hpp"

static void enum_states_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    enum_states_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) enum_states_rec(pos - 1, h - 1, mset(m, pos, R), out);
    enum_states_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> enum_states(int width) {
    std::vector<MateID> out; enum_states_rec(width - 1, 1, 0, out); return out;
}
static inline Count ref_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}
static void fill_factor(
    RamCounts& main_auth, RamCounts& block_auth,
    const std::vector<MateID>& main_states, const std::vector<MateID>& block_states,
    const std::vector<Count>& main_values, const std::vector<Count>& block_values,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    std::memset(main_auth.ptr, 0, main_auth.bytes);
    std::memset(block_auth.ptr, 0, block_auth.bytes);
    for (size_t i = 0; i < main_states.size(); ++i)
        main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)] = main_values[i];
    for (size_t i = 0; i < block_states.size(); ++i)
        block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)] = block_values[i];
}
static bool compare_factor(
    const char* tag, const RamCounts& main_auth, const RamCounts& block_auth,
    const std::vector<MateID>& main_states, const std::vector<MateID>& block_states,
    const std::vector<Count>& rm, const std::vector<Count>& rd,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < main_states.size(); ++i) {
        Count got = main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)];
        if (got != rm[i]) { std::cerr << "FAIL " << tag << " main i=" << i << " got=" << got << " want=" << rm[i] << '\n'; return false; }
    }
    for (size_t i = 0; i < block_states.size(); ++i) {
        Count got = block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)];
        if (got != rd[i]) { std::cerr << "FAIL " << tag << " block i=" << i << " got=" << got << " want=" << rd[i] << '\n'; return false; }
    }
    return true;
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "selftest intentionally uses a small width");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(layout, lowdesc, orbit);

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size || block_states.size() != layout.block_size) return 2;

    std::unordered_map<MateID, size_t> mi, di;
    mi.reserve(main_states.size() * 2); di.reserve(block_states.size() * 2);
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "selftest main");
    block_auth.alloc(layout.block_size, "selftest block");
    std::vector<Count> init_m(main_states.size()), init_d(block_states.size());
    std::mt19937_64 rng(1618);
    for (auto& v : init_m) v = Count(rng() % mod);
    for (auto& v : init_d) v = Count(rng() % mod);
    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);

    std::vector<Count> rm = init_m, rd = init_d;
    for (int p = LOW_LUT_K; p >= 1; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nd(rd.size(), 0);
        for (size_t i = 0; i < main_states.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(main_states[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) { auto it = di.find(z.mate); if (it == di.end()) return 3; nd[it->second] = ref_add(nd[it->second], c, mod); }
            else { auto it = mi.find(z.mate); if (it == mi.end()) return 4; nm[it->second] = ref_add(nm[it->second], c, mod); }
        }
        for (size_t i = 0; i < block_states.size(); ++i) {
            Count c = rd[i]; MateID z = oneesan::gridfp::blocked_exclude(block_states[i], p);
            auto it = mi.find(z); if (it == mi.end()) return 5;
            nm[it->second] = ref_add(nm[it->second], c, mod);
        }
        rm.swap(nm); rd.swap(nd);
    }

    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    CpuLowPool out_pool(2);
    out_pool.run(jobs, main_auth, block_auth, storage, layout, lowdesc, mod);
    if (!compare_factor("out-of-place", main_auth, block_auth, main_states, block_states, rm, rd, storage, layout)) return 10;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowInplacePool in_pool(2);
    in_pool.run(jobs, main_auth, block_auth, storage, layout, lowdesc, orbit, mod);
    if (!compare_factor("in-place", main_auth, block_auth, main_states, block_states, rm, rd, storage, layout)) return 11;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowDirectPool direct_pool(2);
    direct_pool.run(jobs, main_auth, block_auth, storage, layout, lowdesc, orbit, mod);
    if (!compare_factor("direct", main_auth, block_auth, main_states, block_states, rm, rd, storage, layout)) return 12;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePool sparse_pool(2);
    sparse_pool.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse", main_auth, block_auth, main_states, block_states, rm, rd, storage, layout)) return 13;

    std::cout << "cpu-low-selftest OK W=" << W
              << " main=" << main_states.size() << " block=" << block_states.size()
              << " out_groups=" << out_pool.groups() << " in_groups=" << in_pool.groups()
              << " direct_groups=" << direct_pool.groups() << " sparse_groups=" << sparse_pool.groups()
              << " out_scratch_mib=" << double(out_pool.peak_scratch_bytes()) / (1 << 20)
              << " in_scratch_mib=" << double(in_pool.peak_scratch_bytes()) / (1 << 20)
              << " direct_scratch_mib=0 sparse_scratch_mib=0"
              << " dense_orbit_mib=" << double(orbit.rec.size() * sizeof(uint64_t)) / (1 << 20)
              << " sparse_meta_mib=" << double(sparse.orbit_ops.size()*sizeof(CpuLowSparseOrbitOp)+sparse.closure_ops.size()*sizeof(uint64_t))/(1<<20)
              << '\n';

    in_pool.release(); out_pool.release();
    main_auth.release(); block_auth.release();
    return 0;
}
