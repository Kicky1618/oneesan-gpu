#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page.hpp"
#include "../ramstream32_cpu_low_domain_worker_locality.hpp"

static void enum_states_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    enum_states_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) enum_states_rec(pos - 1, h - 1, mset(m, pos, R), out);
    enum_states_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}

static std::vector<MateID> enum_states(int width) {
    std::vector<MateID> out;
    enum_states_rec(width - 1, 1, 0, out);
    return out;
}

static inline Count ref_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static void fill_factor(
    RamCounts& main_auth, RamCounts& block_auth,
    const std::vector<MateID>& main_states,
    const std::vector<MateID>& block_states,
    const std::vector<Count>& main_values,
    const std::vector<Count>& block_values,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    std::memset(main_auth.ptr, 0, main_auth.bytes);
    std::memset(block_auth.ptr, 0, block_auth.bytes);
    for (size_t i = 0; i < main_states.size(); ++i)
        main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)] = main_values[i];
    for (size_t i = 0; i < block_states.size(); ++i)
        block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)] = block_values[i];
}

static bool compare_factor(
    const char* tag,
    const RamCounts& main_auth, const RamCounts& block_auth,
    const std::vector<MateID>& main_states,
    const std::vector<MateID>& block_states,
    const std::vector<Count>& want_main,
    const std::vector<Count>& want_block,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    for (size_t i = 0; i < main_states.size(); ++i) {
        Count got = main_auth.ptr[
            storage_rank_main_host(main_states[i], storage, layout)];
        if (got != want_main[i]) {
            std::cerr << "FAIL " << tag << " main i=" << i
                      << " got=" << got << " want=" << want_main[i] << '\n';
            return false;
        }
    }
    for (size_t i = 0; i < block_states.size(); ++i) {
        Count got = block_auth.ptr[
            storage_rank_block_host(block_states[i], storage, layout)];
        if (got != want_block[i]) {
            std::cerr << "FAIL " << tag << " block i=" << i
                      << " got=" << got << " want=" << want_block[i] << '\n';
            return false;
        }
    }
    return true;
}

static std::pair<std::vector<Count>, std::vector<Count>> reference_window(
    int W, int p_hi, int p_lo, Count mod,
    const std::vector<MateID>& main_states,
    const std::vector<MateID>& block_states,
    const std::unordered_map<MateID,size_t>& mi,
    const std::unordered_map<MateID,size_t>& di,
    const std::vector<Count>& init_main,
    const std::vector<Count>& init_block
) {
    std::vector<Count> rm = init_main, rd = init_block;
    for (int p = p_hi; p >= p_lo; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nd(rd.size(), 0);
        for (size_t i = 0; i < main_states.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(main_states[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = di.find(z.mate); if (it == di.end()) return {};
                nd[it->second] = ref_add(nd[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate); if (it == mi.end()) return {};
                nm[it->second] = ref_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < block_states.size(); ++i) {
            Count c = rd[i];
            MateID z = oneesan::gridfp::blocked_exclude(block_states[i], p);
            auto it = mi.find(z); if (it == mi.end()) return {};
            nm[it->second] = ref_add(nm[it->second], c, mod);
        }
        rm.swap(nm);
        rd.swap(nd);
    }
    return {std::move(rm), std::move(rd)};
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "selftest intentionally uses a small width");

    // Pure partition tests exercise both conversion and fallback paths without
    // depending on the W10 structural weight distribution.
    std::vector<CpuLowStaticJobCost> synthetic = {
        {0,0,4}, {1,1,4}, {2,2,4}, {3,3,4}
    };
    std::vector<std::pair<size_t,size_t>> segs;
    if (!cpu_low_domain_contiguous_segments_under_cap(
            synthetic, 0, synthetic.size(), 2, 8, segs)) return 20;
    if (segs.size() != 2) return 21;
    if (cpu_low_range_cells(synthetic, segs[0].first, segs[0].second) > 8
        || cpu_low_range_cells(synthetic, segs[1].first, segs[1].second) > 8)
        return 22;
    if (cpu_low_domain_contiguous_segments_under_cap(
            synthetic, 0, synthetic.size(), 2, 4, segs)) return 23;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size
        || block_states.size() != layout.block_size) return 2;

    std::unordered_map<MateID,size_t> mi, di;
    mi.reserve(main_states.size() * 2);
    di.reserve(block_states.size() * 2);
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    std::vector<Count> init_main(main_states.size()), init_block(block_states.size());
    std::mt19937_64 rng(1618);
    for (auto& x : init_main) x = Count(rng() % mod);
    for (auto& x : init_block) x = Count(rng() % mod);

    auto one = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, init_main, init_block);
    if (one.first.empty()) return 3;
    auto two = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, one.first, one.second);
    if (two.first.empty()) return 4;

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "worker-locality selftest main");
    block_auth.alloc(layout.block_size, "worker-locality selftest block");

    CpuLowSparsePersistentPool pool(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    pool.prepare_static_schedule(jobs, sparse);
    CpuLowDomainPageTieStats page_stats = cpu_low_apply_domain_page_tiebreak(
        pool, jobs, sparse, storage, layout);
    CpuLowDomainWorkerLocalityStats locality_stats =
        cpu_low_apply_domain_worker_locality(pool, jobs, sparse);
    if (locality_stats.max_worker_cells_after
        > locality_stats.max_worker_cells_before) return 5;

    fill_factor(
        main_auth, block_auth, main_states, block_states,
        init_main, init_block, storage, layout);
    pool.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "domain-worker-locality-1", main_auth, block_auth,
            main_states, block_states, one.first, one.second,
            storage, layout)) return 6;

    pool.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "domain-worker-locality-2", main_auth, block_auth,
            main_states, block_states, two.first, two.second,
            storage, layout)) return 7;

    std::cout << "cpu-low-domain-worker-locality-selftest OK"
              << " page_moves=" << page_stats.boundary_moves
              << " domains=" << locality_stats.domains
              << " converted_domains=" << locality_stats.converted_domains
              << " fallback_domains=" << locality_stats.fallback_domains
              << " max_before=" << locality_stats.max_worker_cells_before
              << " max_after=" << locality_stats.max_worker_cells_after
              << " synthetic_feasible=1 synthetic_fallback=1\n";

    pool.shutdown();
    main_auth.release();
    block_auth.release();
    return 0;
}
