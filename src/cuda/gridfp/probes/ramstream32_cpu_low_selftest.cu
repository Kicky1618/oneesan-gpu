#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low.hpp"

static void enum_states_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) {
        if (h == 0) out.push_back(m);
        return;
    }
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

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size || block_states.size() != layout.block_size) {
        std::cerr << "state count mismatch main=" << main_states.size() << '/' << layout.main_size
                  << " block=" << block_states.size() << '/' << layout.block_size << '\n';
        return 2;
    }

    std::unordered_map<MateID, size_t> mi, di;
    mi.reserve(main_states.size() * 2);
    di.reserve(block_states.size() * 2);
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "selftest main");
    block_auth.alloc(layout.block_size, "selftest block");
    std::vector<Count> rm(main_states.size()), rd(block_states.size());

    std::mt19937_64 rng(1618);
    for (size_t i = 0; i < main_states.size(); ++i) {
        Count v = Count(rng() % mod);
        rm[i] = v;
        main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)] = v;
    }
    for (size_t i = 0; i < block_states.size(); ++i) {
        Count v = Count(rng() % mod);
        rd[i] = v;
        block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)] = v;
    }

    // Direct reference for exactly the LOW+center window p=LOW..1.
    for (int p = LOW_LUT_K; p >= 1; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nd(rd.size(), 0);
        for (size_t i = 0; i < main_states.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(main_states[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = di.find(z.mate);
                if (it == di.end()) {
                    std::cerr << "reference blocked destination missing p=" << p << '\n';
                    return 3;
                }
                nd[it->second] = ref_add(nd[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate);
                if (it == mi.end()) {
                    std::cerr << "reference main destination missing p=" << p << '\n';
                    return 4;
                }
                nm[it->second] = ref_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < block_states.size(); ++i) {
            Count c = rd[i];
            MateID z = oneesan::gridfp::blocked_exclude(block_states[i], p);
            auto it = mi.find(z);
            if (it == mi.end()) {
                std::cerr << "reference blocked-exclude destination missing p=" << p << '\n';
                return 5;
            }
            nm[it->second] = ref_add(nm[it->second], c, mod);
        }
        rm.swap(nm);
        rd.swap(nd);
    }

    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);
    CpuLowPool pool(2);
    pool.run(jobs, main_auth, block_auth, storage, layout, lowdesc, mod);

    for (size_t i = 0; i < main_states.size(); ++i) {
        Count got = main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)];
        if (got != rm[i]) {
            std::cerr << "FAIL main i=" << i << " got=" << got << " want=" << rm[i] << '\n';
            return 10;
        }
    }
    for (size_t i = 0; i < block_states.size(); ++i) {
        Count got = block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)];
        if (got != rd[i]) {
            std::cerr << "FAIL block i=" << i << " got=" << got << " want=" << rd[i] << '\n';
            return 11;
        }
    }

    std::cout << "cpu-low-selftest OK W=" << W
              << " main=" << main_states.size()
              << " block=" << block_states.size()
              << " groups=" << pool.groups()
              << " scratch_mib=" << double(pool.peak_scratch_bytes()) / (1 << 20)
              << '\n';
    pool.release();
    main_auth.release();
    block_auth.release();
    return 0;
}
