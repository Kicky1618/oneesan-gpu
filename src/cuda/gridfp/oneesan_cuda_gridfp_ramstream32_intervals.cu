#include <cstring>

#define main oneesan_ramstream32_v1_unused_main
#include "oneesan_cuda_gridfp_ramstream32.cu"
#undef main

#ifndef RAMSTREAM_MIN_INTERVAL_ELEMS
#define RAMSTREAM_MIN_INTERVAL_ELEMS 16384ULL
#endif

struct Interval {
    Code global = 0;
    Code local = 0;
    Code len = 0;
};

static void add_interval(std::vector<Interval>& out, Code global, Code local, Code len) {
    if (!len) return;
    if (!out.empty() &&
        out.back().global + out.back().len == global &&
        out.back().local + out.back().len == local) {
        out.back().len += len;
    } else {
        out.push_back({global, local, len});
    }
}

static void intervals_rec(
    const GroupSpec& s, int pos, int h, Code global_base, Code local_base,
    std::vector<Interval>& out
) {
    if (pos < 0) {
        if (h == 0) add_interval(out, global_base, local_base, 1);
        return;
    }

    uint32_t lower = (pos == 31) ? 0xffffffffu : ((1u << (pos + 1)) - 1u);
    if ((s.fixed & lower) == 0) {
        add_interval(out, global_base, local_base, H_DP[pos + 1][h]);
        return;
    }

    bool fixed = (s.fixed >> pos) & 1u;
    bool occupied = (s.occ >> pos) & 1u;

    Code global_size = H_DP[pos][h];
    if (!fixed || !occupied) {
        Code local_size = s.dp[pos][h];
        intervals_rec(s, pos - 1, h, global_base, local_base, out);
        local_base += local_size;
    }
    global_base += global_size;

    if (h > 0) {
        global_size = H_DP[pos][h - 1];
        if (!fixed || occupied) {
            Code local_size = s.dp[pos][h - 1];
            intervals_rec(s, pos - 1, h - 1, global_base, local_base, out);
            local_base += local_size;
        }
        global_base += global_size;
    }

    if (h < MAXW + 1 && (!fixed || occupied)) {
        intervals_rec(s, pos - 1, h + 1, global_base, local_base, out);
    }
}

static std::vector<Interval> make_intervals(const GroupSpec& s) {
    std::vector<Interval> out;
    out.reserve(1024);
    intervals_rec(s, s.width - 1, 1, 0, 0, out);
    Code total = 0;
    for (const auto& x : out) total += x.len;
    if (total != s.size) {
        std::cerr << "interval size mismatch " << total << " != " << s.size << "\n";
        std::exit(6);
    }
    return out;
}

static Code interval_leaf_upper_rec(
    const GroupSpec& s, int pos, int h,
    Code memo[MAXW + 1][MAXW + 2], bool seen[MAXW + 1][MAXW + 2]
) {
    if (pos < 0) return h == 0 ? 1 : 0;
    if (seen[pos][h]) return memo[pos][h];
    seen[pos][h] = true;

    uint32_t lower = (pos == 31) ? 0xffffffffu : ((1u << (pos + 1)) - 1u);
    if ((s.fixed & lower) == 0) return memo[pos][h] = H_DP[pos + 1][h] ? 1 : 0;

    bool fixed = (s.fixed >> pos) & 1u;
    bool occupied = (s.occ >> pos) & 1u;
    Code z = 0;
    if (!fixed || !occupied) z += interval_leaf_upper_rec(s, pos - 1, h, memo, seen);
    if (!fixed || occupied) {
        if (h > 0) z += interval_leaf_upper_rec(s, pos - 1, h - 1, memo, seen);
        if (h < MAXW + 1) z += interval_leaf_upper_rec(s, pos - 1, h + 1, memo, seen);
    }
    return memo[pos][h] = z;
}

static Code interval_leaf_upper(const GroupSpec& s) {
    Code memo[MAXW + 1][MAXW + 2]{};
    bool seen[MAXW + 1][MAXW + 2]{};
    return interval_leaf_upper_rec(s, s.width - 1, 1, memo, seen);
}

struct PackingPlan {
    bool use_intervals = false;
    std::vector<Interval> intervals;
};

struct PackingStats {
    uint64_t interval_arrays = 0;
    uint64_t fallback_arrays = 0;
    uint64_t interval_runs = 0;
    long double interval_elems = 0;
};

static PackingPlan make_packing_plan(const GroupSpec& s) {
    PackingPlan plan;
    if (!s.size) return plan;

    Code upper = interval_leaf_upper(s);
    if (upper && s.size < upper * Code(RAMSTREAM_MIN_INTERVAL_ELEMS)) return plan;

    plan.intervals = make_intervals(s);
    if (!plan.intervals.empty() &&
        s.size < Code(plan.intervals.size()) * Code(RAMSTREAM_MIN_INTERVAL_ELEMS)) {
        plan.intervals.clear();
        return plan;
    }

    plan.use_intervals = true;
    return plan;
}

static void pack_with_plan(
    const HostCounts& auth, Count* staging, const GroupSpec& spec,
    const PackingPlan& plan, int cpu_threads, PackingStats& stats
) {
    if (!spec.size) return;
    if (!plan.use_intervals) {
        ++stats.fallback_arrays;
        gather_group(auth, staging, spec, cpu_threads);
        return;
    }

    ++stats.interval_arrays;
    stats.interval_runs += plan.intervals.size();
    stats.interval_elems += spec.size;
    cpu_parallel(Code(plan.intervals.size()), cpu_threads, [&](Code i) {
        const auto& x = plan.intervals[size_t(i)];
        std::memcpy(staging + x.local, auth.ptr + x.global, size_t(x.len) * sizeof(Count));
    });
}

static void unpack_with_plan(
    HostCounts& auth, const Count* staging, const GroupSpec& spec,
    const PackingPlan& plan, int cpu_threads
) {
    if (!spec.size) return;
    if (!plan.use_intervals) {
        scatter_group(auth, staging, spec, cpu_threads);
        return;
    }

    cpu_parallel(Code(plan.intervals.size()), cpu_threads, [&](Code i) {
        const auto& x = plan.intervals[size_t(i)];
        std::memcpy(auth.ptr + x.global, staging + x.local, size_t(x.len) * sizeof(Count));
    });
}

static void process_group_intervals(
    DeviceCtx& c, HostCounts& main_auth, HostCounts& block_auth,
    PackingStats& packing_stats,
    int W, const WindowPlan& wp, int g, int gpu_threads, int cpu_threads
) {
    uint32_t mf, mo, bf, bo;
    window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
    auto ms = make_spec(W, mf, mo);
    auto ds = make_spec(W - 1, bf, bo);
    if (!ms.size && !ds.size) return;
    if (!ms.size && ds.size) {
        std::cerr << "invalid group: blocked states without main states\n";
        std::exit(5);
    }

    c.ensure(ms.size, ds.size);
    PackingPlan main_plan = make_packing_plan(ms);
    PackingPlan block_plan = make_packing_plan(ds);

    auto t = std::chrono::steady_clock::now();
    if (ms.size) pack_with_plan(main_auth, c.hM.ptr, ms, main_plan, cpu_threads, packing_stats);
    if (ds.size) pack_with_plan(block_auth, c.hD.ptr, ds, block_plan, cpu_threads, packing_stats);
    c.pack_s += seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) ck(cudaMemcpy(c.dA, c.hM.ptr, size_t(ms.size) * sizeof(Count), cudaMemcpyHostToDevice), "H2D main");
    if (ds.size) ck(cudaMemcpy(c.dD, c.hD.ptr, size_t(ds.size) * sizeof(Count), cudaMemcpyHostToDevice), "H2D block");
    c.h2d_s += seconds_since(t);

    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "block occ");

    int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));
    int bd = int(std::min<Code>(65535, (ds.size + gpu_threads - 1) / gpu_threads));

    t = std::chrono::steady_clock::now();
    Count* cur = c.dA;
    Count* nxt = c.dB;
    for (int p = wp.p_hi; p >= wp.p_lo; --p) {
        if (ms.size) ck(cudaMemcpy(nxt, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToDevice), "identity");
        if (ds.size) blocked_group_kernel<<<bd, gpu_threads>>>(c.dD, ds.size, nxt, p);
        if (ds.size) ck(cudaMemset(c.dD, 0, size_t(ds.size) * sizeof(Count)), "clear block scratch");
        if (ms.size) main_group_kernel<<<bm, gpu_threads>>>(cur, ms.size, nxt, c.dD, p);
        ck(cudaGetLastError(), "transition kernel");
        std::swap(cur, nxt);
    }
    ck(cudaDeviceSynchronize(), "transition sync");
    c.kernel_s += seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) ck(cudaMemcpy(c.hM.ptr, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToHost), "D2H main");
    if (ds.size) ck(cudaMemcpy(c.hD.ptr, c.dD, size_t(ds.size) * sizeof(Count), cudaMemcpyDeviceToHost), "D2H block");
    c.d2h_s += seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) unpack_with_plan(main_auth, c.hM.ptr, ms, main_plan, cpu_threads);
    if (ds.size) unpack_with_plan(block_auth, c.hD.ptr, ds, block_plan, cpu_threads);
    c.unpack_s += seconds_since(t);
    ++c.groups;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 2147483647u;
    int target_mib = argc > 3 ? std::atoi(argv[3]) : 4096;
    int max_window = argc > 4 ? std::atoi(argv[4]) : 14;
    int cpu_threads = argc > 5 ? std::atoi(argv[5]) : int(std::max(1u, std::thread::hardware_concurrency()));
    int W = n + 1;

    if (n < 2 || W > MAXW) {
        std::cerr << "n=2..27\n";
        return 1;
    }
    if (W != TARGET_W) {
        std::cerr << "specialized for width " << TARGET_W << " (n=" << (TARGET_W - 1) << ")\n";
        return 1;
    }
    if (target_mib <= 0 || max_window <= 0 || cpu_threads <= 0) {
        std::cerr << "target_mib, max_window and cpu_threads must be positive\n";
        return 1;
    }

    build_full_dp();
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) {
        std::cerr << "need a CUDA GPU\n";
        return 2;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");

    Code mainN = H_DP[W][1];
    Code blockN = H_DP[W - 1][1];
    HostCounts main_auth, block_auth;
    main_auth.alloc(mainN);
    block_auth.alloc(blockN);

    MateID init = MateID(R) << (2 * (W - 1));
    main_auth.ptr[rank_full(init, W)] = 1;

    DeviceCtx ctx;
    ctx.init(mod);
    PackingStats packing_stats;

    size_t target = size_t(target_mib) << 20;
    int gpu_threads = 256;
    int total_windows = 0;
    int max_groups = 0;
    auto wall0 = std::chrono::steady_clock::now();

    for (int row = 0; row < W; ++row) {
        int hi = W - 1;
        while (hi >= 1) {
            WindowPlan wp;
            bool found = false;
            for (int lo = std::max(1, hi - max_window + 1); lo <= hi; ++lo) {
                auto candidate = plan_window(W, hi, lo, target);
                if (candidate.max_bytes && candidate.max_bytes <= target) {
                    wp = std::move(candidate);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "cannot fit window hi=" << hi << " target_mib=" << target_mib << "\n";
                return 4;
            }

            int groups = 1 << int(wp.fixed_pos.size());
            max_groups = std::max(max_groups, groups);
            ++total_windows;

            struct Job { int group; Code work; };
            std::vector<Job> jobs;
            jobs.reserve(groups);
            for (int g = 0; g < groups; ++g) {
                uint32_t mf, mo, bf, bo;
                window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
                auto ms = make_spec(W, mf, mo);
                auto ds = make_spec(W - 1, bf, bo);
                jobs.push_back({g, 2 * ms.size + ds.size});
            }
            std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.work > b.work; });

            for (const auto& job : jobs) {
                process_group_intervals(
                    ctx, main_auth, block_auth, packing_stats,
                    W, wp, job.group, gpu_threads, cpu_threads
                );
            }
            hi = wp.p_lo - 1;
        }
        std::cerr << "row " << (row + 1) << "/" << W
                  << " windows=" << total_windows
                  << " groups=" << ctx.groups
                  << " interval_arrays=" << packing_stats.interval_arrays
                  << " fallback_arrays=" << packing_stats.fallback_arrays
                  << "\n";
    }

    double wall_s = seconds_since(wall0);
    Code final_rank = rank_full(MateID(R), W);
    Count answer = main_auth.ptr[final_rank];
    double auth_gib = double(mainN + blockN) * sizeof(Count) / double(1ULL << 30);
    double avg_interval_elems = packing_stats.interval_runs
        ? double(packing_stats.interval_elems / packing_stats.interval_runs)
        : 0.0;

    std::cout
        << "backend=gridfp-ramstream32-intervals-v2"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " main_states=" << mainN
        << " blocked_states=" << blockN
        << " auth_gib=" << auth_gib
        << " scratch_target_mib=" << target_mib
        << " max_window=" << max_window
        << " cpu_threads=" << cpu_threads
        << " min_interval_elems=" << RAMSTREAM_MIN_INTERVAL_ELEMS
        << " windows=" << total_windows
        << " max_groups=" << max_groups
        << " groups=" << ctx.groups
        << " interval_arrays=" << packing_stats.interval_arrays
        << " fallback_arrays=" << packing_stats.fallback_arrays
        << " interval_runs=" << packing_stats.interval_runs
        << " avg_interval_elems=" << avg_interval_elems
        << " pack_s=" << ctx.pack_s
        << " h2d_s=" << ctx.h2d_s
        << " kernel_s=" << ctx.kernel_s
        << " d2h_s=" << ctx.d2h_s
        << " unpack_s=" << ctx.unpack_s
        << " wall_s=" << wall_s
        << "\n";

    ctx.destroy();
    main_auth.release();
    block_auth.release();
    return 0;
}
