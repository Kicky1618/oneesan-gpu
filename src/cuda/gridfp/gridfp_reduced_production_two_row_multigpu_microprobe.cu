#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_turn_high_owner_subwarp_microprobe_main_unused
#include "gridfp_reduced_production_turn_high_owner_subwarp_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_shift_cycle_device.cuh"

namespace {

static constexpr int SCHEDULE_MAX_GPU = 8;

__global__ void schedule_p2p_cycle_kernel(
    std::uint32_t** __restrict__ peer_state,
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int gpu_id,
    const Rank64* __restrict__ owner_begin,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, K, K, reverse);
            if (cycle_len < 0) {
                if (lane == 0) set_error(error, 291);
                continue;
            }
            if (cycle_len <= 1) continue;
            const DeviceKey leader = equal_run_key0_device(
                run.support, blocked, W, q, reverse);
            const GroupedDeviceRank lr = grouped_rank_device(
                leader, W, q, reverse, old_start, K, ngpu, owner_begin);
            if (lr.owner != gpu_id) continue;

            const int occupied = __popc(run.support);
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = peer_state[lr.owner][lr.local + i];
                std::uint32_t cur_support = shift_next_support_device(
                    run.support, blocked, W, q, K, K, reverse);
                int hops = 1;
                while (cur_support != run.support) {
                    const DeviceKey cur = equal_run_key0_device(
                        cur_support, blocked, W, q, reverse);
                    const GroupedDeviceRank cr = grouped_rank_device(
                        cur, W, q, reverse, old_start, K, ngpu, owner_begin);
                    const std::uint32_t next_value = peer_state[cr.owner][cr.local + i];
                    peer_state[cr.owner][cr.local + i] = temp;
                    temp = next_value;
                    cur_support = shift_next_support_device(
                        cur_support, blocked, W, q, K, K, reverse);
                    if (++hops > cycle_len) {
                        set_error(error, 292);
                        break;
                    }
                }
                peer_state[lr.owner][lr.local + i] = temp;
            }
            __syncwarp();
        }
    }
}

struct ScheduleDevice {
    std::uint32_t* state = nullptr;
    std::uint32_t** peer = nullptr;
    Rank64* owner_begin = nullptr;
    Rank64 *iprefix = nullptr, *isr = nullptr, *icg = nullptr;
    Rank64 *cprefix = nullptr, *csr = nullptr, *ccg = nullptr;
    unsigned long long* processed = nullptr;
    int* error = nullptr;
    Rank64 interior_components = 0;
    Rank64 compress_components = 0;
};

void schedule_enable_peer(int ngpu) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule peer source");
        for (int h = 0; h < ngpu; ++h) {
            if (g == h) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, g, h), "schedule can peer");
            if (!can) fail("schedule requires full peer access");
            const cudaError_t e = cudaDeviceEnablePeerAccess(h, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "schedule enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

using ModMap = std::map<Key,std::uint32_t>;

void schedule_add(ModMap& out, Key k, std::uint32_t v, int coef, std::uint32_t mod) {
    auto& z = out[k];
    add_mod_signed(z, v, coef, mod);
    if (z == 0) out.erase(k);
}

ModMap schedule_raw_step(const ModMap& in, int W, int p, bool reverse, std::uint32_t mod) {
    ModMap out;
    for (const auto& [k,v] : in)
        for (const auto& [d,c] : step_basis(k, W, p, reverse))
            schedule_add(out, d, v, int(c), mod);
    return out;
}

ModMap schedule_projected_step(const ModMap& in, int W, int p, bool reverse, std::uint32_t mod) {
    const int next = reverse ? p + 1 : p - 1;
    ModMap out;
    for (const auto& [k,v] : in) {
        const Vec col = project_vec(step_basis(k, W, p, reverse), W, next, reverse);
        for (const auto& [d,c] : col) schedule_add(out, d, v, int(c), mod);
    }
    return out;
}

void schedule_sync_all(int ngpu) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule sync device");
        ck(cudaDeviceSynchronize(), "schedule sync");
    }
}

unsigned schedule_blocks(Rank64 components, unsigned cap) {
    const Rank64 one_pass =
        (components + WARPS_PER_BLOCK * SUBGROUPS_PER_WARP - 1) /
        (WARPS_PER_BLOCK * SUBGROUPS_PER_WARP);
    return static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(cap, one_pass)));
}

void launch_interior_all(
    std::vector<ScheduleDevice>& dev,
    int W, int q, bool reverse, int tile_start, int K,
    int ngpu, unsigned blocks, std::uint32_t mod
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule interior set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        owner_component_subwarp_inplace_kernel<<<schedule_blocks(d.interior_components, blocks), THREADS>>>(
            d.state, d.interior_components, W, q, reverse, tile_start, K, g, ngpu,
            d.owner_begin, d.iprefix, d.isr, d.icg, mod, d.processed, d.error);
        ck(cudaGetLastError(), "schedule interior launch");
    }
}

void launch_turn_all(
    std::vector<ScheduleDevice>& dev,
    int W, int K, bool high, bool expand,
    int ngpu, unsigned blocks, std::uint32_t mod
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule turn set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        const Rank64 n = expand ? d.interior_components : d.compress_components;
        Rank64* prefix = expand ? d.iprefix : d.cprefix;
        Rank64* sr = expand ? d.isr : d.csr;
        Rank64* cg = expand ? d.icg : d.ccg;
        if (!high) {
            turn_owner_subwarp_inplace_kernel<<<schedule_blocks(n, blocks), THREADS>>>(
                d.state, n, W, K, expand, g, ngpu,
                d.owner_begin, prefix, sr, cg, mod, d.processed, d.error);
        } else {
            turn_high_owner_subwarp_inplace_kernel<<<schedule_blocks(n, blocks), THREADS>>>(
                d.state, n, W, K, expand, g, ngpu,
                d.owner_begin, prefix, sr, cg, mod, d.processed, d.error);
        }
        ck(cudaGetLastError(), "schedule turn launch");
    }
}

void launch_redistribution_all(
    std::vector<ScheduleDevice>& dev,
    int W, int K, bool reverse, int ngpu, unsigned blocks
) {
    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 one_pass = (base_supports + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule redistribute set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        schedule_p2p_cycle_kernel<<<launch_blocks, THREADS>>>(
            d.peer, base_supports, W, K, reverse, ngpu, g, d.owner_begin, d.error);
        ck(cudaGetLastError(), "schedule redistribute launch");
    }
    // A peer kernel on one GPU can still be touching another GPU's state after
    // that peer's own kernel has finished, so a global device barrier is needed
    // before any local in-place component kernel resumes.
    schedule_sync_all(ngpu);
}

void run_two_row_schedule(
    int W, int ngpu, unsigned blocks, std::uint32_t mod
) {
    if ((W & 1) || W < 8) fail("two-row schedule requires even W>=8");
    const int K = (W - 2) / 2;
    ProductionFactorTables tables(W);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const OwnerPlan owner_plan{tile.owner_begin, tile.owner_size};
    const auto main_words = gen_words(W);

    ModMap initial;
    Rank64 serial = 0;
    for (MateID m : main_words) {
        const std::uint32_t v = static_cast<std::uint32_t>(
            1 + (serial++ * 2654435761ULL) % (mod - 1ULL));
        initial.emplace(Key{false,m}, v);
    }

    ModMap ref = initial;
    for (int p = W - 1; p >= 1; --p) ref = schedule_raw_step(ref, W, p, false, mod);
    for (int p = 1; p < W; ++p) ref = schedule_raw_step(ref, W, p, true, mod);
    ref = schedule_projected_step(ref, W, W - 1, false, mod); // next row enters Q_{W-2}

    std::vector<std::uint32_t> flat_input(static_cast<std::size_t>(tables.size()), 0);
    std::vector<std::uint32_t> flat_expected(static_cast<std::size_t>(tables.size()), 0);
    for (const auto& [k,v] : initial) {
        const GroupedRank gr = grouped_rank(
            k, tables, W, W - 1, false, W - 1, K, ngpu, owner_plan);
        flat_input[static_cast<std::size_t>(tile.shard_base[gr.owner] + gr.local)] = v;
    }
    for (const auto& [k,v] : ref) {
        const GroupedRank gr = grouped_rank(
            k, tables, W, W - 2, false, W - 1, K, ngpu, owner_plan);
        flat_expected[static_cast<std::size_t>(tile.shard_base[gr.owner] + gr.local)] = v;
    }

    schedule_enable_peer(ngpu);
    std::vector<ScheduleDevice> dev(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    std::vector<HostOwnerComponentPlan> ip(static_cast<std::size_t>(ngpu));
    std::vector<HostOwnerComponentPlan> cp(static_cast<std::size_t>(ngpu));

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule alloc set device");
        install_tables(tables);
        ip[static_cast<std::size_t>(g)] = make_host_owner_component_plan(tables, K, g, ngpu);
        cp[static_cast<std::size_t>(g)] = make_host_turn_compress_plan(tables, K, g, ngpu);
        auto& d = dev[static_cast<std::size_t>(g)];
        d.interior_components = ip[static_cast<std::size_t>(g)].prefix.back();
        d.compress_components = cp[static_cast<std::size_t>(g)].prefix.back();
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d.state, nstate * sizeof(std::uint32_t)), "schedule alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = d.state;
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(d.state, flat_input.data() + base,
                      nstate * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "schedule copy state");
        ck(cudaMalloc(&d.owner_begin, ngpu * sizeof(Rank64)), "schedule alloc owner begin");
        ck(cudaMemcpy(d.owner_begin, tile.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "schedule copy owner begin");
        upload_plan(ip[static_cast<std::size_t>(g)], d.iprefix, d.isr, d.icg);
        upload_plan(cp[static_cast<std::size_t>(g)], d.cprefix, d.csr, d.ccg);
        ck(cudaMalloc(&d.processed, sizeof(unsigned long long)), "schedule alloc processed");
        ck(cudaMalloc(&d.error, sizeof(int)), "schedule alloc error");
        ck(cudaMemset(d.processed, 0, sizeof(unsigned long long)), "schedule zero processed");
        ck(cudaMemset(d.error, 0, sizeof(int)), "schedule zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule peer table set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d.peer, ngpu * sizeof(std::uint32_t*)), "schedule alloc peer table");
        ck(cudaMemcpy(d.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "schedule copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();

    // Enter forward row at the high edge: main -> Q_{W-2}.
    launch_turn_all(dev, W, K, true, true, ngpu, blocks, mod);
    for (int p = W - 2; p >= K + 2; --p)
        launch_interior_all(dev, W, p, false, W - 1, K, ngpu, blocks, mod);
    schedule_sync_all(ngpu);

    launch_redistribution_all(dev, W, K, false, ngpu, blocks);
    for (int p = K + 1; p >= 2; --p)
        launch_interior_all(dev, W, p, false, K + 1, K, ngpu, blocks, mod);

    // Low edge reverses scan direction in-place.
    launch_turn_all(dev, W, K, false, false, ngpu, blocks, mod);
    launch_turn_all(dev, W, K, false, true, ngpu, blocks, mod);
    for (int p = 2; p <= K; ++p)
        launch_interior_all(dev, W, p, true, 1, K, ngpu, blocks, mod);
    schedule_sync_all(ngpu);

    launch_redistribution_all(dev, W, K, true, ngpu, blocks);
    for (int p = K + 1; p <= W - 2; ++p)
        launch_interior_all(dev, W, p, true, K + 1, K, ngpu, blocks, mod);

    // Finish reverse row and immediately enter the next forward row.
    launch_turn_all(dev, W, K, true, false, ngpu, blocks, mod);
    launch_turn_all(dev, W, K, true, true, ngpu, blocks, mod);
    schedule_sync_all(ngpu);

    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule gather set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, d.error, sizeof(error), cudaMemcpyDeviceToHost), "schedule copy error");
        if (error) fail("two-row schedule device error=" + std::to_string(error) + " gpu=" + std::to_string(g));
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, d.state,
                      nstate * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "schedule gather state");
    }
    if (flat_output != flat_expected) fail("two-row multi-GPU schedule mismatch");

    std::cout << "gridfp-reduced-two-row-multigpu"
              << " W=" << W << " K=" << K << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " two_rows=1 next_row_entry=1"
              << " redistributions=2"
              << " state_streams_per_gpu=1"
              << " second_full_state_buffer_bytes=0"
              << " component_table_bytes=0 inverse_table_bytes=0"
              << " wall_ms_with_validation_counters=" << wall_ms
              << " exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "schedule free set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        cudaFree(d.error); cudaFree(d.processed);
        cudaFree(d.ccg); cudaFree(d.csr); cudaFree(d.cprefix);
        cudaFree(d.icg); cudaFree(d.isr); cudaFree(d.iprefix);
        cudaFree(d.owner_begin); cudaFree(d.peer); cudaFree(d.state);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 2;
    const unsigned blocks = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const std::uint32_t mod = argc > 4 ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 8 || W > RP_MAX_W || (W & 1) || ngpu < 2 || ngpu > SCHEDULE_MAX_GPU || !blocks || mod < 3)
        return 2;
    const int K = (W - 2) / 2;
    if (plan_only) {
        ProductionFactorTables tables(W);
        const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
        Rank64 max_state = 0, max_ic = 0, max_cc = 0;
        for (int g = 0; g < ngpu; ++g) {
            max_state = std::max(max_state, tile.owner_size[static_cast<std::size_t>(g)]);
            max_ic = std::max(max_ic, make_host_owner_component_plan(tables, K, g, ngpu).prefix.back());
            max_cc = std::max(max_cc, make_host_turn_compress_plan(tables, K, g, ngpu).prefix.back());
        }
        std::cout << "gridfp-reduced-two-row-multigpu-plan"
                  << " W=" << W << " K=" << K << " ngpu=" << ngpu
                  << " states=" << tables.size()
                  << " max_state_GiB=" << double(max_state) * 4.0 / double(1ULL<<30)
                  << " max_interior_components_per_gpu=" << max_ic
                  << " max_turn_compress_components_per_gpu=" << max_cc
                  << " components_per_warp=4 subgroup_width=8"
                  << " state_streams_per_gpu=1 second_full_state_buffer_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "schedule device count");
    if (visible < ngpu) return 4;
    run_two_row_schedule(W, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_two_row_multigpu=1 W=" << W << '\n';
    return 0;
}
