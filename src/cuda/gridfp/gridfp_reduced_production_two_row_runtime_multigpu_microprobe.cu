#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_two_row_multigpu_microprobe_main_unused
#include "gridfp_reduced_production_two_row_multigpu_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_runtime_turn.cuh"

namespace {

struct RuntimeScheduleDevice {
    std::uint32_t* state = nullptr;
    std::uint32_t** peer = nullptr;
    Rank64* owner_begin = nullptr;
    Rank64 *iprefix = nullptr, *isr = nullptr, *icg = nullptr;
    Rank64 *cprefix = nullptr, *csr = nullptr, *ccg = nullptr;
    int* error = nullptr;
    Rank64 interior_components = 0;
    Rank64 compress_components = 0;
};

unsigned runtime_schedule_blocks(Rank64 components, unsigned cap) {
    const Rank64 per_block = Rank64(RP_RUNTIME_WARPS_PER_BLOCK) *
                             RP_RUNTIME_SUBGROUPS_PER_WARP;
    const Rank64 one_pass = (components + per_block - 1) / per_block;
    return static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(cap, one_pass)));
}

void runtime_launch_interior_all(
    std::vector<RuntimeScheduleDevice>& dev,
    int W, int q, bool reverse, int tile_start, int K,
    int ngpu, unsigned blocks, std::uint32_t mod
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime interior set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        owner_component_runtime_subwarp_kernel<<<
            runtime_schedule_blocks(d.interior_components, blocks), RP_RUNTIME_THREADS>>>(
            d.state, d.interior_components, W, q, reverse, tile_start, K, g, ngpu,
            d.owner_begin, d.iprefix, d.isr, d.icg, mod, d.error);
        ck(cudaGetLastError(), "runtime interior launch");
    }
}

void runtime_launch_turn_all(
    std::vector<RuntimeScheduleDevice>& dev,
    int W, int K, bool high, bool expand,
    int ngpu, unsigned blocks, std::uint32_t mod
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime turn set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        const Rank64 n = expand ? d.interior_components : d.compress_components;
        Rank64* prefix = expand ? d.iprefix : d.cprefix;
        Rank64* sr = expand ? d.isr : d.csr;
        Rank64* cg = expand ? d.icg : d.ccg;
        owner_turn_runtime_subwarp_kernel<<<
            runtime_schedule_blocks(n, blocks), RP_RUNTIME_THREADS>>>(
            d.state, n, W, K, high, expand, g, ngpu,
            d.owner_begin, prefix, sr, cg, mod, d.error);
        ck(cudaGetLastError(), "runtime turn launch");
    }
}

void runtime_sync_all(int ngpu) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime sync set device");
        ck(cudaDeviceSynchronize(), "runtime sync");
    }
}

void runtime_launch_redistribution_all(
    std::vector<RuntimeScheduleDevice>& dev,
    int W, int K, bool reverse, int ngpu, unsigned blocks
) {
    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 one_pass =
        (base_supports + RP_RUNTIME_WARPS_PER_BLOCK - 1) /
        RP_RUNTIME_WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime redistribute set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        schedule_p2p_cycle_kernel<<<launch_blocks, RP_RUNTIME_THREADS>>>(
            d.peer, base_supports, W, K, reverse, ngpu, g,
            d.owner_begin, d.error);
        ck(cudaGetLastError(), "runtime redistribute launch");
    }
    runtime_sync_all(ngpu);
}

void runtime_upload_plan(
    const HostOwnerComponentPlan& hp,
    Rank64*& d_prefix,
    Rank64*& d_sr,
    Rank64*& d_cg
) {
    ck(cudaMalloc(&d_prefix, hp.prefix.size() * sizeof(Rank64)), "runtime alloc prefix");
    ck(cudaMalloc(&d_sr, hp.sr_begin.size() * sizeof(Rank64)), "runtime alloc sr");
    ck(cudaMalloc(&d_cg, hp.component_group.size() * sizeof(Rank64)), "runtime alloc cg");
    ck(cudaMemcpy(d_prefix, hp.prefix.data(), hp.prefix.size() * sizeof(Rank64),
                  cudaMemcpyHostToDevice), "runtime copy prefix");
    ck(cudaMemcpy(d_sr, hp.sr_begin.data(), hp.sr_begin.size() * sizeof(Rank64),
                  cudaMemcpyHostToDevice), "runtime copy sr");
    ck(cudaMemcpy(d_cg, hp.component_group.data(), hp.component_group.size() * sizeof(Rank64),
                  cudaMemcpyHostToDevice), "runtime copy cg");
}

void run_two_row_runtime_schedule(
    int W, int ngpu, unsigned blocks, std::uint32_t mod
) {
    if ((W & 1) || W < 8) fail("runtime two-row schedule requires even W>=8");
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
    for (int p = W - 1; p >= 1; --p)
        ref = schedule_raw_step(ref, W, p, false, mod);
    for (int p = 1; p < W; ++p)
        ref = schedule_raw_step(ref, W, p, true, mod);
    ref = schedule_projected_step(ref, W, W - 1, false, mod);

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
    std::vector<RuntimeScheduleDevice> dev(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    std::vector<HostOwnerComponentPlan> ip(static_cast<std::size_t>(ngpu));
    std::vector<HostOwnerComponentPlan> cp(static_cast<std::size_t>(ngpu));

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime alloc set device");
        install_tables(tables);
        ip[static_cast<std::size_t>(g)] =
            make_host_owner_component_plan(tables, K, g, ngpu);
        cp[static_cast<std::size_t>(g)] =
            make_host_turn_compress_plan(tables, K, g, ngpu);
        auto& d = dev[static_cast<std::size_t>(g)];
        d.interior_components = ip[static_cast<std::size_t>(g)].prefix.back();
        d.compress_components = cp[static_cast<std::size_t>(g)].prefix.back();
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d.state, nstate * sizeof(std::uint32_t)), "runtime alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = d.state;
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(d.state, flat_input.data() + base,
                      nstate * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "runtime copy state");
        ck(cudaMalloc(&d.owner_begin, ngpu * sizeof(Rank64)), "runtime alloc owner begin");
        ck(cudaMemcpy(d.owner_begin, tile.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice),
           "runtime copy owner begin");
        runtime_upload_plan(ip[static_cast<std::size_t>(g)], d.iprefix, d.isr, d.icg);
        runtime_upload_plan(cp[static_cast<std::size_t>(g)], d.cprefix, d.csr, d.ccg);
        ck(cudaMalloc(&d.error, sizeof(int)), "runtime alloc error");
        ck(cudaMemset(d.error, 0, sizeof(int)), "runtime zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime peer table set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d.peer, ngpu * sizeof(std::uint32_t*)), "runtime alloc peer table");
        ck(cudaMemcpy(d.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "runtime copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();

    runtime_launch_turn_all(dev, W, K, true, true, ngpu, blocks, mod);
    for (int p = W - 2; p >= K + 2; --p)
        runtime_launch_interior_all(dev, W, p, false, W - 1, K, ngpu, blocks, mod);
    runtime_sync_all(ngpu);

    runtime_launch_redistribution_all(dev, W, K, false, ngpu, blocks);
    for (int p = K + 1; p >= 2; --p)
        runtime_launch_interior_all(dev, W, p, false, K + 1, K, ngpu, blocks, mod);

    runtime_launch_turn_all(dev, W, K, false, false, ngpu, blocks, mod);
    runtime_launch_turn_all(dev, W, K, false, true, ngpu, blocks, mod);
    for (int p = 2; p <= K; ++p)
        runtime_launch_interior_all(dev, W, p, true, 1, K, ngpu, blocks, mod);
    runtime_sync_all(ngpu);

    runtime_launch_redistribution_all(dev, W, K, true, ngpu, blocks);
    for (int p = K + 1; p <= W - 2; ++p)
        runtime_launch_interior_all(dev, W, p, true, K + 1, K, ngpu, blocks, mod);

    runtime_launch_turn_all(dev, W, K, true, false, ngpu, blocks, mod);
    runtime_launch_turn_all(dev, W, K, true, true, ngpu, blocks, mod);
    runtime_sync_all(ngpu);

    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime gather set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, d.error, sizeof(error), cudaMemcpyDeviceToHost),
           "runtime copy error");
        if (error)
            fail("runtime two-row device error=" + std::to_string(error) +
                 " gpu=" + std::to_string(g));
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, d.state,
                      nstate * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "runtime gather state");
    }
    if (flat_output != flat_expected)
        fail("counter-free two-row multi-GPU schedule mismatch");

    std::cout << "gridfp-reduced-two-row-runtime-multigpu"
              << " W=" << W << " K=" << K << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " two_rows=1 next_row_entry=1"
              << " redistributions=2"
              << " components_per_warp=4 subgroup_width=8"
              << " validation_component_atomics=0"
              << " state_streams_per_gpu=1"
              << " second_full_state_buffer_bytes=0"
              << " component_table_bytes=0 inverse_table_bytes=0"
              << " wall_ms=" << wall_ms
              << " exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "runtime free set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        cudaFree(d.error);
        cudaFree(d.ccg); cudaFree(d.csr); cudaFree(d.cprefix);
        cudaFree(d.icg); cudaFree(d.isr); cudaFree(d.iprefix);
        cudaFree(d.owner_begin); cudaFree(d.peer); cudaFree(d.state);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 2;
    const unsigned blocks = argc > 3
        ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const std::uint32_t mod = argc > 4
        ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 8 || W > RP_MAX_W || (W & 1) || ngpu < 2 ||
        ngpu > SCHEDULE_MAX_GPU || !blocks || mod < 3)
        return 2;
    const int K = (W - 2) / 2;
    if (plan_only) {
        ProductionFactorTables tables(W);
        const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
        Rank64 max_state = 0, max_ic = 0, max_cc = 0;
        for (int g = 0; g < ngpu; ++g) {
            max_state = std::max(
                max_state, tile.owner_size[static_cast<std::size_t>(g)]);
            max_ic = std::max(
                max_ic,
                make_host_owner_component_plan(tables, K, g, ngpu).prefix.back());
            max_cc = std::max(
                max_cc,
                make_host_turn_compress_plan(tables, K, g, ngpu).prefix.back());
        }
        std::cout << "gridfp-reduced-two-row-runtime-multigpu-plan"
                  << " W=" << W << " K=" << K << " ngpu=" << ngpu
                  << " states=" << tables.size()
                  << " max_state_GiB="
                  << double(max_state) * 4.0 / double(1ULL << 30)
                  << " max_interior_components_per_gpu=" << max_ic
                  << " max_turn_compress_components_per_gpu=" << max_cc
                  << " components_per_warp=4 subgroup_width=8"
                  << " validation_component_atomics=0"
                  << " state_streams_per_gpu=1 second_full_state_buffer_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; "
                     "use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "runtime schedule device count");
    if (visible < ngpu) return 4;
    run_two_row_runtime_schedule(W, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_two_row_runtime_multigpu=1 W="
              << W << '\n';
    return 0;
}
