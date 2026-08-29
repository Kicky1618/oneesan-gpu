        if (ms.size)
            pm32_main_group_kernel<<<bm,threads,0,c.sMain>>>(
                cur, useMate ? c.dMate : nullptr, ms.size, nxt, dnext, p);
        if (ds.size)
            pm32_blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(
                dcur, ds.size, nxt, p);
        ck(cudaGetLastError(), "pm32 transition");
        ck(cudaEventRecord(c.mainDone, c.sMain), "record main");
        ck(cudaEventRecord(c.blockDone, c.sBlock), "record block");
        ck(cudaStreamWaitEvent(c.sMain, c.blockDone, 0), "main wait block");
        ck(cudaStreamWaitEvent(c.sBlock, c.mainDone, 0), "block wait main");
        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    ck(cudaStreamSynchronize(c.sMain), "main sync");
    ck(cudaStreamSynchronize(c.sBlock), "block sync");

    if (ms.size) {
        if (pg.use_mi)
            interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads>>>(
                cur, c.dIM, pg.mi.size());
        else
            scatter_main_kernel<<<bm,threads>>>(cur, ms.size);
    }
    if (ds.size) {
        if (pg.use_di)
            interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads>>>(
                dcur, c.dID, pg.di.size());
        else
            scatter_block_kernel<<<bd,threads>>>(dcur, ds.size);
    }
    ck(cudaGetLastError(), "pm32 scatter");
    ck(cudaDeviceSynchronize(), "pm32 group sync");
    c.groups++;
    c.active += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

static inline Count pm32_canonical_host(Count x, Count mod) {
    return mod == PM32_MOD && x >= PM32_MOD ? x - PM32_MOD : x;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 16;
    int target_mib = argc > 2 ? std::atoi(argv[2]) : 16384;
    int max_window = argc > 3 ? std::atoi(argv[3]) : 14;
    int requested = argc > 4 ? std::atoi(argv[4]) : 0;
    std::vector<Count> mods;
    for (int i=5; i<argc; ++i) {
        unsigned long long raw = std::strtoull(argv[i], nullptr, 10);
        if (raw < 2 || raw > 0xffffffffULL) {
            std::cerr << "HBM32 modulus must be in [2, 4294967295], got " << raw << "\n";
            return 1;
        }
        mods.push_back(Count(raw));
    }
    if (mods.empty()) mods.push_back(PM32_MOD);
    int W = n + 1;
    if (n < 2 || W > MAXW) { std::cerr << "n=2..27\n"; return 1; }
    if (W != TARGET_W) {
        std::cerr << "specialized for width " << TARGET_W
                  << " (n=" << (TARGET_W-1) << ")\n";
        return 1;
    }

    build_full_dp();
    G_FACTOR = build_factor_tables();
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "count");
    int ng = requested <= 0 ? visible : std::min(requested, visible);
    if (ng < 1 || ng > MAXGPU) { std::cerr << "need 1..8 GPUs\n"; return 2; }

    int peers = 0;
    for (int a=0; a<ng; ++a) for (int b=0; b<ng; ++b) if (a != b) {
        int can = 0;
        ck(cudaDeviceCanAccessPeer(&can, a, b), "can peer");
        if (can) {
            cudaSetDevice(a);
            auto e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
