        ml[d] = std::min<Code>(mc, mainN - std::min<Code>(mainN, Code(d)*mc));
        bl[d] = std::min<Code>(bc, blockN - std::min<Code>(blockN, Code(d)*bc));
        cudaSetDevice(d);
        if (ml[d]) ck(cudaMalloc(&mp[d], size_t(ml[d])*sizeof(Count)), "auth main");
        if (bl[d]) ck(cudaMalloc(&bp[d], size_t(bl[d])*sizeof(Count)), "auth block");
    }

    std::vector<DeviceCtx> ctx(ng);
    for (int d=0; d<ng; ++d) ctx[d].init(d, mods[0], mp, bp, mc, bc, ng);

    size_t min_free = ~size_t(0), min_total = ~size_t(0);
    for (int d=0; d<ng; ++d) {
        ck(cudaSetDevice(d), "set meminfo");
        size_t f=0, t=0;
        ck(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
        min_free = std::min(min_free, f);
        min_total = std::min(min_total, t);
    }
    int reserve_mib = std::min(8192, std::max(256, int((min_total>>20)/32)));
    if (const char* e = std::getenv("GRIDFP_VRAM_RESERVE_MIB")) {
        int v = std::atoi(e);
        if (v >= 0) reserve_mib = v;
    }
    size_t requested_target = size_t(std::max(1, target_mib)) << 20;
    size_t reserve = size_t(reserve_mib) << 20;
    if (min_free <= reserve + (64ull<<20)) {
        std::cerr << "insufficient HBM after authoritative state: min_free_mib="
                  << (min_free>>20) << " reserve_mib=" << reserve_mib << "\n";
        return 5;
    }
    size_t target = std::min(requested_target, min_free-reserve);
    int effective_target_mib = int(target>>20);
    std::cerr << "HBM32 pm32 batch memory: auth_gib="
              << double(mainN+blockN)*sizeof(Count)/(1ull<<30)
              << " auth_per_gpu_gib=" << double(mainN+blockN)*sizeof(Count)/ng/(1ull<<30)
              << " min_total_gib=" << double(min_total)/(1ull<<30)
              << " min_free_after_auth_gib=" << double(min_free)/(1ull<<30)
              << " requested_scratch_mib=" << target_mib
              << " effective_scratch_mib=" << effective_target_mib
              << " reserve_mib=" << reserve_mib
              << " moduli=" << mods.size() << "\n";

    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W-1) {
        std::cerr << "forced2 requires LOW+HIGH=W-1\n";
        return 4;
    }
    (void)max_window;
    int threads = 256, maxgroups = 0;
    auto prep0 = std::chrono::steady_clock::now();
    std::vector<PreparedWindow> schedule;
    {
        const int ranges[2][2] = {{W-1, LOW_LUT_K+1}, {LOW_LUT_K, 1}};
        for (auto const& r : ranges) {
            int hi=r[0], lo=r[1];
            WindowPlan wp;
            wp.p_hi=hi; wp.p_lo=lo;
            wp.fixed_pos=window_candidates(W,hi,lo);
            int k=int(wp.fixed_pos.size());
            int nj=1<<k;
            PreparedWindow pw;
            pw.wp=wp;
            pw.groups.reserve(nj);
            size_t mx=0;
            for (int g=0; g<nj; ++g) {
                auto pg=prepare_group(W,pw.wp,g,mc,bc,ng);
                size_t b=size_t(2*pg.ms.size+2*pg.ds.size)*sizeof(Count);
                mx=std::max(mx,b);
                pw.groups.push_back(std::move(pg));
            }
            if (mx>target) {
                std::cerr << "forced window does not fit p=" << hi << ".." << lo
