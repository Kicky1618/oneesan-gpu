                          << " max_bytes=" << mx << " target=" << target << "\n";
                return 4;
            }
            maxgroups=std::max(maxgroups,nj);
            std::sort(pw.groups.begin(),pw.groups.end(),
                      [](auto const& a, auto const& b){ return a.work>b.work; });
            std::cerr << "forced window p=" << hi << ".." << lo
                      << " fixed=" << k << " groups=" << nj
                      << " max_mib=" << (mx>>20) << "\n";
            schedule.push_back(std::move(pw));
        }
    }
    double prepare_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-prep0).count();
    std::cerr << "prepared windows=" << schedule.size()
              << " max_groups=" << maxgroups << " prepare_s=" << prepare_s << "\n";

    MateID init = MateID(R) << (2*(W-1));
    Code ig=rank_full(init,W);
    int io=int(ig/mc);
    Code fg=rank_full(MateID(R),W);
    int fo=int(fg/mc);
    Count one=1;
    for (size_t ri=0; ri<mods.size(); ++ri) {
        Count mod=mods[ri];
        for (int d=0; d<ng; ++d) {
            ck(cudaSetDevice(d), "set residue reset");
            ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "set modulus");
            if (ml[d]) ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)), "zero main");
            if (bl[d]) ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)), "zero block");
            ck(cudaDeviceSynchronize(), "zero sync");
            ctx[d].active=0;
            ctx[d].groups=0;
        }
        ck(cudaSetDevice(io), "set init device");
        ck(cudaMemcpy(mp[io]+(ig-Code(io)*mc), &one, sizeof(one),
                      cudaMemcpyHostToDevice), "init one");

        auto wall0=std::chrono::steady_clock::now();
        int done_windows=0;
        for (int row=0; row<W; ++row) {
            for (auto const& pw : schedule) {
                int nj=int(pw.groups.size());
                std::atomic<int> next{0};
                std::vector<std::thread> ths;
                ths.reserve(ng);
                for (int d=0; d<ng; ++d) {
                    ths.emplace_back([&,d]{
                        for (;;) {
                            int q=next.fetch_add(1,std::memory_order_relaxed);
                            if (q>=nj) break;
                            process_group_pm32(ctx[d],W,pw.wp,pw.groups[q],threads,target);
                        }
                    });
                }
                for (auto& t : ths) t.join();
                ++done_windows;
            }
            std::cerr << "mod " << (ri+1) << "/" << mods.size()
                      << " p=" << mod << " row " << (row+1) << "/" << W << "\n";
        }
        double wall=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-wall0).count();
        Count ans=0;
        ck(cudaSetDevice(fo), "set answer device");
        ck(cudaMemcpy(&ans,mp[fo]+(fg-Code(fo)*mc),sizeof(ans),
                      cudaMemcpyDeviceToHost), "answer");
        ans=pm32_canonical_host(ans,mod);

        double mx=0,sum=0;
        size_t maxIntervals=0;
        for (auto& c : ctx) {
            mx=std::max(mx,c.active);
            sum+=c.active;
