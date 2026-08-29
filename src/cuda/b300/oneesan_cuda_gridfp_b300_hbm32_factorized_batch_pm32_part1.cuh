    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, fdb.data(), fdb.size()*sizeof(FBlock)), "factor block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &fm, sizeof(fm)), "factor main n");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &fd, sizeof(fd)), "factor block n");
    ck(cudaMemcpyToSymbol(D_F_MASK, &fmask, sizeof(fmask)), "factor mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fl, sizeof(fl)), "factor mode");

    size_t countBytes = size_t(2*ms.size + 2*ds.size) * sizeof(Count);
    size_t mateBytes = size_t(ms.size) * sizeof(MateID);
    bool useMate = !pg.use_mi && (countBytes + mateBytes <= target);
    c.ensure(ms.size, ds.size, useMate, pg.mi.size(), pg.di.size());

    if (!pg.mi.empty())
        ck(cudaMemcpy(c.dIM, pg.mi.data(), pg.mi.size()*sizeof(PeerInterval),
                      cudaMemcpyHostToDevice), "copy main intervals");
    if (!pg.di.empty())
        ck(cudaMemcpy(c.dID, pg.di.data(), pg.di.size()*sizeof(PeerInterval),
                      cudaMemcpyHostToDevice), "copy block intervals");
    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &pg.mf, sizeof(pg.mf)), "mf");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &pg.mo, sizeof(pg.mo)), "mo");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &pg.bf, sizeof(pg.bf)), "bf");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &pg.bo, sizeof(pg.bo)), "bo");
    int mw = W, bw = W - 1;
    ck(cudaMemcpyToSymbol(D_MAIN_W, &mw, sizeof(mw)), "mw");
    ck(cudaMemcpyToSymbol(D_BLOCK_W, &bw, sizeof(bw)), "bw");

    int bm = int(std::min<Code>(65535, (ms.size + threads - 1) / threads));
    int bd = int(std::min<Code>(65535, (ds.size + threads - 1) / threads));
    if (ms.size) {
        if (pg.use_mi)
            interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(
                c.dA, c.dIM, pg.mi.size());
        else
            gather_main_kernel<<<bm,threads>>>(c.dA, useMate ? c.dMate : nullptr, ms.size);
    }
    if (ds.size) {
        if (pg.use_di)
            interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(
                c.dD, c.dID, pg.di.size());
        else
            gather_block_kernel<<<bd,threads>>>(c.dD, ds.size);
    }
    ck(cudaGetLastError(), "pm32 gather");
    ck(cudaDeviceSynchronize(), "pm32 gather sync");

    Count* cur = c.dA;
    Count* nxt = c.dB;
    Count* dcur = c.dD;
    Count* dnext = c.dE;
    for (int p = wp.p_hi; p >= wp.p_lo; --p) {
        if (ms.size)
            ck(cudaMemcpyAsync(nxt, cur, size_t(ms.size)*sizeof(Count),
                               cudaMemcpyDeviceToDevice, c.sMain), "identity async");
        if (ds.size)
            ck(cudaMemsetAsync(dnext, 0, size_t(ds.size)*sizeof(Count),
                               c.sBlock), "clear next D");
        ck(cudaEventRecord(c.copyDone, c.sMain), "record copy");
        ck(cudaEventRecord(c.clearDone, c.sBlock), "record clear");
        ck(cudaStreamWaitEvent(c.sMain, c.clearDone, 0), "main wait clear");
        ck(cudaStreamWaitEvent(c.sBlock, c.copyDone, 0), "block wait copy");
