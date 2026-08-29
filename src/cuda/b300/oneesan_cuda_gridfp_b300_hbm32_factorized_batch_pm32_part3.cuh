            else ck(e, "enable peer");
            peers++;
        }
    }
    if (ng > 1 && peers != ng*(ng-1)) {
        std::cerr << "HBM mode requires full P2P: " << peers << "/"
                  << ng*(ng-1) << "\n";
        return 3;
    }

    uint32_t *fLA[MAXGPU]{}, *fLM[MAXGPU]{}, *fLO[MAXGPU]{}, *fLR[MAXGPU]{};
    uint32_t *fHA[MAXGPU]{}, *fHM[MAXGPU]{}, *fHO[MAXGPU]{}, *fHR[MAXGPU]{};
    Code *fHMB[MAXGPU]{}, *fHBB[MAXGPU]{};
    for (int d=0; d<ng; ++d) {
        cudaSetDevice(d);
        auto cp = [&](uint32_t** dst, const std::vector<uint32_t>& v, const char* w) {
            if (v.empty()) return;
            ck(cudaMalloc(dst, v.size()*sizeof(uint32_t)), w);
            ck(cudaMemcpy(*dst, v.data(), v.size()*sizeof(uint32_t),
                          cudaMemcpyHostToDevice), w);
        };
        cp(&fLA[d], G_FACTOR.low_all_codes, "f low all");
        cp(&fLM[d], G_FACTOR.low_mask_codes, "f low mask");
        cp(&fLO[d], G_FACTOR.low_mask_off, "f low off");
        cp(&fLR[d], G_FACTOR.low_packed_rank, "f low rank");
        cp(&fHA[d], G_FACTOR.high_all_codes, "f high all");
        cp(&fHM[d], G_FACTOR.high_mask_codes, "f high mask");
        cp(&fHO[d], G_FACTOR.high_mask_off, "f high off");
        cp(&fHR[d], G_FACTOR.high_packed_rank, "f high rank");
        auto cpc = [&](Code** dst, const std::vector<Code>& v, const char* w) {
            ck(cudaMalloc(dst, v.size()*sizeof(Code)), w);
            ck(cudaMemcpy(*dst, v.data(), v.size()*sizeof(Code),
                          cudaMemcpyHostToDevice), w);
        };
        cpc(&fHMB[d], G_FACTOR.high_main_base, "f high main base");
        cpc(&fHBB[d], G_FACTOR.high_block_base, "f high block base");
        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_CODES, &fLA[d], sizeof(fLA[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_MASK_CODES, &fLM[d], sizeof(fLM[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_MASK_OFF, &fLO[d], sizeof(fLO[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_PACKED_RANK, &fLR[d], sizeof(fLR[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_CODES, &fHA[d], sizeof(fHA[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_CODES, &fHM[d], sizeof(fHM[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_OFF, &fHO[d], sizeof(fHO[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_PACKED_RANK, &fHR[d], sizeof(fHR[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MAIN_BASE, &fHMB[d], sizeof(fHMB[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_BLOCK_BASE, &fHBB[d], sizeof(fHBB[d])), "f ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF, G_FACTOR.low_all_off.data(),
                              sizeof(uint32_t)*(MAXW+2)), "f low all off");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF, G_FACTOR.high_all_off.data(),
                              sizeof(uint32_t)*(MAXW+2)), "f high all off");
    }

    Code mainN = H_DP[W][1], blockN = H_DP[W-1][1];
    Code mc = (mainN + ng - 1) / ng, bc = (blockN + ng - 1) / ng;
    Count* mp[MAXGPU]{};
    Count* bp[MAXGPU]{};
    std::vector<Code> ml(ng), bl(ng);
    for (int d=0; d<ng; ++d) {
