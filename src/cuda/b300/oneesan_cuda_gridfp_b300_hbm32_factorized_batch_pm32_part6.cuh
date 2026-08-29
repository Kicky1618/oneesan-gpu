            maxIntervals=std::max(maxIntervals,c.maxIntervals);
        }
        std::cout << "backend=gridfp-b300-hbm32-factorized-batch-pm32"
                  << " n=" << n
                  << " residue=" << ans
                  << " modulus=" << mod
                  << " atomic_mode=" << (mod==PM32_MOD ? "pm32-add32" : "cas")
                  << " residue_index=" << ri
                  << " residues_total=" << mods.size()
                  << " gpus=" << ng
                  << " peers=" << peers
                  << " main_states=" << mainN
                  << " blocked_states=" << blockN
                  << " scratch_target_mib=" << effective_target_mib
                  << " windows=" << done_windows
                  << " max_groups=" << maxgroups
                  << " max_intervals=" << maxIntervals
                  << " active_max_s=" << mx
                  << " active_sum_s=" << sum
                  << " prepare_s=" << prepare_s
                  << " wall_s=" << wall
                  << std::endl;
    }

    for (auto& c : ctx) c.destroy();
    for (int d=0; d<ng; ++d) {
        cudaSetDevice(d);
        if (mp[d]) cudaFree(mp[d]);
        if (bp[d]) cudaFree(bp[d]);
        if (fLA[d]) cudaFree(fLA[d]);
        if (fLM[d]) cudaFree(fLM[d]);
        if (fLO[d]) cudaFree(fLO[d]);
        if (fLR[d]) cudaFree(fLR[d]);
        if (fHA[d]) cudaFree(fHA[d]);
        if (fHM[d]) cudaFree(fHM[d]);
        if (fHO[d]) cudaFree(fHO[d]);
        if (fHR[d]) cudaFree(fHR[d]);
        if (fHMB[d]) cudaFree(fHMB[d]);
        if (fHBB[d]) cudaFree(fHBB[d]);
    }
    return 0;
}
