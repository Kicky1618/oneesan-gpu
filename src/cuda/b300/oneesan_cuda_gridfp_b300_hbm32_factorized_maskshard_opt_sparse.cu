// Reuse the validated direct mask-shard runtime and replace only the placement
// policy.  Keeping the baseline main available under another symbol makes A/B
// builds straightforward while avoiding a second copy of all CUDA kernels.
#define main(...) b300_maskshard_lpt_main(__VA_ARGS__)
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_maskshard_sparse.cu"
#undef main

#include "../gridfp/ramstream32_b300_direct_mask_partition_opt.cuh"

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int ng=argc>3?std::max(1,std::atoi(argv[3])):8;
    int th=argc>4?std::max(32,std::atoi(argv[4])):256;
    bool plan=argc>5&&std::strcmp(argv[5],"--plan-only")==0;
    int lambda=argc>6?std::max(1,std::atoi(argv[6])):16;
    double slack_gib=argc>7?std::max(0.25,std::atof(argv[7])):4.0;
    int pair_limit=argc>8?std::max(0,std::atoi(argv[8])):512;
    int W=n+1;
    if(W!=TARGET_W||W>MAXW||n<2||ng>MAXGPU)return 1;
    if constexpr(LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;

    build_full_dp();
    G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);
    StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);
    HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);
    HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);

    auto part0=std::chrono::steady_clock::now();
    B300MaskPartitionStats ps{};
    B300DirectMaskShardHost shard=build_b300_direct_mask_shards_optimized(
        f,l,sparse,ng,lambda,slack_gib,pair_limit,&ps);
    double partition_s=ram_seconds_since(part0);
    B300DirectSparsePartitionHost part=b300_direct_partition_high_by_mask(sparse,f,l,shard);

    Code ip=storage_rank_main_host(MateID(R)<<(2*(W-1)),f,l);
    Code ap=storage_rank_main_host(MateID(R),f,l);
    MaskLoc il=locate_mask(ip,l.main_blocks),al=locate_mask(ap,l.main_blocks);
    long double maskb=(long double)(G_FACTOR.low_mask_codes.size()+G_FACTOR.low_mask_off.size()+G_FACTOR.high_mask_codes.size()+G_FACTOR.high_mask_off.size())*4;
    long double storeb=(long double)(f.low_all_codes.size()+f.high_all_codes.size()+f.low_mask_begin.size()+f.high_mask_begin.size())*4;
    long double mapcommon=(long double)shard.high_owner.size()+(long double)shard.high_local.size()*4;
    long double lows=(long double)sparse.low_orbit.size()*sizeof(B300SparseOrbitOp)+(long double)sparse.low_closure.size()*8
                    +(long double)(sparse.low_orbit_off.size()+sparse.low_closure_off.size())*4;
    long double layoutb=(long double)(l.main_blocks.size()+l.block_blocks.size())*sizeof(StorageBlock)+sizeof(shard.main_off)+sizeof(shard.block_off);
    std::array<long double,MAXGPU> need{},hs{};
    long double maxneed=0,minauth=1e100L,maxauth=0;
    for(int g=0;g<ng;++g){
        hs[g]=(long double)part.high_orbit[g].size()*sizeof(B300SparseOrbitOp)+(long double)part.high_closure[g].size()*8
             +(long double)(part.high_orbit_off[g].size()+part.high_closure_off[g].size())*4;
        long double auth=(long double)(shard.main_count[g]+shard.block_count[g])*4;
        minauth=std::min(minauth,auth);maxauth=std::max(maxauth,auth);
        need[g]=auth+maskb+storeb+mapcommon+(long double)shard.owned_rows[g].size()*4+lows+hs[g]+layoutb;
        maxneed=std::max(maxneed,need[g]);
    }
    long double cgs=(long double)(4*l.main_size+2*l.block_size)*4*W;
    long double rp=(long double)(2*l.main_size+l.block_size)*4*W;
    double orbit_cut_frac=ps.orbit_total?double(ps.orbit_cut_after)/double(ps.orbit_total):0.0;
    double closure_cut_frac=ps.closure_total?double(ps.closure_cut_after)/double(ps.closure_total):0.0;
    long double remote_payload=(long double)(ps.orbit_cut_after+ps.closure_cut_after)*sizeof(Count)*W;
    long double before_weight=(long double)ps.orbit_cut_before+lambda*(long double)ps.closure_cut_before;
    long double after_weight=(long double)ps.orbit_cut_after+lambda*(long double)ps.closure_cut_after;

    if(plan){
        std::cout<<std::fixed<<std::setprecision(6)
            <<"backend=gridfp-b300-hbm32-factorized-maskshard-opt-sparse-plan n="<<n<<" gpus="<<ng
            <<" partition_lambda="<<lambda<<" auth_slack_gib="<<slack_gib<<" pair_limit="<<pair_limit
            <<" partition_s="<<partition_s<<" partition_moves="<<ps.moves<<" partition_swaps="<<ps.swaps
            <<" auth_total_gib="<<GiB((long double)(l.main_size+l.block_size)*4)
            <<" auth_min_gib="<<GiB(minauth)<<" auth_max_gib="<<GiB(maxauth)
            <<" auth_imbalance="<<double(maxauth/minauth)<<" partition_work_max_over_avg="<<ps.work_max_over_avg
            <<" runtime_groups=0 scratch_gib=0.000 low_p2p_bytes=0"
            <<" high_orbit_cut_fraction="<<orbit_cut_frac
            <<" high_closure_cut_fraction="<<closure_cut_frac
            <<" high_weighted_cut_reduction="<<(before_weight?double(1.0L-after_weight/before_weight):0.0)
            <<" high_remote_target_payload_tib_per_residue="<<double(remote_payload/(1ull<<40))
            <<" gather_scatter_tib_per_residue=0.000 host_pcie_tib_per_residue=0.000"
            <<" eliminated_canonical_gs_tib="<<double(cgs/(1ull<<40))
            <<" eliminated_ramstream_pcie_tib="<<double(rp/(1ull<<40))
            <<" low_sparse_replicated_mib="<<MiB(lows)
            <<" common_meta_mib="<<MiB(maskb+storeb+mapcommon+layoutb)
            <<" max_need_gib="<<GiB(maxneed)
            <<" headroom_288GB_gib="<<GiB(288.0e9L-maxneed)
            <<" headroom_279GB_gib="<<GiB(279.0e9L-maxneed)<<"\n";
        for(int g=0;g<ng;++g)std::cout
            <<"mask_opt_gpu="<<g
            <<" auth_gib="<<GiB((long double)(shard.main_count[g]+shard.block_count[g])*4)
            <<" owned_high_rows="<<shard.owned_rows[g].size()
            <<" high_sparse_mib="<<MiB(hs[g])<<" need_gib="<<GiB(need[g])<<"\n";
        return 0;
    }

    int visible=0;ck(cudaGetDeviceCount(&visible),"device count");
    if(visible<ng){std::cerr<<"need "<<ng<<" GPUs visible="<<visible<<"\n";return 3;}
    ld.main_desc.clear();ld.block_desc.clear();hd.main_desc.clear();hd.block_desc.clear();lo.rec.clear();ho.rec.clear();
    G_FACTOR.low_packed_rank.clear();G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear();G_FACTOR.high_packed_rank.shrink_to_fit();
    f.low_packed_rank.clear();f.low_packed_rank.shrink_to_fit();
    f.high_packed_rank.clear();f.high_packed_rank.shrink_to_fit();

    std::vector<std::unique_ptr<MaskGpu>> gpu;
    for(int g=0;g<ng;++g){auto c=std::make_unique<MaskGpu>();c->dev=g;c->alloc(shard.main_count[g],shard.block_count[g]);gpu.push_back(std::move(c));}
    peer_atomic_mesh(ng);
    std::array<Count*,MAXGPU>mp{},bp{};
    for(int g=0;g<ng;++g){mp[g]=gpu[g]->main;bp[g]=gpu[g]->block;}
    for(int g=0;g<ng;++g)gpu[g]->install(f,l,shard,sparse,part,mod,mp.data(),bp.data());

    uint32_t iai=f.high_all_off[l.main_blocks[il.bid].he]+il.hr;
    int io=shard.high_owner[iai];
    Code ix=local_mask_index(il,shard,f,l.main_blocks[il.bid],false);
    Count one=1;
    ck(cudaSetDevice(io),"init dev");
    ck(cudaMemcpy(gpu[io]->main+ix,&one,sizeof(one),cudaMemcpyHostToDevice),"init");

    auto t0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p)run_high(part,ng,p,th);
        for(int p=LOW_LUT_K;p>=1;--p)run_low(sparse,ng,p,th);
        std::cerr<<"maskshard-opt row "<<row+1<<'/'<<W<<"\n";
    }
    sync_all(ng,"final sync");
    double sec=ram_seconds_since(t0);
    uint32_t aai=f.high_all_off[l.main_blocks[al.bid].he]+al.hr;
    int ao=shard.high_owner[aai];
    Code ax=local_mask_index(al,shard,f,l.main_blocks[al.bid],false);
    Count ans=0;
    ck(cudaSetDevice(ao),"answer dev");
    ck(cudaMemcpy(&ans,gpu[ao]->main+ax,sizeof(ans),cudaMemcpyDeviceToHost),"answer");
    std::cout<<"backend=gridfp-b300-hbm32-factorized-maskshard-opt-sparse n="<<n
             <<" residue="<<ans<<" modulus="<<mod<<" gpus="<<ng
             <<" partition_lambda="<<lambda<<" partition_s="<<partition_s
             <<" high_orbit_cut_fraction="<<orbit_cut_frac
             <<" high_closure_cut_fraction="<<closure_cut_frac
             <<" groups=0 low_p2p_bytes=0 bulk_transfer_bytes=0 wall_s="<<sec<<"\n";
    for(auto&c:gpu)c->release();
    return 0;
}
