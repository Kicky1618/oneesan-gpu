#define main bucket_fused_host_reference_main
#include "ramstream32_gpu_direct_gather_selftest.cu"
#undef main

#include "../ramstream32_gpu_direct_gather_cross.cuh"
#include "../ramstream32_gpu_direct_fused.cuh"
#include "../ramstream32_gpu_direct_fused_validate.hpp"
#include "../ramstream32_cpu_low_sparse.hpp"
#include "../ramstream32_bucket_layout.hpp"
#include "../ramstream32_bucket_direct.hpp"
#include "../ramstream32_bucket_fused.cuh"

using BkHostGrid=std::array<std::array<std::vector<Count>,BUCKET_NGPU>,BUCKET_NGPU>;

static BkHostGrid bkh_make_grid(
    const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::vector<Count>&mv,const std::vector<Count>&bv,
    const StorageFactorHost&storage,const StorageLayout&layout,
    const BucketOwnerHost&owner,const BucketPhysicalLayoutHost&phy
){
    BkHostGrid out;
    for(int a=0;a<BUCKET_NGPU;++a)for(int b=0;b<BUCKET_NGPU;++b)
        out[a][b].assign(size_t(phy.pair[a][b].size),0);
    for(size_t i=0;i<ms.size();++i){auto x=bucket_rank_main_host(ms[i],storage,layout,owner,phy);out[x.owner_h][x.owner_l][size_t(x.off)]=mv[i];}
    for(size_t i=0;i<bs.size();++i){auto x=bucket_rank_block_host(bs[i],storage,layout,owner,phy);out[x.owner_h][x.owner_l][size_t(x.off)]=bv[i];}
    return out;
}

static bool bkh_compare(
    const char*tag,const BkHostGrid&g,
    const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::vector<Count>&mv,const std::vector<Count>&bv,
    const StorageFactorHost&storage,const StorageLayout&layout,
    const BucketOwnerHost&owner,const BucketPhysicalLayoutHost&phy
){
    for(size_t i=0;i<ms.size();++i){auto x=bucket_rank_main_host(ms[i],storage,layout,owner,phy);Count got=g[x.owner_h][x.owner_l][size_t(x.off)];if(got!=mv[i]){std::cerr<<"FAIL "<<tag<<" main i="<<i<<" got="<<got<<" expected="<<mv[i]<<'\n';return false;}}
    for(size_t i=0;i<bs.size();++i){auto x=bucket_rank_block_host(bs[i],storage,layout,owner,phy);Count got=g[x.owner_h][x.owner_l][size_t(x.off)];if(got!=bv[i]){std::cerr<<"FAIL "<<tag<<" block i="<<i<<" got="<<got<<" expected="<<bv[i]<<'\n';return false;}}
    return true;
}

static Count bkh_sum_high_preimages(
    BkHostGrid&g,const BucketPhysicalLayoutHost&phy,const BucketFusedHost&f,
    uint32_t fixed,uint32_t dest_code,uint32_t depth,
    uint32_t source_bid,uint32_t source_low_loc,Count mod
){
    Count sum=0;int s=int(depth);
    uint32_t low_slot=bkf_loc_owner(source_low_loc),low_rank=bkf_loc_rank(source_low_loc);
    const BucketPhysicalBlock&sb=phy.pair[fixed][low_slot].main_blocks[source_bid];
    for(int pos=0;pos<HIGH_LUT_K;++pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(::L)){if(s==1)break;--s;}
        else if(v==uint32_t(R)){
            if(s==1){
                uint32_t z=3u<<(2*pos),src_code=(dest_code&~z)|(uint32_t(::L)<<(2*pos));
                uint32_t x=f.high_direct[gpu_direct_ternary_key_host(src_code,HIGH_LUT_K)];
                uint32_t xh=(x>>BKF_DIRECT_HEIGHT_SHIFT)&BKF_DIRECT_HEIGHT_MASK;
                if(x!=BKF_DIRECT_INVALID&&xh==sb.he){
                    uint32_t hl=x&BKF_LOC_MASK;
                    if(bkf_loc_owner(hl)==fixed){
                        uint32_t hr=bkf_loc_rank(hl);
                        Count v0=g[fixed][low_slot][size_t(sb.off+Code(hr)*sb.cols+low_rank)];
                        sum=gdg_add(sum,v0,mod);
                    }
                }
            }
            ++s;
        }
    }
    return sum;
}

static Count bkh_sum_low_preimages(
    BkHostGrid&g,const BucketPhysicalLayoutHost&phy,const BucketFusedHost&f,
    uint32_t fixed,uint32_t dest_code,uint32_t depth,
    uint32_t source_bid,uint32_t source_high_loc,Count mod
){
    Count sum=0;int s=int(depth);
    uint32_t high_slot=bkf_loc_owner(source_high_loc),high_rank=bkf_loc_rank(source_high_loc);
    const BucketPhysicalBlock&sb=phy.pair[high_slot][fixed].main_blocks[source_bid];
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(R)){if(s==1)break;--s;}
        else if(v==uint32_t(::L)){
            if(s==1){
                uint32_t z=3u<<(2*pos),src_code=(dest_code&~z)|(uint32_t(R)<<(2*pos));
                uint32_t x=f.low_direct[gpu_direct_ternary_key_host(src_code,LOW_LUT_K)];
                uint32_t xh=(x>>BKF_DIRECT_HEIGHT_SHIFT)&BKF_DIRECT_HEIGHT_MASK;
                if(x!=BKF_DIRECT_INVALID&&xh==sb.hs){
                    uint32_t ll=x&BKF_LOC_MASK;
                    if(bkf_loc_owner(ll)==fixed){
                        uint32_t lr=bkf_loc_rank(ll);
                        Count v0=g[high_slot][fixed][size_t(sb.off+Code(high_rank)*sb.cols+lr)];
                        sum=gdg_add(sum,v0,mod);
                    }
                }
            }
            ++s;
        }
    }
    return sum;
}

static void bkh_low_owner(
    BkHostGrid&g,uint32_t fixed,const StorageLayout&layout,
    const BucketPhysicalLayoutHost&phy,const BucketOrbitStreamsHost&o,
    const BucketFusedHost&f,Count mod
){
    size_t opitch=size_t(o.low_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        for(uint32_t bid=0;bid<o.low_nblocks;++bid){
            auto run=[&](const std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,uint32_t kind){
                uint32_t a=off[size_t(pi)*opitch+bid],b=off[size_t(pi)*opitch+bid+1];
                for(uint32_t q=a;q<b;++q){
                    BucketOrbitOp op=ops[q];uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op);
                    uint32_t ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);
                    const BucketPhysicalBlock&xb=phy.pair[fixed][ss].main_blocks[bid];if(!xb.valid||!xb.rows||!xb.cols)continue;
                    uint32_t jbid=bid;if(p==LOW_LUT_K){uint32_t center=kind==CPU_ORBIT_NR?uint32_t(R):uint32_t(::L);jbid=3u*uint32_t(xb.he)+center;}
                    const BucketPhysicalBlock&jb=phy.pair[fixed][js].main_blocks[jbid];
                    const BucketPhysicalBlock&db=phy.pair[fixed][ds].block_blocks[uint32_t(xb.he)];
                    uint32_t sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
                    for(uint32_t hr=0;hr<xb.rows;++hr){
                        Count&iv=g[fixed][ss][size_t(xb.off+Code(hr)*xb.cols+sr)];
                        Count&jv=g[fixed][js][size_t(jb.off+Code(hr)*jb.cols+jr)];
                        Count&dv=g[fixed][ds][size_t(db.off+Code(hr)*db.cols+dr)];
                        Count c=iv,old=dv;
                        if(kind==CPU_ORBIT_NN){jv=gdg_add(jv,c,mod);iv=gdg_add(c,old,mod);dv=0;}
                        else{Count cc=jv,all=gdg_add(gdg_add(c,cc,mod),old,mod);if(p==1){iv=all;jv=gdg_add(c,cc,mod);dv=0;}else{iv=all;dv=c;}}
                    }
                }
            };
            run(o.low_nn,o.low_nn_off,CPU_ORBIT_NN);
            run(o.low_nr,o.low_nr_off,CPU_ORBIT_NR);
            run(o.low_nl,o.low_nl_off,CPU_ORBIT_NL);
        }

        bool target_main=p==1;uint32_t nt=target_main?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){
            size_t oi=size_t(pi)*f.low_pitch+dbid;uint32_t a=f.low_off[oi],b=f.low_off[oi+1];
            for(uint32_t q=a;q<b;++q){
                const BucketFusedDst&rec=f.low_dst[q];uint32_t dslot=bkf_loc_owner(rec.dst_locator),dr=bkf_loc_rank(rec.dst_locator);
                const BucketPhysicalBlock&db=target_main?phy.pair[fixed][dslot].main_blocks[dbid]:phy.pair[fixed][dslot].block_blocks[dbid];
                uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
                for(uint32_t hr=0;hr<db.rows;++hr){
                    Count&dv=g[fixed][dslot][size_t(db.off+Code(hr)*db.cols+dr)];Count sum=dv;
                    for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=f.low_local_src[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);const auto&sb=phy.pair[fixed][ss].main_blocks[bkf_src_block(x)];sum=gdg_add(sum,g[fixed][ss][size_t(sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))],mod);}
                    if(cc){uint32_t dest_code=f.high_codes[f.high_code_off[size_t(fixed)*f.code_pitch+db.he]+hr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=f.low_cross_op[e],sl=bkf_src_locator(x);sum=gdg_add(sum,bkh_sum_high_preimages(g,phy,f,fixed,dest_code,bkf_cross_depth(x),bkf_src_block(x),sl,mod),mod);}}
                    dv=sum;
                }
            }
        }
    }
}

static void bkh_high_owner(
    BkHostGrid&g,uint32_t fixed,const StorageLayout&layout,
    const BucketPhysicalLayoutHost&phy,const BucketOrbitStreamsHost&o,
    const BucketFusedHost&f,Count mod
){
    size_t opitch=size_t(o.high_nblocks)+1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t pi=uint32_t((TARGET_W-1)-p);
        for(uint32_t bid=0;bid<o.high_nblocks;++bid){
            auto run=[&](const std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,bool nn){
                uint32_t a=off[size_t(pi)*opitch+bid],b=off[size_t(pi)*opitch+bid+1];
                for(uint32_t q=a;q<b;++q){
                    BucketOrbitOp op=ops[q];uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op);
                    uint32_t ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);
                    const BucketPhysicalBlock&xb=phy.pair[ss][fixed].main_blocks[bid];if(!xb.valid||!xb.rows||!xb.cols)continue;
                    uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}
                    const BucketPhysicalBlock&jb=phy.pair[js][fixed].main_blocks[jbid];
                    const BucketPhysicalBlock&db=phy.pair[ds][fixed].block_blocks[uint32_t(xb.hs)];
                    uint32_t sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
                    for(uint32_t lr=0;lr<xb.cols;++lr){
                        Count&iv=g[ss][fixed][size_t(xb.off+Code(sr)*xb.cols+lr)];
                        Count&jv=g[js][fixed][size_t(jb.off+Code(jr)*jb.cols+lr)];
                        Count&dv=g[ds][fixed][size_t(db.off+Code(dr)*db.cols+lr)];
                        Count c=iv,old=dv;if(nn){jv=gdg_add(jv,c,mod);iv=gdg_add(c,old,mod);dv=0;}else{Count cc=jv;iv=gdg_add(gdg_add(c,cc,mod),old,mod);dv=c;}
                    }
                }
            };
            run(o.high_nn,o.high_nn_off,true);run(o.high_nrnl,o.high_nrnl_off,false);
        }

        uint32_t nt=uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){
            size_t oi=size_t(pi)*f.high_pitch+dbid;uint32_t a=f.high_off[oi],b=f.high_off[oi+1];
            for(uint32_t q=a;q<b;++q){
                const BucketFusedDst&rec=f.high_dst[q];uint32_t dslot=bkf_loc_owner(rec.dst_locator),dr=bkf_loc_rank(rec.dst_locator);
                const BucketPhysicalBlock&db=phy.pair[dslot][fixed].block_blocks[dbid];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
                for(uint32_t lr=0;lr<db.cols;++lr){
                    Count&dv=g[dslot][fixed][size_t(db.off+Code(dr)*db.cols+lr)];Count sum=dv;
                    for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=f.high_local_src[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);const auto&sb=phy.pair[ss][fixed].main_blocks[bkf_src_block(x)];sum=gdg_add(sum,g[ss][fixed][size_t(sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)],mod);}
                    if(cc){uint32_t dest_code=f.low_codes[f.low_code_off[size_t(fixed)*f.code_pitch+db.hs]+lr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=f.high_cross_op[e],sl=bkf_src_locator(x);sum=gdg_add(sum,bkh_sum_low_preimages(g,phy,f,fixed,dest_code,bkf_cross_depth(x),bkf_src_block(x),sl,mod),mod);}}
                    dv=sum;
                }
            }
        }
    }
}

static void bkh_low(BkHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,const BucketOrbitStreamsHost&o,const BucketFusedHost&f,Count mod){for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed)bkh_low_owner(g,fixed,layout,phy,o,f,mod);}
static void bkh_high(BkHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,const BucketOrbitStreamsHost&o,const BucketFusedHost&f,Count mod){for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed)bkh_high_owner(g,fixed,layout,phy,o,f,mod);}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W;
    static_assert(W==LOW_LUT_K+HIGH_LUT_K+1);static_assert(W<=12,"host bucket selftest intentionally uses small width");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);

    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);if(ms.size()!=layout.main_size||bs.size()!=layout.block_size)return 2;
    std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);
    std::vector<Count>init_m(ms.size()),init_b(bs.size());std::mt19937_64 rng(1618);for(auto&x:init_m)x=Count(rng()%mod);for(auto&x:init_b)x=Count(rng()%mod);
    auto[low_m,low_b]=gdg_reference_window(W,LOW_LUT_K,1,mod,ms,bs,mi,di,init_m,init_b);auto[high_m,high_b]=gdg_reference_window(W,W-1,LOW_LUT_K+1,mod,ms,bs,mi,di,init_m,init_b);auto[row_m,row_b]=gdg_reference_window(W,LOW_LUT_K,1,mod,ms,bs,mi,di,high_m,high_b);

    auto lowg=bkh_make_grid(ms,bs,init_m,init_b,storage,layout,owner,phy);bkh_low(lowg,layout,phy,borbit,bfused,mod);if(!bkh_compare("host-bucket-low",lowg,ms,bs,low_m,low_b,storage,layout,owner,phy))return 10;
    auto highg=bkh_make_grid(ms,bs,init_m,init_b,storage,layout,owner,phy);bkh_high(highg,layout,phy,borbit,bfused,mod);if(!bkh_compare("host-bucket-high",highg,ms,bs,high_m,high_b,storage,layout,owner,phy))return 11;
    auto rowg=bkh_make_grid(ms,bs,init_m,init_b,storage,layout,owner,phy);bkh_high(rowg,layout,phy,borbit,bfused,mod);bkh_low(rowg,layout,phy,borbit,bfused,mod);if(!bkh_compare("host-bucket-row",rowg,ms,bs,row_m,row_b,storage,layout,owner,phy))return 12;

    std::cout<<"gpu-free-bucket-fused-selftest OK W="<<W<<" main="<<ms.size()<<" block="<<bs.size()<<" locator_bits="<<BUCKET_LOCATOR_BITS<<" closure_atomic=0\n";
    return 0;
}
