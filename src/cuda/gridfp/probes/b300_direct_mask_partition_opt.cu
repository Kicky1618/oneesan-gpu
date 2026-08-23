#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"

struct MpEdge { uint32_t v; uint64_t w; };

static uint32_t mp_mask(const StorageFactorHost& f,int h,uint32_t hr){
    return seg_occ(f.high_all_codes[f.high_all_off[h]+hr],HIGH_LUT_K);
}
static uint64_t mp_key(uint32_t a,uint32_t b){if(a>b)std::swap(a,b);return(uint64_t(a)<<32)|b;}

int main(){
    constexpr int NG=8,S=MAXW+2;constexpr uint32_t NM=1u<<HIGH_LUT_K;
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sp=build_b300_sparse_actions(l,ld,lo,hd,ho);

    std::vector<uint64_t> bytes(NM,0),work(NM,0);
    auto add_block=[&](const StorageBlock&b){if(!b.valid||!b.rows||!b.cols)return;for(uint32_t m=0;m<NM;++m){size_t ix=size_t(m)*S+b.he;
        bytes[m]+=uint64_t(G_FACTOR.high_mask_off[ix+1]-G_FACTOR.high_mask_off[ix])*b.cols*sizeof(Count);}};
    for(auto&b:l.main_blocks)add_block(b);for(auto&b:l.block_blocks)add_block(b);

    std::unordered_map<uint64_t,uint64_t> ew;ew.reserve(1<<20);uint64_t total_links=0;
    auto edge=[&](uint32_t a,uint32_t b,uint64_t w){total_links+=w;if(a==b)return;ew[mp_key(a,b)]+=w;};
    for(const auto&op:sp.high_orbit){const auto&x=l.main_blocks[b300_sparse_sblock(op)],&j=l.main_blocks[b300_sparse_jblock(op)];const auto&d=l.block_blocks[b300_sparse_dblock(op)];
        uint32_t sm=mp_mask(f,x.he,b300_sparse_src(op)),jm=mp_mask(f,j.he,b300_sparse_jrank(op)),dm=mp_mask(f,d.he,b300_sparse_drank(op));
        work[sm]+=x.cols;edge(sm,jm,x.cols);edge(sm,dm,x.cols);}
    for(uint64_t op:sp.high_closure){const auto&x=l.main_blocks[b300_sparse_closure_sblock(op)];uint32_t desc=b300_sparse_closure_desc(op);const auto&d=l.block_blocks[highdesc_block(desc)];
        uint32_t sm=mp_mask(f,x.he,b300_sparse_closure_src(op)),dm=mp_mask(f,d.he,highdesc_rank(desc));work[sm]+=x.cols;edge(sm,dm,x.cols);}

    std::vector<std::vector<MpEdge>> adj(NM);for(auto&kv:ew){uint32_t a=uint32_t(kv.first>>32),b=uint32_t(kv.first);uint64_t w=kv.second;adj[a].push_back({b,w});adj[b].push_back({a,w});}
    std::vector<uint32_t> ord(NM);std::iota(ord.begin(),ord.end(),0u);std::sort(ord.begin(),ord.end(),[&](uint32_t a,uint32_t b){if(bytes[a]!=bytes[b])return bytes[a]>bytes[b];return a<b;});
    std::vector<uint8_t> owner(NM);std::array<uint64_t,NG> gb{},gw{};for(uint32_t m:ord){int g=int(std::min_element(gb.begin(),gb.end())-gb.begin());owner[m]=uint8_t(g);gb[g]+=bytes[m];gw[g]+=work[m];}

    auto cut=[&](){uint64_t z=0;for(auto&kv:ew){uint32_t a=uint32_t(kv.first>>32),b=uint32_t(kv.first);if(owner[a]!=owner[b])z+=kv.second;}return z;};
    uint64_t cut0=cut();long double avg_b=(long double)std::accumulate(bytes.begin(),bytes.end(),uint64_t(0))/NG;
    long double avg_w=(long double)std::accumulate(work.begin(),work.end(),uint64_t(0))/NG;
    const uint64_t BSLACK=uint64_t(7.0L*(1ull<<30));const long double WMAX=avg_w*1.08L;
    uint64_t bmin=uint64_t(std::max<long double>(0,avg_b-BSLACK)),bmax=uint64_t(avg_b+BSLACK);

    int moves=0,passes=0;for(int pass=0;pass<30;++pass){++passes;bool changed=false;
        std::vector<uint32_t> cand(NM);std::iota(cand.begin(),cand.end(),0u);
        std::sort(cand.begin(),cand.end(),[&](uint32_t a,uint32_t b){uint64_t da=0,db=0;for(auto e:adj[a])da+=e.w;for(auto e:adj[b])db+=e.w;return da>db;});
        for(uint32_t u:cand){int a=owner[u];int best=a;int64_t bestgain=0;
            std::array<uint64_t,NG> conn{};for(auto e:adj[u])conn[owner[e.v]]+=e.w;
            for(int b=0;b<NG;++b)if(b!=a){if(gb[b]+bytes[u]>bmax||gb[a]-bytes[u]<bmin)continue;if((long double)(gw[b]+work[u])>WMAX)continue;
                int64_t gain=int64_t(conn[b])-int64_t(conn[a]);if(gain>bestgain){bestgain=gain;best=b;}}
            if(best!=a){owner[u]=uint8_t(best);gb[a]-=bytes[u];gb[best]+=bytes[u];gw[a]-=work[u];gw[best]+=work[u];++moves;changed=true;}}
        if(!changed)break;
    }
    uint64_t cut1=cut();uint64_t mn=*std::min_element(gb.begin(),gb.end()),mx=*std::max_element(gb.begin(),gb.end());
    uint64_t wmn=*std::min_element(gw.begin(),gw.end()),wmx=*std::max_element(gw.begin(),gw.end());
    auto gib=[](long double x){return double(x/(1ull<<30));};
    std::cout<<std::fixed<<std::setprecision(6)
        <<"b300-direct-mask-partition-opt W="<<TARGET_W<<" masks="<<NM<<" graph_edges="<<ew.size()<<" total_relation_cells="<<total_links
        <<" initial_cut_cells="<<cut0<<" initial_cut_fraction="<<(total_links?double(cut0)/total_links:0.0)
        <<" optimized_cut_cells="<<cut1<<" optimized_cut_fraction="<<(total_links?double(cut1)/total_links:0.0)
        <<" cut_reduction="<<(cut0?1.0-double(cut1)/cut0:0.0)<<" moves="<<moves<<" passes="<<passes<<"\n"
        <<"auth_min_gib="<<gib(mn)<<" auth_max_gib="<<gib(mx)<<" auth_imbalance="<<double(mx)/mn
        <<" high_work_imbalance="<<(wmn?double(wmx)/wmn:0.0)<<" auth_slack_gib=7.000000 work_cap_over_avg=1.080000\n";
    for(int g=0;g<NG;++g)std::cout<<"opt_gpu="<<g<<" auth_gib="<<gib(gb[g])<<" high_work="<<gw[g]<<"\n";
    return 0;
}
