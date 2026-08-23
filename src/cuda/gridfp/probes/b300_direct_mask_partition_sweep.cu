#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>
#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"

struct SW{uint64_t orbit=0,closure=0;};struct SE{uint32_t v;uint64_t o,c;};
static uint64_t key2(uint32_t a,uint32_t b){if(a>b)std::swap(a,b);return(uint64_t(a)<<32)|b;}
static uint32_t maskr(const StorageFactorHost&f,int h,uint32_t r){return seg_occ(f.high_all_codes[f.high_all_off[h]+r],HIGH_LUT_K);}

int main(){constexpr int NG=8,S=MAXW+2;constexpr uint32_t NM=1u<<HIGH_LUT_K;
 build_full_dp();G_FACTOR=build_factor_tables();auto f=build_storage_factor_tables(G_FACTOR);auto l=build_storage_layout(f);
 auto ld=build_low_descriptors(f,l);auto hd=build_high_descriptors(f,l);auto lo=build_cpu_low_orbit(f,l,ld);auto ho=build_high_orbit(f,l);auto sp=build_b300_sparse_actions(l,ld,lo,hd,ho);
 std::vector<uint64_t> bytes(NM),work(NM);auto addb=[&](const StorageBlock&b){if(!b.valid||!b.cols)return;for(uint32_t m=0;m<NM;++m){size_t ix=size_t(m)*S+b.he;bytes[m]+=uint64_t(G_FACTOR.high_mask_off[ix+1]-G_FACTOR.high_mask_off[ix])*b.cols*4;}};
 for(auto&b:l.main_blocks)addb(b);for(auto&b:l.block_blocks)addb(b);
 std::unordered_map<uint64_t,SW> em;em.reserve(1<<20);uint64_t otot=0,ctot=0;
 auto oe=[&](uint32_t a,uint32_t b,uint64_t w){otot+=w;if(a!=b)em[key2(a,b)].orbit+=w;};auto ce=[&](uint32_t a,uint32_t b,uint64_t w){ctot+=w;if(a!=b)em[key2(a,b)].closure+=w;};
 for(auto&op:sp.high_orbit){auto&x=l.main_blocks[b300_sparse_sblock(op)];auto&j=l.main_blocks[b300_sparse_jblock(op)];auto&d=l.block_blocks[b300_sparse_dblock(op)];uint32_t s=maskr(f,x.he,b300_sparse_src(op));
   work[s]+=x.cols;oe(s,maskr(f,j.he,b300_sparse_jrank(op)),x.cols);oe(s,maskr(f,d.he,b300_sparse_drank(op)),x.cols);}
 for(uint64_t op:sp.high_closure){auto&x=l.main_blocks[b300_sparse_closure_sblock(op)];uint32_t desc=b300_sparse_closure_desc(op);auto&d=l.block_blocks[highdesc_block(desc)];uint32_t s=maskr(f,x.he,b300_sparse_closure_src(op));work[s]+=x.cols;ce(s,maskr(f,d.he,highdesc_rank(desc)),x.cols);}
 std::vector<std::vector<SE>> adj(NM);for(auto&kv:em){uint32_t a=kv.first>>32,b=uint32_t(kv.first);adj[a].push_back({b,kv.second.orbit,kv.second.closure});adj[b].push_back({a,kv.second.orbit,kv.second.closure});}
 std::vector<uint32_t> baseord(NM);std::iota(baseord.begin(),baseord.end(),0u);std::sort(baseord.begin(),baseord.end(),[&](uint32_t a,uint32_t b){return bytes[a]!=bytes[b]?bytes[a]>bytes[b]:a<b;});
 long double avgb=(long double)std::accumulate(bytes.begin(),bytes.end(),uint64_t(0))/NG,avgw=(long double)std::accumulate(work.begin(),work.end(),uint64_t(0))/NG;uint64_t slack=uint64_t(7.0L*(1ull<<30));uint64_t bmin=uint64_t(avgb-slack),bmax=uint64_t(avgb+slack);long double wmax=avgw*1.08L;
 for(int lambda:{1,2,4,8,16}){std::vector<uint8_t> own(NM);std::array<uint64_t,NG>gb{},gw{};for(uint32_t m:baseord){int g=std::min_element(gb.begin(),gb.end())-gb.begin();own[m]=g;gb[g]+=bytes[m];gw[g]+=work[m];}
   auto cuts=[&](){uint64_t o=0,c=0;for(auto&kv:em){uint32_t a=kv.first>>32,b=uint32_t(kv.first);if(own[a]!=own[b]){o+=kv.second.orbit;c+=kv.second.closure;}}return std::pair<uint64_t,uint64_t>(o,c);};auto c0=cuts();int moves=0,passes=0;
   for(int pass=0;pass<30;++pass){++passes;bool ch=false;std::vector<uint32_t> cand(NM);std::iota(cand.begin(),cand.end(),0u);std::sort(cand.begin(),cand.end(),[&](uint32_t a,uint32_t b){uint64_t da=0,db=0;for(auto e:adj[a])da+=e.o+uint64_t(lambda)*e.c;for(auto e:adj[b])db+=e.o+uint64_t(lambda)*e.c;return da>db;});
     for(uint32_t u:cand){int a=own[u],best=a;int64_t bg=0;std::array<uint64_t,NG>con{};for(auto e:adj[u])con[own[e.v]]+=e.o+uint64_t(lambda)*e.c;
       for(int b=0;b<NG;++b)if(b!=a&&gb[b]+bytes[u]<=bmax&&gb[a]-bytes[u]>=bmin&&(long double)(gw[b]+work[u])<=wmax){int64_t gain=int64_t(con[b])-int64_t(con[a]);if(gain>bg){bg=gain;best=b;}}
       if(best!=a){own[u]=best;gb[a]-=bytes[u];gb[best]+=bytes[u];gw[a]-=work[u];gw[best]+=work[u];++moves;ch=true;}}
     if(!ch)break;}
   auto c1=cuts();uint64_t mn=*std::min_element(gb.begin(),gb.end()),mx=*std::max_element(gb.begin(),gb.end());uint64_t wmn=*std::min_element(gw.begin(),gw.end()),wmx=*std::max_element(gw.begin(),gw.end());
   std::cout<<std::fixed<<std::setprecision(6)<<"mask_partition lambda="<<lambda<<" edges="<<em.size()<<" orbit_total="<<otot<<" closure_total="<<ctot
    <<" initial_orbit_cut="<<c0.first<<" initial_closure_cut="<<c0.second<<" orbit_cut="<<c1.first<<" closure_cut="<<c1.second
    <<" orbit_cut_fraction="<<(otot?double(c1.first)/otot:0)<<" closure_cut_fraction="<<(ctot?double(c1.second)/ctot:0)
    <<" weighted_reduction="<<((c0.first+uint64_t(lambda)*c0.second)?1.0-double(c1.first+uint64_t(lambda)*c1.second)/double(c0.first+uint64_t(lambda)*c0.second):0)
    <<" auth_imbalance="<<double(mx)/mn<<" work_imbalance="<<(wmn?double(wmx)/wmn:0)<<" moves="<<moves<<" passes="<<passes<<"\n";}
 return 0;}
