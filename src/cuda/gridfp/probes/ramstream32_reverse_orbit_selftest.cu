#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_reverse_desc.hpp"
#include "../ramstream32_reverse_orbit.hpp"

static void ros_enum_rec(int pos,int h,MateID m,std::vector<MateID>&out){
    if(pos<0){if(h==0)out.push_back(m);return;}
    ros_enum_rec(pos-1,h,mset(m,pos,N),out);
    if(h>0)ros_enum_rec(pos-1,h-1,mset(m,pos,R),out);
    ros_enum_rec(pos-1,h+1,mset(m,pos,::L),out);
}
static std::vector<MateID> ros_states(int w){std::vector<MateID>v;ros_enum_rec(w-1,1,0,v);return v;}
static Count ros_add(Count a,Count b,Count mod){if(!b)return a;return a>=mod-b?a-(mod-b):a+b;}

static std::pair<std::vector<Count>,std::vector<Count>> ros_reference(
    int p0,int p1,Count mod,const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::unordered_map<MateID,size_t>&mi,const std::unordered_map<MateID,size_t>&bi,
    std::vector<Count> rm,std::vector<Count> rb
){
    for(int p=p0;p<=p1;++p){
        std::vector<Count> nm=rm,nb(rb.size(),0);
        for(size_t i=0;i<ms.size();++i){Count c=rm[i];auto z=oneesan::gridfp::include_horizontal_reverse(ms[i],TARGET_W,p);if(!z.valid)continue;if(z.blocked){auto it=bi.find(z.mate);if(it==bi.end())std::exit(280);nb[it->second]=ros_add(nb[it->second],c,mod);}else{auto it=mi.find(z.mate);if(it==mi.end())std::exit(281);nm[it->second]=ros_add(nm[it->second],c,mod);}}
        for(size_t i=0;i<bs.size();++i){Count c=rb[i];MateID z=oneesan::gridfp::blocked_exclude_reverse(bs[i],TARGET_W,p);auto it=mi.find(z);if(it==mi.end())std::exit(282);nm[it->second]=ros_add(nm[it->second],c,mod);}
        rm.swap(nm);rb.swap(nb);
    }
    return {std::move(rm),std::move(rb)};
}

static void ros_fill_auth(
    std::vector<Count>&ma,std::vector<Count>&ba,const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::vector<Count>&mv,const std::vector<Count>&bv,const StorageFactorHost&s,const StorageLayout&layout
){
    std::fill(ma.begin(),ma.end(),0);std::fill(ba.begin(),ba.end(),0);
    for(size_t i=0;i<ms.size();++i)ma[size_t(storage_rank_main_host(ms[i],s,layout))]=mv[i];
    for(size_t i=0;i<bs.size();++i)ba[size_t(storage_rank_block_host(bs[i],s,layout))]=bv[i];
}
static bool ros_compare(
    const char*tag,const std::vector<Count>&ma,const std::vector<Count>&ba,const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::vector<Count>&mv,const std::vector<Count>&bv,const StorageFactorHost&s,const StorageLayout&layout
){
    for(size_t i=0;i<ms.size();++i){Count got=ma[size_t(storage_rank_main_host(ms[i],s,layout))];if(got!=mv[i]){std::cerr<<"FAIL "<<tag<<" main i="<<i<<" got="<<got<<" want="<<mv[i]<<'\n';return false;}}
    for(size_t i=0;i<bs.size();++i){Count got=ba[size_t(storage_rank_block_host(bs[i],s,layout))];if(got!=bv[i]){std::cerr<<"FAIL "<<tag<<" block i="<<i<<" got="<<got<<" want="<<bv[i]<<'\n';return false;}}
    return true;
}

static void ros_execute(
    bool active_low,std::vector<Count>&ma,std::vector<Count>&ba,const StorageFactorHost&s,const StorageLayout&layout,
    const ReverseLowDescHost&ld,const ReverseHighDescHost&hd,const ReverseOrbitHost&o,Count mod
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,W=TARGET_W;
    int p0=active_low?1:L+1,p1=active_low?L:W-1;
    for(int p=p0;p<=p1;++p){
        uint32_t pi=uint32_t(p-p0);
        // Three-state orbit pass. kind is the forward kind in reflected coordinates.
        for(uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid){
            const StorageBlock&sb=layout.main_blocks[bid];if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t n=active_low?sb.cols:sb.rows;
            for(uint32_t ar=0;ar<n;++ar){
                uint64_t ow=o.rec[size_t(pi)*o.main_total+o.main_base[bid]+ar];uint32_t kind=cpu_orbit_kind(ow);if(kind<CPU_ORBIT_NN||kind>CPU_ORBIT_NL)continue;
                const StorageBlock&jb=layout.main_blocks[cpu_orbit_jblock(ow)];const StorageBlock&db=layout.block_blocks[cpu_orbit_dblock(ow)];
                if(active_low){
                    for(uint32_t hr=0;hr<sb.rows;++hr){size_t ii=size_t(sb.off+Code(hr)*sb.cols+ar),ji=size_t(jb.off+Code(hr)*jb.cols+cpu_orbit_jlr(ow)),di=size_t(db.off+Code(hr)*db.cols+cpu_orbit_dlr(ow));Count c=ma[ii],old=ba[di];if(kind==CPU_ORBIT_NN){ma[ji]=ros_add(ma[ji],c,mod);ma[ii]=ros_add(c,old,mod);ba[di]=0;}else{Count cc=ma[ji],all=ros_add(ros_add(c,cc,mod),old,mod);ma[ii]=all;ba[di]=c;}}
                }else{
                    bool edge=(p==W-1);
                    for(uint32_t lr=0;lr<sb.cols;++lr){size_t ii=size_t(sb.off+Code(ar)*sb.cols+lr),ji=size_t(jb.off+Code(cpu_orbit_jlr(ow))*jb.cols+lr),di=size_t(db.off+Code(cpu_orbit_dlr(ow))*db.cols+lr);Count c=ma[ii],old=ba[di];if(kind==CPU_ORBIT_NN){ma[ji]=ros_add(ma[ji],c,mod);ma[ii]=ros_add(c,old,mod);ba[di]=0;}else{Count cc=ma[ji],all=ros_add(ros_add(c,cc,mod),old,mod);ma[ii]=all;if(edge){ma[ji]=ros_add(c,cc,mod);ba[di]=0;}else ba[di]=c;}}
                }
            }
        }
        // Closure pass follows the complete orbit pass.
        for(uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid){
            const StorageBlock&sb=layout.main_blocks[bid];if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t n=active_low?sb.cols:sb.rows;
            for(uint32_t ar=0;ar<n;++ar){
                uint64_t ow=o.rec[size_t(pi)*o.main_total+o.main_base[bid]+ar];if(cpu_orbit_kind(ow)!=CPU_ORBIT_CLOSURE)continue;
                if(active_low){
                    const ReverseDesc&x=ld.main_desc[size_t(pi)*ld.main_total+ld.main_base[bid]+ar];if(x.kind==REVDESC_INVALID)continue;
                    bool target_block=x.kind==REVDESC_BLOCK||x.kind==REVDESC_CROSS_BLOCK,cross=x.kind==REVDESC_CROSS_MAIN||x.kind==REVDESC_CROSS_BLOCK;
                    uint32_t high0=s.high_all_off[sb.he];
                    for(uint32_t hr=0;hr<sb.rows;++hr){Count c=ma[size_t(sb.off+Code(hr)*sb.cols+ar)];if(!c)continue;uint32_t hr2=hr;if(cross){uint32_t hc=s.high_all_codes[high0+hr],hc2=reverse_desc_flip_high(hc,x.depth),hp=s.high_packed_rank[hc2];if(hp==0xffffffffu)std::exit(283);hr2=hp>>H;}
                        if(target_block){const auto&d=layout.block_blocks[x.block];size_t j=size_t(d.off+Code(hr2)*d.cols+x.rank);ba[j]=ros_add(ba[j],c,mod);}else{const auto&d=layout.main_blocks[x.block];size_t j=size_t(d.off+Code(hr2)*d.cols+x.rank);ma[j]=ros_add(ma[j],c,mod);}}
                }else{
                    const ReverseDesc&x=hd.main_desc[size_t(pi)*hd.main_total+hd.main_base[bid]+ar];if(x.kind==REVDESC_INVALID)continue;
                    bool target_block=x.kind==REVDESC_BLOCK||x.kind==REVDESC_CROSS_BLOCK,cross=x.kind==REVDESC_CROSS_MAIN||x.kind==REVDESC_CROSS_BLOCK;
                    uint32_t low0=s.low_all_off[sb.hs];
                    for(uint32_t lr=0;lr<sb.cols;++lr){Count c=ma[size_t(sb.off+Code(ar)*sb.cols+lr)];if(!c)continue;uint32_t lr2=lr;if(cross){uint32_t lc=s.low_all_codes[low0+lr],lc2=reverse_desc_flip_low(lc,x.depth),lp=s.low_packed_rank[lc2];if(lp==0xffffffffu)std::exit(284);lr2=lp>>L;}
                        if(target_block){const auto&d=layout.block_blocks[x.block];size_t j=size_t(d.off+Code(x.rank)*d.cols+lr2);ba[j]=ros_add(ba[j],c,mod);}else{const auto&d=layout.main_blocks[x.block];size_t j=size_t(d.off+Code(x.rank)*d.cols+lr2);ma[j]=ros_add(ma[j],c,mod);}}
                }
            }
        }
    }
}

int main(){
    constexpr int W=TARGET_W,L=LOW_LUT_K;constexpr Count mod=4294967291u;
    static_assert(W<=12,"reverse orbit selftest intentionally uses small width");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    ReverseLowDescHost ld=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost hd=build_reverse_high_descriptors(storage,layout);
    ReverseOrbitHost lo=build_reverse_orbit(storage,layout,true),ho=build_reverse_orbit(storage,layout,false);
    auto ms=ros_states(W),bs=ros_states(W-1);std::unordered_map<MateID,size_t>mi,bi;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)bi.emplace(bs[i],i);
    std::mt19937_64 rng(0x1618beefULL);std::vector<Count>mv(ms.size()),bv(bs.size());for(auto&x:mv)x=Count(rng()%mod);for(auto&x:bv)x=Count(rng()%mod);
    std::vector<Count>ma(size_t(layout.main_size)),ba(size_t(layout.block_size));
    auto[wlm,wlb]=ros_reference(1,L,mod,ms,bs,mi,bi,mv,bv);ros_fill_auth(ma,ba,ms,bs,mv,bv,storage,layout);ros_execute(true,ma,ba,storage,layout,ld,hd,lo,mod);if(!ros_compare("reverse-low",ma,ba,ms,bs,wlm,wlb,storage,layout))return 10;
    auto[whm,whb]=ros_reference(L+1,W-1,mod,ms,bs,mi,bi,mv,bv);ros_fill_auth(ma,ba,ms,bs,mv,bv,storage,layout);ros_execute(false,ma,ba,storage,layout,ld,hd,ho,mod);if(!ros_compare("reverse-high",ma,ba,ms,bs,whm,whb,storage,layout))return 11;
    std::cout<<"reverse-orbit-selftest OK W="<<W<<" low_orbits="<<lo.orbit_sources<<" high_orbits="<<ho.orbit_sources<<" low_closures="<<lo.closures<<" high_closures="<<ho.closures<<'\n';return 0;
}
