#pragma once

#include "ramstream32_cpu_low_inplace.hpp"
#include "ramstream32_reverse_desc.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

struct ReverseOrbitHost {
    std::vector<uint64_t> rec;
    std::array<uint32_t,64> main_base{};
    uint32_t main_total=0;
    uint32_t steps=0;
    bool active_low=false;
    uint64_t orbit_sources=0,closures=0;
};

static inline uint32_t reverse_orbit_main_bid(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    uint32_t hc=uint32_t((m>>(2*(L+1)))&HM);
    return uint32_t(3*seg_end_height_host(hc,H)+int(mget(m,L)));
}
static inline uint32_t reverse_orbit_block_bid(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    uint32_t hc=uint32_t((m>>(2*L))&HM);
    return uint32_t(seg_end_height_host(hc,H));
}

static ReverseOrbitHost build_reverse_orbit(
    const StorageFactorHost&storage,const StorageLayout&layout,bool active_low
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,W=TARGET_W;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    ReverseOrbitHost o;o.active_low=active_low;o.steps=active_low?L:H;
    uint32_t total=0;
    for(uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid){
        o.main_base[bid]=total;total+=active_low?layout.main_blocks[bid].cols:layout.main_blocks[bid].rows;
    }
    o.main_total=total;o.rec.assign(size_t(total)*o.steps,0);
    auto rep_high=[&](int he){uint32_t a=storage.high_all_off[he],b=storage.high_all_off[he+1];return a<b?storage.high_all_codes[a]:0xffffffffu;};
    auto rep_low=[&](int hs){uint32_t a=storage.low_all_off[hs],b=storage.low_all_off[hs+1];return a<b?storage.low_all_codes[a]:0xffffffffu;};

    int p0=active_low?1:L+1,p1=active_low?L:W-1;
    for(int p=p0;p<=p1;++p){
        uint32_t pi=uint32_t(p-p0),q=uint32_t(W-p);
        for(uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid){
            const StorageBlock&sb=layout.main_blocks[bid];if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t n=active_low?sb.cols:sb.rows;
            uint32_t low0=storage.low_all_off[sb.hs],high0=storage.high_all_off[sb.he];
            uint32_t fixed_h=rep_high(sb.he),fixed_l=rep_low(sb.hs);
            if(fixed_h==0xffffffffu||fixed_l==0xffffffffu)continue;
            for(uint32_t ar=0;ar<n;++ar){
                uint32_t lc=active_low?storage.low_all_codes[low0+ar]:fixed_l;
                uint32_t hc=active_low?fixed_h:storage.high_all_codes[high0+ar];
                MateID m=MateID(lc)|(MateID(sb.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
                MateID mm=oneesan::gridfp::mirror_mate(m,W);
                MateValue a=mget(mm,int(q)),b=mget(mm,int(q)-1);uint64_t word=0;
                if(a==N){
                    CpuOrbitKind kind=CPU_ORBIT_NONE;MateValuePair pair=NN;
                    if(b==N){kind=CPU_ORBIT_NN;pair=LR;}
                    else if(b==R){kind=CPU_ORBIT_NR;pair=RN;}
                    else if(b==::L){kind=CPU_ORBIT_NL;pair=LN;}
                    if(kind!=CPU_ORBIT_NONE){
                        MateID jm=oneesan::gridfp::mirror_mate(msetpair(mm,int(q),pair),W);
                        MateID dm=oneesan::gridfp::mirror_mate(mshrink(mm,int(q)),W-1);
                        uint32_t jbid=reverse_orbit_main_bid(jm),dbid=reverse_orbit_block_bid(dm);
                        uint32_t jrank=0,drank=0;
                        if(active_low){
                            uint32_t jlc=uint32_t(jm)&LM,dlc=uint32_t(dm)&LM;
                            uint32_t jp=storage.low_packed_rank[jlc],dp=storage.low_packed_rank[dlc];if(jp==0xffffffffu||dp==0xffffffffu)std::exit(270);
                            jrank=jp>>L;drank=dp>>L;
                            uint32_t jhc=uint32_t((jm>>(2*(L+1)))&HM),dhc=uint32_t((dm>>(2*L))&HM);
                            if(jhc!=hc||dhc!=hc)std::exit(271);
                            if(jrank>=layout.main_blocks[jbid].cols||drank>=layout.block_blocks[dbid].cols)std::exit(272);
                        }else{
                            uint32_t jhc=uint32_t((jm>>(2*(L+1)))&HM),dhc=uint32_t((dm>>(2*L))&HM);
                            uint32_t jp=storage.high_packed_rank[jhc],dp=storage.high_packed_rank[dhc];if(jp==0xffffffffu||dp==0xffffffffu)std::exit(273);
                            jrank=jp>>H;drank=dp>>H;
                            uint32_t jlc=uint32_t(jm)&LM,dlc=uint32_t(dm)&LM;if(jlc!=lc||dlc!=lc)std::exit(274);
                            if(jrank>=layout.main_blocks[jbid].rows||drank>=layout.block_blocks[dbid].rows)std::exit(275);
                        }
                        word=cpu_orbit_pack(kind,jbid,jrank,dbid,drank);++o.orbit_sources;
                    }
                }else if((a==::L&&b==::L)||(a==R&&b==R)||(a==R&&b==::L)){
                    word=cpu_orbit_pack(CPU_ORBIT_CLOSURE);++o.closures;
                }
                o.rec[size_t(pi)*o.main_total+o.main_base[bid]+ar]=word;
            }
        }
    }
    std::cerr<<"reverse_orbit side="<<(active_low?"LOW":"HIGH")
             <<" active="<<o.main_total<<" steps="<<o.steps
             <<" orbit_sources="<<o.orbit_sources<<" closures="<<o.closures
             <<" mib="<<double(o.rec.size()*sizeof(uint64_t))/(1<<20)<<'\n';
    return o;
}
