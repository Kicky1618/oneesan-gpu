#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using oneesan::gridfp::MateID;
using oneesan::gridfp::MateValue;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;

struct DSU {
    std::vector<int> p, sz;
    explicit DSU(int n): p(n), sz(n,1) { std::iota(p.begin(),p.end(),0); }
    int find(int x){ return p[x]==x?x:p[x]=find(p[x]); }
    void join(int a,int b){ a=find(a); b=find(b); if(a==b)return; if(sz[a]<sz[b])std::swap(a,b); p[b]=a; sz[a]+=sz[b]; }
};

static uint32_t occ(uint32_t code,int len){
    uint32_t m=0;
    for(int p=0;p<len;++p) if((code>>(2*p))&3u) m|=1u<<p;
    return m;
}

static std::vector<std::vector<uint32_t>> enumerate_low(int len){
    std::vector<std::vector<uint32_t>> by_h(len+2);
    for(int h0=0;h0<=len+1;++h0){
        auto rec = [&](auto&& self,int pos,int h,uint32_t code)->void{
            if(pos<0){ if(h==0) by_h[h0].push_back(code); return; }
            if(h<0 || h>pos+1) return;
            self(self,pos-1,h,code);
            if(h>0) self(self,pos-1,h-1,code|(uint32_t(R)<<(2*pos)));
            self(self,pos-1,h+1,code|(uint32_t(L)<<(2*pos)));
        };
        rec(rec,len-1,h0,0);
    }
    return by_h;
}

static std::vector<std::vector<uint32_t>> enumerate_high(int len){
    std::vector<std::vector<uint32_t>> by_end(len+2);
    auto rec = [&](auto&& self,int pos,int h,uint32_t code)->void{
        if(pos<0){ if(h>=0 && h<(int)by_end.size()) by_end[h].push_back(code); return; }
        self(self,pos-1,h,code);
        if(h>0) self(self,pos-1,h-1,code|(uint32_t(R)<<(2*pos)));
        self(self,pos-1,h+1,code|(uint32_t(L)<<(2*pos)));
    };
    rec(rec,len-1,1,0);
    return by_end;
}

static void report(const char* name, DSU& d, const std::vector<uint8_t>& used){
    std::vector<int> cnt(used.size());
    int nused=0;
    for(int i=0;i<(int)used.size();++i) if(used[i]){ ++nused; ++cnt[d.find(i)]; }
    std::vector<int> comps;
    for(int i=0;i<(int)cnt.size();++i) if(cnt[i]) comps.push_back(cnt[i]);
    std::sort(comps.rbegin(),comps.rend());
    int largest=comps.empty()?0:comps[0];
    std::cout << name
              << " used_masks=" << nused
              << " components=" << comps.size()
              << " largest=" << largest
              << " largest_frac=" << (nused?double(largest)/nused:0.0);
    std::cout << " top=";
    for(size_t i=0;i<std::min<size_t>(8,comps.size());++i){ if(i)std::cout<<','; std::cout<<comps[i]; }
    std::cout << '\n';
}

int main(int argc,char**argv){
    int W = argc>1 ? std::atoi(argv[1]) : 28;
    int low = argc>2 ? std::atoi(argv[2]) : W/2;
    int high = W-low-1;
    if(W<3 || W>28 || low<=0 || high<=0 || low+high+1!=W){
        std::cerr << "usage: factor_window_mask_components [W [LOW]], W<=28\n";
        return 1;
    }

    auto lows = enumerate_low(low);
    auto highs = enumerate_high(high);
    const uint32_t LM=(1u<<low)-1u, HM=(1u<<high)-1u;
    const uint64_t LCODE=(uint64_t(1)<<(2*low))-1u;
    const uint64_t HCODE=(uint64_t(1)<<(2*high))-1u;

    DSU ld(1u<<low), hd(1u<<high);
    std::vector<uint8_t> lu(1u<<low), hu(1u<<high);
    for(auto const& v:lows) for(uint32_t c:v) lu[occ(c,low)]=1;
    for(auto const& v:highs) for(uint32_t c:v) hu[occ(c,high)]=1;

    uint64_t ledges=0, hedges=0;

    // LOW+center window: HIGH is inactive.  Scan every active-side topology;
    // one representative inactive topology per ending height is sufficient for
    // the active occupancy transition (the descriptor implementation verifies
    // exact-topology independence separately).
    for(int he=0;he<(int)highs.size();++he){
        if(highs[he].empty()) continue;
        uint32_t hc=highs[he][0];
        for(int cv=0;cv<3;++cv){
            int hs=he+(cv==int(L)?1:cv==int(R)?-1:0);
            if(hs<0 || hs>low+1 || hs>=(int)lows.size()) continue;
            for(uint32_t lc:lows[hs]){
                uint32_t a=occ(lc,low);
                MateID m=MateID(lc)|(MateID(cv)<<(2*low))|(MateID(hc)<<(2*(low+1)));
                for(int p=low;p>=1;--p){
                    auto z=oneesan::gridfp::include_horizontal(m,W,p);
                    if(!z.valid) continue;
                    uint32_t lc2=uint32_t(z.mate & LCODE);
                    uint32_t b=occ(lc2,low);
                    ld.join(a,b); ++ledges;
                }
            }
        }
    }
    // blocked -> main excluded branch also participates in the LOW window.
    for(int h=0;h<(int)highs.size() && h<(int)lows.size();++h){
        if(highs[h].empty()) continue;
        uint32_t hc=highs[h][0];
        for(uint32_t lc:lows[h]){
            uint32_t a=occ(lc,low);
            MateID m=MateID(lc)|(MateID(hc)<<(2*low));
            for(int p=low;p>=1;--p){
                MateID z=oneesan::gridfp::blocked_exclude(m,p);
                uint32_t lc2=uint32_t(z & LCODE);
                uint32_t b=occ(lc2,low);
                ld.join(a,b); ++ledges;
            }
        }
    }

    // HIGH+center window: LOW is inactive.
    for(int he=0;he<(int)highs.size();++he){
        if(highs[he].empty()) continue;
        for(int cv=0;cv<3;++cv){
            int hs=he+(cv==int(L)?1:cv==int(R)?-1:0);
            if(hs<0 || hs>low+1 || hs>=(int)lows.size() || lows[hs].empty()) continue;
            uint32_t lc=lows[hs][0];
            for(uint32_t hc:highs[he]){
                uint32_t a=occ(hc,high);
                MateID m=MateID(lc)|(MateID(cv)<<(2*low))|(MateID(hc)<<(2*(low+1)));
                for(int p=W-1;p>=low+1;--p){
                    auto z=oneesan::gridfp::include_horizontal(m,W,p);
                    if(!z.valid) continue;
                    uint32_t hc2 = z.blocked
                        ? uint32_t((z.mate>>(2*low)) & HCODE)
                        : uint32_t((z.mate>>(2*(low+1))) & HCODE);
                    uint32_t b=occ(hc2,high);
                    hd.join(a,b); ++hedges;
                }
            }
        }
    }
    for(int h=0;h<(int)highs.size() && h<(int)lows.size();++h){
        if(lows[h].empty()) continue;
        uint32_t lc=lows[h][0];
        for(uint32_t hc:highs[h]){
            uint32_t a=occ(hc,high);
            MateID m=MateID(lc)|(MateID(hc)<<(2*low));
            for(int p=W-1;p>=low+1;--p){
                MateID z=oneesan::gridfp::blocked_exclude(m,p);
                uint32_t hc2=uint32_t((z>>(2*(low+1))) & HCODE);
                uint32_t b=occ(hc2,high);
                hd.join(a,b); ++hedges;
            }
        }
    }

    std::cout << "W=" << W << " low=" << low << " high=" << high
              << " low_edges=" << ledges << " high_edges=" << hedges << '\n';
    report("LOW",ld,lu);
    report("HIGH",hd,hu);
    return 0;
}
