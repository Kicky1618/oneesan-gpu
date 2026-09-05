#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
#include <unordered_map>

static std::vector<Mate> states(MateCodec const& mc){
    std::vector<Mate> o(mc.codeSize());
    for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}
    return o;
}

int main(int argc,char**argv){
    msg=NONE; int W=argc>1?std::atoi(argv[1]):12;
    PathCounter<uint64_t> pc(W,W,false,false);
    auto ms=states(pc.mc); auto ds=states(pc.wc);
    std::unordered_map<MateID,Code> mi,di; mi.reserve(ms.size()*2); di.reserve(ds.size()*2); for(Code i=0;i<ms.size();++i) mi.emplace(ms[i].id(),i); for(Code i=0;i<ds.size();++i) di.emplace(ds[i].id(),i);
    std::cout<<"W="<<W<<" main="<<ms.size()<<" block="<<ds.size()<<"\n";
    for(int p=1;p<W;++p){
        std::vector<uint16_t> im(ms.size(),0), ib(ds.size(),0);
        uint64_t valid=0,toM=0,toB=0;
        for(Code i=0;i<ms.size();++i){
            auto z=oneesan::gridfp::include_horizontal(ms[i].id(),W,p);
            if(!z.valid) continue; ++valid;
            if(z.blocked){auto it=di.find(z.mate); if(it!=di.end()){Code j=it->second; if(ib[j]!=UINT16_MAX) ++ib[j]; ++toB;}}
            else {auto it=mi.find(z.mate); if(it!=mi.end()){Code j=it->second; if(im[j]!=UINT16_MAX) ++im[j]; ++toM;}}
        }
        for(Code i=0;i<ds.size();++i){
            auto t=oneesan::gridfp::blocked_exclude(ds[i].id(),p);
            auto it=mi.find(t); if(it!=mi.end()){Code j=it->second; if(im[j]!=UINT16_MAX) ++im[j];}
        }
        auto stats=[&](auto const&v){
            uint64_t nz=0,sum=0;uint16_t mx=0;std::array<uint64_t,9> hist{};
            for(auto x:v){sum+=x;if(x)++nz;mx=std::max(mx,x);hist[std::min<int>(8,x)]++;}
            return std::tuple{nz,sum,mx,hist};
        };
        auto [mnz,msum,mmx,mh]=stats(im); auto [bnz,bsum,bmx,bh]=stats(ib);
        std::cout<<"p="<<std::setw(2)<<p<<" valid="<<valid<<" M<- inc+block avg="<<std::fixed<<std::setprecision(3)<<double(msum)/ms.size()<<" max="<<mmx<<" nz="<<100.0*mnz/ms.size()<<"% hist";
        for(int k=0;k<=std::min<int>(8,mmx);++k)std::cout<<' '<<k<<':'<<mh[k];
        std::cout<<" | B<-inc avg="<<double(bsum)/ds.size()<<" max="<<bmx<<" nz="<<100.0*bnz/ds.size()<<"% hist";
        for(int k=0;k<=std::min<int>(8,bmx);++k)std::cout<<' '<<k<<':'<<bh[k];
        std::cout<<"\n";
    }
}
