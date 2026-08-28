#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <vector>
using namespace oneesan::gridfp;
using Set=std::unordered_set<MateID>;

static std::vector<MateID> words(int W){
    std::vector<MateID> out;
    auto rec=[&](auto&& self,int pos,int h,MateID m)->void{
        if(pos<0){if(h==0)out.push_back(m);return;}
        self(self,pos-1,h,m);
        if(h>0)self(self,pos-1,h-1,m|(MateID(R)<<(2*pos)));
        self(self,pos-1,h+1,m|(MateID(L)<<(2*pos)));
    };
    rec(rec,W-1,1,0);return out;
}
static bool capped(MateID m,int W,int row,int cut){
    int h=1;if(h>(0<cut?row:row-1))return false;
    for(int k=0;k<W;++k){auto v=mget(m,W-1-k);if(v==R)--h;else if(v==L)++h;
        int vertex=k+1,cap=vertex<cut?row:row-1;if(h<0||h>cap)return false;}
    return h==0;
}
static Set filter(std::vector<MateID>const&all,int W,int row,int cut){Set s;for(auto m:all)if(capped(m,W,row,cut))s.insert(m);return s;}
static void step(Set&main,Set&block,int W,int p){
    Set nm=main,nb;nm.reserve(main.size()*2+block.size()+1);nb.reserve(main.size()/2+1);
    for(auto m:main){auto z=include_horizontal(m,W,p);if(z.valid)(z.blocked?nb:nm).insert(z.mate);}
    for(auto b:block)nm.insert(blocked_exclude(b,p));main.swap(nm);block.swap(nb);
}
static std::uint64_t bounded(int W,int H){
    if(H<1)return 0;std::vector<std::uint64_t> a(H+1),b(H+1);a[1]=1;
    for(int k=0;k<W;++k){std::fill(b.begin(),b.end(),0);for(int h=0;h<=H;++h)if(a[h]){
        b[h]+=a[h];if(h)b[h-1]+=a[h];if(h<H)b[h+1]+=a[h];}a.swap(b);}return a[0];
}
static void eq(Set const&a,Set const&b,char const*kind,int W,int row,int t){if(a==b)return;
    std::cerr<<"FAIL "<<kind<<" W="<<W<<" row="<<row<<" t="<<t<<" got="<<a.size()<<" want="<<b.size()<<'\n';std::exit(2);}
int main(int argc,char**argv){
    int maxW=argc>1?std::atoi(argv[1]):10;if(maxW<3||maxW>12)return 1;
    for(int W=3;W<=maxW;++W){auto am=words(W),ab=words(W-1);Set main{MateID(R)<<(2*(W-1))},block;
        for(int row=1;row<=W;++row){
            if(row>1){eq(main,filter(am,W,row,0),"row-start",W,row,0);if(!block.empty())return 3;}
            for(int t=1;t<=W-1;++t){int p=W-t;step(main,block,W,p);
                eq(main,filter(am,W,row,t+1),"main",W,row,t);
                Set wb=t==W-1?Set{}:filter(ab,W-1,row,std::max(0,t-1));eq(block,wb,"blocked",W,row,t);}
            if(main.size()!=bounded(W,row)||!block.empty())return 4;
        }
        std::cout<<"height-cap-reachability W="<<W<<" states="<<am.size()<<" OK\n";
    }
    if(maxW>=10){std::cout<<"W28 row-end:";for(int h=1;h<=14;++h)std::cout<<' '<<bounded(28,h);std::cout<<'\n';}
}
