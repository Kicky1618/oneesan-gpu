#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

using namespace oneesan::gridfp;
using Code=std::uint64_t;
using Delta=std::int64_t;
static constexpr int MAXW=28;

namespace {
Code DP[MAXW+1][MAXW+2]{};

void build_dp(){
    for(int h=0;h<=MAXW+1;++h)DP[0][h]=(h==0);
    for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){
        Code z=DP[w-1][h];if(h>0)z+=DP[w-1][h-1];if(h<MAXW+1)z+=DP[w-1][h+1];DP[w][h]=z;
    }
}

Code contrib(MateValue v,int pos,int h){
    if(h<0||h>MAXW+1||v==N)return 0;
    Code z=DP[pos][h];
    if(v==L&&h>0)z+=DP[pos][h-1];
    return z;
}
Delta shift2(MateValue v,int pos,int h){return Delta(contrib(v,pos,h+2))-Delta(contrib(v,pos,h));}
Delta cross_ll(int pos,int h){return Delta(contrib(R,pos,h+2))-Delta(contrib(L,pos,h));}
Delta cross_rr(int pos,int h){return Delta(contrib(L,pos,h))-Delta(contrib(R,pos,h));}
int adv(int h,MateValue v){return h+(v==L)-(v==R);}

bool valid_mate(MateID m,int W){
    int h=1;for(int q=W-1;q>=0;--q){MateValue v=mget(m,q);if(v==R)--h;else if(v==L)++h;if(h<0)return false;}return h==0;
}
int height_before(MateID m,int W,int p){int h=1;for(int q=W-1;q>p;--q)h=adv(h,mget(m,q));return h;}

std::vector<MateID> gen_valid(int W){
    std::vector<MateID> out;
    auto rec=[&](auto&& self,int pos,int h,MateID m)->void{
        int rem=W-pos;if(h<0||h>rem)return;
        if(pos==W){if(h==0)out.push_back(m);return;}
        int q=W-1-pos;self(self,pos+1,h,m);if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*q)));self(self,pos+1,h+1,m|(MateID(L)<<(2*q)));
    };
    rec(rec,0,1,0);return out;
}

struct E{char kind;int q;Delta d;bool operator==(const E&o)const{return kind==o.kind&&q==o.q&&d==o.d;}};
bool less_e(const E&a,const E&b){return std::tie(a.kind,a.q,a.d)<std::tie(b.kind,b.q,b.d);}

std::vector<E> sequential(MateID b,int W,int p){
    const MateID d=minsert(b,p-1,N);const int H=height_before(d,W,p);std::vector<E> out;
    if(H>0)out.push_back({'A',-1,Delta(contrib(R,p,H))+Delta(contrib(L,p-1,H-1))});
    Delta ld=Delta(contrib(L,p,H))+Delta(contrib(L,p-1,H+1));int hb=H,bal=0;
    for(int q=p-2;q>=0;--q){MateValue v=mget(d,q);if(!is_endpoint(v))continue;
        if(bal==0&&v==L)out.push_back({'L',q,ld+cross_ll(q,hb)});
        ld+=shift2(v,q,hb);hb=adv(hb,v);if(v==L)++bal;else --bal;if(bal<0)break;
    }
    Delta rs=Delta(contrib(R,p,H+2))+Delta(contrib(R,p-1,H+1));int hbelow=H;bal=0;
    for(int q=p+1;q<W;++q){MateValue v=mget(d,q);if(!is_endpoint(v))continue;
        int hq=hbelow+(v==R)-(v==L);
        if(bal==0&&v==R)out.push_back({'R',q,cross_rr(q,hq)+rs});
        rs+=shift2(v,q,hq);hbelow=hq;if(v==R)++bal;else --bal;if(bal<0)break;
    }
    std::sort(out.begin(),out.end(),less_e);return out;
}

template<class T,class Op>void scan32(std::array<T,32>&a,Op op){
    for(int off=1;off<32;off<<=1){auto old=a;for(int lane=off;lane<32;++lane)a[lane]=op(old[lane-off],old[lane]);}
}

std::vector<E> warpscan(MateID b,int W,int p){
    const MateID d=minsert(b,p-1,N);const int H=height_before(d,W,p);std::vector<E> out;
    if(H>0)out.push_back({'A',-1,Delta(contrib(R,p,H))+Delta(contrib(L,p-1,H-1))});

    std::array<int,32> ls{},lmins{};std::array<MateValue,32> lv{};std::array<int,32> lq{};
    for(int lane=0;lane<32;++lane){int q=p-2-lane;lq[lane]=q;MateValue v=q>=0?mget(d,q):N;lv[lane]=v;ls[lane]=(v==L)-(v==R);}
    auto lbal=ls;scan32(lbal,[](int a,int b){return a+b;});lmins=lbal;scan32(lmins,[](int a,int b){return std::min(a,b);});
    std::array<Delta,32> lt{};
    for(int lane=0;lane<32;++lane){int pre=lbal[lane]-ls[lane],h=H+pre;if(lq[lane]>=0&&h>=0&&h<=MAXW+1)lt[lane]=shift2(lv[lane],lq[lane],h);}
    auto lp=lt;scan32(lp,[](Delta a,Delta b){return a+b;});
    const Delta li=Delta(contrib(L,p,H))+Delta(contrib(L,p-1,H+1));
    for(int lane=0;lane<32;++lane){int q=lq[lane],pre=lbal[lane]-ls[lane],h=H+pre;if(q>=0&&lv[lane]==L&&pre==0&&lmins[lane]>=0)out.push_back({'L',q,li+(lp[lane]-lt[lane])+cross_ll(q,h)});}

    std::array<int,32> rs{},rmins{};std::array<MateValue,32> rv{};std::array<int,32> rq{};
    for(int lane=0;lane<32;++lane){int q=p+1+lane;rq[lane]=q;MateValue v=q<W?mget(d,q):N;rv[lane]=v;rs[lane]=(v==R)-(v==L);}
    auto rbal=rs;scan32(rbal,[](int a,int b){return a+b;});rmins=rbal;scan32(rmins,[](int a,int b){return std::min(a,b);});
    std::array<Delta,32> rt{};
    for(int lane=0;lane<32;++lane){int h=H+rbal[lane];if(rq[lane]<W&&h>=0&&h<=MAXW+1)rt[lane]=shift2(rv[lane],rq[lane],h);}
    auto rp=rt;scan32(rp,[](Delta a,Delta b){return a+b;});
    const Delta ri=Delta(contrib(R,p,H+2))+Delta(contrib(R,p-1,H+1));
    for(int lane=0;lane<32;++lane){int q=rq[lane],pre=rbal[lane]-rs[lane],h=H+rbal[lane];if(q<W&&rv[lane]==R&&pre==0&&rmins[lane]>=0)out.push_back({'R',q,ri+(rp[lane]-rt[lane])+cross_rr(q,h)});}

    std::sort(out.begin(),out.end(),less_e);return out;
}
}

int main(){
    build_dp();std::uint64_t states=0,closures=0,candidates=0;std::size_t maxcand=0;
    for(int W=4;W<=12;++W){
        auto block=gen_valid(W-1);
        for(MateID b:block)for(int p=2;p<W;++p){
            ++states;if(mget(b,p-1)!=N)continue;++closures;
            MateID d=minsert(b,p-1,N);if(!valid_mate(d,W))return 20;
            auto a=sequential(b,W,p),c=warpscan(b,W,p);
            if(a!=c){
                std::cerr<<"mismatch W="<<W<<" p="<<p<<" b="<<b<<" seq="<<a.size()<<" warp="<<c.size()<<'\n';return 2;
            }
            candidates+=a.size();maxcand=std::max(maxcand,a.size());
        }
    }
    std::cout<<"b300-block-closure-warpscan-proof OK exhaustive_width_max=12 states="<<states
             <<" closure_states="<<closures<<" candidates="<<candidates<<" max_candidates="<<maxcand
             <<" lane_mapping=physical_positions prefix_balance=exact prefix_min_break=exact"
             <<" rank_delta_prefix_scan=exact shared_rank_queue_required=0 exact=1\n";
    return 0;
}
