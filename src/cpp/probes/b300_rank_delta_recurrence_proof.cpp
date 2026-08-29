#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace oneesan::gridfp;
using Code = std::uint64_t;
using SCode = std::int64_t;

namespace {
constexpr int MAXW = 10;
struct Spec {
    int width=0; std::uint32_t fixed=0,occ=0;
    Code dp[MAXW+1][MAXW+3]{};
};

bool allowed(std::uint32_t fixed,std::uint32_t occ,int pos,MateValue v){
    if(!((fixed>>pos)&1u)) return v!=X;
    return ((occ>>pos)&1u)?(v==R||v==L):(v==N);
}

Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){
    Spec s;s.width=width;s.fixed=fixed;s.occ=occ;
    for(int h=0;h<=MAXW+2;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){
        int pos=w-1; bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;
        for(int h=0;h<=MAXW+1;++h){
            Code x=0;
            if(!f||!o)x+=s.dp[w-1][h];
            if(!f||o){if(h>0)x+=s.dp[w-1][h-1];x+=s.dp[w-1][h+1];}
            s.dp[w][h]=x;
        }
    }
    return s;
}

std::vector<MateID> gen_valid(int W){
    std::vector<MateID> out;
    auto rec=[&](auto&& self,int pos,int h,MateID m)->void{
        int rem=W-pos;if(h<0||h>rem)return;
        if(pos==W){if(h==0)out.push_back(m);return;}
        int bit=W-1-pos;
        self(self,pos+1,h,m);
        if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*bit)));
        self(self,pos+1,h+1,m|(MateID(L)<<(2*bit)));
    };
    rec(rec,0,1,0);return out;
}

bool in_spec(MateID m,int width,std::uint32_t fixed,std::uint32_t occ){
    for(int p=0;p<width;++p)if((fixed>>p)&1u){
        MateValue v=mget(m,p); bool o=(occ>>p)&1u;
        if(o?(v!=R&&v!=L):(v!=N))return false;
    }
    return true;
}

Code rank_group(MateID m,const Spec&s){
    Code rank=0;int h=1;
    for(int pos=s.width-1;pos>=0;--pos){
        MateValue v=mget(m,pos);
        if(v>N&&allowed(s.fixed,s.occ,pos,N))rank+=s.dp[pos][h];
        if(v>R&&h>0&&allowed(s.fixed,s.occ,pos,R))rank+=s.dp[pos][h-1];
        if(v==R)--h;else if(v==L)++h;
    }
    return rank;
}

int height_before(MateID m,int width,int pos){
    int h=1;
    for(int q=width-1;q>pos;--q){MateValue v=mget(m,q);if(v==R)--h;else if(v==L)++h;}
    return h;
}

SCode contrib_delta(MateValue v,int p,int h,const Spec&ms,const Spec&bs){
    Code a=0,b=0;
    if(v>N&&allowed(ms.fixed,ms.occ,p,N))a+=ms.dp[p][h];
    if(v>R&&h>0&&allowed(ms.fixed,ms.occ,p,R))a+=ms.dp[p][h-1];
    int q=p-1;
    if(v>N&&allowed(bs.fixed,bs.occ,q,N))b+=bs.dp[q][h];
    if(v>R&&h>0&&allowed(bs.fixed,bs.occ,q,R))b+=bs.dp[q][h-1];
    return SCode(b)-SCode(a);
}

SCode prefix_delta_full(MateID full,int W,int p,const Spec&ms,const Spec&bs){
    Code a=0,b=0;int h=1;
    for(int pos=W-1;pos>p;--pos){
        MateValue v=mget(full,pos);
        if(v>N&&allowed(ms.fixed,ms.occ,pos,N))a+=ms.dp[pos][h];
        if(v>R&&h>0&&allowed(ms.fixed,ms.occ,pos,R))a+=ms.dp[pos][h-1];
        int q=pos-1;
        if(v>N&&allowed(bs.fixed,bs.occ,q,N))b+=bs.dp[q][h];
        if(v>R&&h>0&&allowed(bs.fixed,bs.occ,q,R))b+=bs.dp[q][h-1];
        if(v==R)--h;else if(v==L)++h;
    }
    return SCode(b)-SCode(a);
}

Code add_signed(Code x,SCode d){return d>=0?x+Code(d):x-Code(-d);}
Code sub_signed(Code x,SCode d){return d>=0?x-Code(d):x+Code(-d);}

std::vector<int> candidates(int W,int hi,int lo){
    std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;
}
void masks(int hi,int lo,const std::vector<int>&fp,std::uint32_t group,
           std::uint32_t&mf,std::uint32_t&mo,std::uint32_t&bf,std::uint32_t&bo){
    mf=mo=bf=bo=0;
    for(std::size_t i=0;i<fp.size();++i){
        int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;
        int bq=q<lo-1?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;
    }
}
}

int main(){
    std::uint64_t recurrence_main=0,recurrence_block=0,drop_exact=0,lift_exact=0;
    for(int W=4;W<=MAXW;++W){
        const auto main_all=gen_valid(W),block_all=gen_valid(W-1);
        for(int hi=W-1;hi>=2;--hi)for(int lo=2;lo<=hi;++lo){
            auto cand=candidates(W,hi,lo);int klim=std::min<int>(4,cand.size());
            for(int k=0;k<=klim;++k){
                std::vector<int>fp(cand.begin(),cand.begin()+k);
                for(std::uint32_t g=0;g<(1u<<k);++g){
                    std::uint32_t mf,mo,bf,bo;masks(hi,lo,fp,g,mf,mo,bf,bo);
                    Spec ms=make_spec(W,mf,mo),bs=make_spec(W-1,bf,bo);
                    std::vector<MateID>mv,bv;
                    for(MateID m:main_all)if(in_spec(m,W,mf,mo))mv.push_back(m);
                    for(MateID b:block_all)if(in_spec(b,W-1,bf,bo))bv.push_back(b);
                    std::unordered_map<MateID,Code>mi,bi;
                    for(Code i=0;i<mv.size();++i){if(rank_group(mv[i],ms)!=i)return 2;mi[mv[i]]=i;}
                    for(Code i=0;i<bv.size();++i){if(rank_group(bv[i],bs)!=i)return 3;bi[bv[i]]=i;}
                    for(int p=hi;p>=lo;--p){
                        for(Code i=0;i<mv.size();++i){
                            MateID m=mv[i];SCode d=prefix_delta_full(m,W,p,ms,bs);
                            if(p>1){
                                int h=height_before(m,W,p);SCode dn=d+contrib_delta(mget(m,p),p,h,ms,bs);
                                if(dn!=prefix_delta_full(m,W,p-1,ms,bs))return 4;
                                ++recurrence_main;
                            }
                            if(mget(m,p)==N){MateID b=mshrink(m,p);auto it=bi.find(b);if(it!=bi.end()){
                                if(add_signed(i,d)!=it->second)return 5;++drop_exact;
                            }}
                        }
                        for(Code i=0;i<bv.size();++i){
                            MateID b=bv[i];MateID full=minsert(b,p,N);SCode d=prefix_delta_full(full,W,p,ms,bs);
                            if(p>1){
                                int h=height_before(b,W-1,p-1);MateValue v=mget(b,p-1);
                                SCode dn=d+contrib_delta(v,p,h,ms,bs);
                                MateID full_next=minsert(b,p-1,N);
                                if(dn!=prefix_delta_full(full_next,W,p-1,ms,bs))return 6;
                                ++recurrence_block;
                            }
                            auto it=mi.find(full);if(it!=mi.end()){
                                if(sub_signed(i,d)!=it->second)return 7;++lift_exact;
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout<<"b300-rank-delta-recurrence-proof OK width_max="<<MAXW
             <<" main_recurrence_cases="<<recurrence_main
             <<" block_recurrence_cases="<<recurrence_block
             <<" drop_exact_cases="<<drop_exact
             <<" lift_exact_cases="<<lift_exact
             <<" signed_delta=1 step_O1=1 exact=1\n";
    return 0;
}
