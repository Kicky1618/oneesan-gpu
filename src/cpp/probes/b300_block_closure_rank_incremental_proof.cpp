#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace oneesan::gridfp;
using Code=std::uint64_t;
using Delta=std::int64_t;

namespace {
constexpr int MAXW=10;
struct Spec{int width=0;std::uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+3]{};};

bool allowed(std::uint32_t fixed,std::uint32_t occ,int pos,MateValue v){
    if(!((fixed>>pos)&1u))return v!=X;
    return ((occ>>pos)&1u)?(v==R||v==L):(v==N);
}
Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){
    Spec s;s.width=width;s.fixed=fixed;s.occ=occ;
    for(int h=0;h<=MAXW+2;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;
        for(int h=0;h<=MAXW+1;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h>0)x+=s.dp[w-1][h-1];x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}
    return s;
}
std::vector<MateID> gen_valid(int W){
    std::vector<MateID>out;auto rec=[&](auto&&self,int pos,int h,MateID m)->void{
        int rem=W-pos;if(h<0||h>rem)return;if(pos==W){if(h==0)out.push_back(m);return;}
        int bit=W-1-pos;self(self,pos+1,h,m);if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*bit)));self(self,pos+1,h+1,m|(MateID(L)<<(2*bit)));};
    rec(rec,0,1,0);return out;
}
bool in_spec(MateID m,int width,std::uint32_t fixed,std::uint32_t occ){
    for(int p=0;p<width;++p)if((fixed>>p)&1u){MateValue v=mget(m,p);bool o=(occ>>p)&1u;if(o?(v!=R&&v!=L):(v!=N))return false;}return true;
}
bool valid_mate(MateID m,int W){int h=1;for(int p=W-1;p>=0;--p){MateValue v=mget(m,p);if(v==R)--h;else if(v==L)++h;if(h<0)return false;}return h==0;}
Code rank_group(MateID m,const Spec&s){Code r=0;int h=1;for(int p=s.width-1;p>=0;--p){MateValue v=mget(m,p);if(v>N&&allowed(s.fixed,s.occ,p,N))r+=s.dp[p][h];if(v>R&&h>0&&allowed(s.fixed,s.occ,p,R))r+=s.dp[p][h-1];if(v==R)--h;else if(v==L)++h;}return r;}
Code contrib(MateValue v,int p,int h,const Spec&s){Code z=0;if(v>N&&allowed(s.fixed,s.occ,p,N))z+=s.dp[p][h];if(v>R&&h>0&&allowed(s.fixed,s.occ,p,R))z+=s.dp[p][h-1];return z;}
int height_before(MateID m,int W,int p){int h=1;for(int q=W-1;q>p;--q){MateValue v=mget(m,q);if(v==R)--h;else if(v==L)++h;}return h;}
int advance(int h,MateValue v){return h+(v==L)-(v==R);}
Code add_delta(Code r,Delta d){return d>=0?r+Code(d):r-Code(-d);}

bool forced_block_dest(MateID m,int W,int p,MateID&b){
    MateID t=m;switch(mpair(m,p)){
    case NR:case NL:b=mshrink(m,p);return true;
    case LL:{t=msetpair(m,p,NN);int q=closure_match_left(t,p);if(q<0)return false;t=mset(t,q,L);b=mshrink(t,p-1);return true;}
    case RR:{t=msetpair(m,p,NN);int q=closure_match_right(t,W,p);if(q<0)return false;t=mset(t,q,R);b=mshrink(t,p-1);return true;}
    case RL:t=msetpair(m,p,NN);b=mshrink(t,p-1);return true;
    default:return false;}
}
std::vector<int> candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
void masks(int hi,int lo,const std::vector<int>&fp,std::uint32_t group,std::uint32_t&mf,std::uint32_t&mo,std::uint32_t&bf,std::uint32_t&bo){
    mf=mo=bf=bo=0;for(std::size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=q<lo-1?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
}

int main(){
    std::uint64_t blocked_bases=0,left_candidates=0,right_candidates=0,endpoint_steps=0;
    for(int W=4;W<=MAXW;++W){
        auto main_all=gen_valid(W),block_all=gen_valid(W-1);
        for(int hi=W-1;hi>=2;--hi)for(int lo=1;lo<=hi;++lo){
            auto cand=candidates(W,hi,lo);int klim=std::min<int>(4,cand.size());
            for(int k=0;k<=klim;++k){std::vector<int>fp(cand.begin(),cand.begin()+k);
                for(std::uint32_t g=0;g<(1u<<k);++g){
                    std::uint32_t mf,mo,bf,bo;masks(hi,lo,fp,g,mf,mo,bf,bo);Spec ms=make_spec(W,mf,mo),bs=make_spec(W-1,bf,bo);
                    std::unordered_map<MateID,Code>mi;std::vector<MateID>bv;
                    for(MateID m:main_all)if(in_spec(m,W,mf,mo))mi.emplace(m,rank_group(m,ms));
                    for(MateID b:block_all)if(in_spec(b,W-1,bf,bo))bv.push_back(b);
                    for(int p=std::max(2,lo);p<=hi;++p)for(MateID b:bv){
                        if(mget(b,p-1)!=N)continue;MateID d=minsert(b,p-1,N);auto dit=mi.find(d);if(dit==mi.end())continue;++blocked_bases;Code base_rank=dit->second;
                        const int H=height_before(d,W,p);
                        Delta ldelta=Delta(contrib(L,p,H,ms))-Delta(contrib(N,p,H,ms));
                        ldelta+=Delta(contrib(L,p-1,H+1,ms))-Delta(contrib(N,p-1,H,ms));
                        int hb=H,bal=0;std::uint32_t endpoints=mate_non_n_mask(d,W);
                        std::uint32_t left=endpoints&((std::uint32_t(1)<<(p-1))-1u);
                        while(left){int q=mate_msb_index32(left);MateValue v=mget(d,q);++endpoint_steps;
                            if(bal==0&&v==L){Delta xdelta=ldelta+Delta(contrib(R,q,hb+2,ms))-Delta(contrib(L,q,hb,ms));MateID x=msetpair(d,p,LL);x=mset(x,q,R);auto it=mi.find(x);if(it==mi.end()||!valid_mate(x,W))return 10;MateID got=0;if(!forced_block_dest(x,W,p,got)||got!=b)return 11;if(add_delta(base_rank,xdelta)!=it->second)return 12;++left_candidates;}
                            ldelta+=Delta(contrib(v,q,hb+2,ms))-Delta(contrib(v,q,hb,ms));hb=advance(hb,v);if(v==L)++bal;else --bal;left^=std::uint32_t(1)<<q;if(bal<0)break;}

                        Delta rsuffix=Delta(contrib(R,p,H+2,ms))-Delta(contrib(N,p,H,ms));
                        rsuffix+=Delta(contrib(R,p-1,H+1,ms))-Delta(contrib(N,p-1,H,ms));
                        int hbelow=H;bal=0;std::uint32_t right=endpoints&~((std::uint32_t(1)<<(p+1))-1u);
                        while(right){int q=mate_lsb_index32(right);MateValue v=mget(d,q);++endpoint_steps;int hq=hbelow+(v==R)-(v==L);
                            if(bal==0&&v==R){Delta xdelta=Delta(contrib(L,q,hq,ms))-Delta(contrib(R,q,hq,ms))+rsuffix;MateID x=msetpair(d,p,RR);x=mset(x,q,L);auto it=mi.find(x);if(it==mi.end()||!valid_mate(x,W))return 20;MateID got=0;if(!forced_block_dest(x,W,p,got)||got!=b)return 21;if(add_delta(base_rank,xdelta)!=it->second)return 22;++right_candidates;}
                            rsuffix+=Delta(contrib(v,q,hq+2,ms))-Delta(contrib(v,q,hq,ms));hbelow=hq;if(v==R)++bal;else --bal;right&=right-1u;if(bal<0)break;}
                    }
                }
            }
        }
    }
    if(!left_candidates||!right_candidates)return 30;
    std::cout<<"b300-block-closure-rank-incremental-proof OK width_max="<<MAXW
             <<" blocked_bases="<<blocked_bases<<" endpoint_steps="<<endpoint_steps
             <<" left_candidates="<<left_candidates<<" right_candidates="<<right_candidates
             <<" repeated_rank_same_scans=0 incremental_rank_delta=1 grouped_masks=1 exact=1\n";
    return 0;
}
