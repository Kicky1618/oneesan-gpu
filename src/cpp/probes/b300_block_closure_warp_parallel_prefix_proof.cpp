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
Code rank_group(MateID m,const Spec&s){Code r=0;int h=1;for(int p=s.width-1;p>=0;--p){MateValue v=mget(m,p);if(v>N&&allowed(s.fixed,s.occ,p,N))r+=s.dp[p][h];if(v>R&&h>0&&allowed(s.fixed,s.occ,p,R))r+=s.dp[p][h-1];if(v==R)--h;else if(v==L)++h;}return r;}
Code contrib(MateValue v,int p,int h,const Spec&s){Code z=0;if(v>N&&allowed(s.fixed,s.occ,p,N))z+=s.dp[p][h];if(v>R&&h>0&&allowed(s.fixed,s.occ,p,R))z+=s.dp[p][h-1];return z;}
int height_before(MateID m,int W,int p){int h=1;for(int q=W-1;q>p;--q){MateValue v=mget(m,q);if(v==R)--h;else if(v==L)++h;}return h;}
Code add_delta(Code r,Delta d){return d>=0?r+Code(d):r-Code(-d);}

struct Term{int kind=0,q=-1;Delta d=0;};
bool operator==(const Term&a,const Term&b){return a.kind==b.kind&&a.q==b.q&&a.d==b.d;}
bool term_less(const Term&a,const Term&b){return a.kind!=b.kind?a.kind<b.kind:a.q<b.q;}

std::vector<Term> serial_terms(MateID d,int W,int p,int H,const Spec&ms){
    std::vector<Term> out;
    if(H>0)out.push_back({0,p,Delta(contrib(R,p,H,ms))+Delta(contrib(L,p-1,H-1,ms))});
    std::uint32_t endpoints=mate_non_n_mask(d,W);
    Delta ldelta=Delta(contrib(L,p,H,ms))+Delta(contrib(L,p-1,H+1,ms));
    int hb=H,bal=0;std::uint32_t left=endpoints&((std::uint32_t(1)<<(p-1))-1u);
    while(left){int q=mate_msb_index32(left);MateValue v=mget(d,q);
        if(bal==0&&v==L){Delta x=ldelta+Delta(contrib(R,q,hb+2,ms))-Delta(contrib(L,q,hb,ms));out.push_back({1,q,x});}
        ldelta+=Delta(contrib(v,q,hb+2,ms))-Delta(contrib(v,q,hb,ms));hb+=(v==L)-(v==R);if(v==L)++bal;else --bal;left^=std::uint32_t(1)<<q;if(bal<0)break;}
    Delta rs=Delta(contrib(R,p,H+2,ms))+Delta(contrib(R,p-1,H+1,ms));
    int hbelow=H;bal=0;std::uint32_t right=endpoints&~((std::uint32_t(1)<<(p+1))-1u);
    while(right){int q=mate_lsb_index32(right);MateValue v=mget(d,q);int hq=hbelow+(v==R)-(v==L);
        if(bal==0&&v==R){Delta x=Delta(contrib(L,q,hq,ms))-Delta(contrib(R,q,hq,ms))+rs;out.push_back({2,q,x});}
        rs+=Delta(contrib(v,q,hq+2,ms))-Delta(contrib(v,q,hq,ms));hbelow=hq;if(v==R)++bal;else --bal;right&=right-1u;if(bal<0)break;}
    std::sort(out.begin(),out.end(),term_less);return out;
}

std::vector<Term> parallel_prefix_terms(MateID d,int W,int p,int H,const Spec&ms){
    int delta[32]{},pref[32]{},pref_prev[32]{};
    for(int q=0;q<32;++q){MateValue v=q<W?mget(d,q):N;delta[q]=(v==L)-(v==R);pref[q]=delta[q]+(q?pref[q-1]:0);pref_prev[q]=q?pref[q-1]:0;}
    const int pp2=p>=2?pref[p-2]:0,pp=pref[p];
    bool lbar[32]{},rbar[32]{};Delta ld[32]{},rd[32]{},lp[32]{},rp[32]{};
    for(int q=0;q<32;++q){MateValue v=q<W?mget(d,q):N;
        if(q<p-1){int before=pp2-pref[q],after=pp2-pref_prev[q];lbar[q]=after<0;int hb=H+before;if(v==R||v==L)ld[q]=Delta(contrib(v,q,hb+2,ms))-Delta(contrib(v,q,hb,ms));}
        if(q>p&&q<W){int before=-(pref_prev[q]-pp),after=-(pref[q]-pp);rbar[q]=after<0;int hq=H+after;if(v==R||v==L)rd[q]=Delta(contrib(v,q,hq+2,ms))-Delta(contrib(v,q,hq,ms));}
        lp[q]=ld[q]+(q?lp[q-1]:0);rp[q]=rd[q]+(q?rp[q-1]:0);
    }
    std::vector<Term> out;
    if(H>0)out.push_back({0,p,Delta(contrib(R,p,H,ms))+Delta(contrib(L,p-1,H-1,ms))});
    const Delta l0=Delta(contrib(L,p,H,ms))+Delta(contrib(L,p-1,H+1,ms));
    for(int q=0;q<p-1;++q){MateValue v=mget(d,q);const int before=pp2-pref[q];bool blocked=false;for(int r=q+1;r<p-1;++r)blocked|=lbar[r];
        if(v==L&&before==0&&!blocked){const int hb=H+before;const Delta span=lp[p-2]-lp[q];const Delta x=l0+span+Delta(contrib(R,q,hb+2,ms))-Delta(contrib(L,q,hb,ms));out.push_back({1,q,x});}}
    const Delta r0=Delta(contrib(R,p,H+2,ms))+Delta(contrib(R,p-1,H+1,ms));
    for(int q=p+1;q<W;++q){MateValue v=mget(d,q);const int before=-(pref_prev[q]-pp);bool blocked=false;for(int r=p+1;r<q;++r)blocked|=rbar[r];
        if(v==R&&before==0&&!blocked){const int hq=H-(pref[q]-pp);const Delta span=rp[q-1]-rp[p];const Delta x=r0+span+Delta(contrib(L,q,hq,ms))-Delta(contrib(R,q,hq,ms));out.push_back({2,q,x});}}
    std::sort(out.begin(),out.end(),term_less);return out;
}

std::vector<int> candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
void masks(int hi,int lo,const std::vector<int>&fp,std::uint32_t group,std::uint32_t&mf,std::uint32_t&mo,std::uint32_t&bf,std::uint32_t&bo){
    mf=mo=bf=bo=0;for(std::size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=q<lo-1?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
}

int main(){
    std::uint64_t states=0,terms=0,ll=0,rr=0,rl=0;
    for(int W=4;W<=MAXW;++W){auto main_all=gen_valid(W),block_all=gen_valid(W-1);
        for(int hi=W-1;hi>=2;--hi)for(int lo=1;lo<=hi;++lo){auto cand=candidates(W,hi,lo);int klim=std::min<int>(4,cand.size());
            for(int k=0;k<=klim;++k){std::vector<int>fp(cand.begin(),cand.begin()+k);for(std::uint32_t g=0;g<(1u<<k);++g){
                std::uint32_t mf,mo,bf,bo;masks(hi,lo,fp,g,mf,mo,bf,bo);Spec ms=make_spec(W,mf,mo);std::unordered_map<MateID,Code>mi;for(MateID m:main_all)if(in_spec(m,W,mf,mo))mi.emplace(m,rank_group(m,ms));
                for(int p=std::max(2,lo);p<=hi;++p)for(MateID b:block_all){if(!in_spec(b,W-1,bf,bo)||mget(b,p-1)!=N)continue;MateID d=minsert(b,p-1,N);auto base=mi.find(d);if(base==mi.end())continue;const int H=height_before(d,W,p);
                    auto a=serial_terms(d,W,p,H,ms),z=parallel_prefix_terms(d,W,p,H,ms);if(a!=z){std::cerr<<"mismatch W="<<W<<" p="<<p<<" b="<<b<<" serial="<<a.size()<<" parallel="<<z.size()<<'\n';return 2;}
                    for(auto const&t:z){Code r=add_delta(base->second,t.d);bool found=false;for(auto const&kv:mi)if(kv.second==r){found=true;break;}if(!found){std::cerr<<"rank outside group W="<<W<<" p="<<p<<"\n";return 3;}++terms;if(t.kind==0)++rl;else if(t.kind==1)++ll;else ++rr;}++states;
                }
            }}
        }
    }
    if(!states||!ll||!rr||!rl)return 4;
    std::cout<<"b300-block-closure-warp-parallel-prefix-proof OK width_max="<<MAXW
             <<" states="<<states<<" terms="<<terms<<" rl="<<rl<<" ll="<<ll<<" rr="<<rr
             <<" serial_scan_equivalent=1 candidate_barrier_ballot_equivalent=1 rank_delta_prefix_scan_equivalent=1 exact=1\n";
    return 0;
}
