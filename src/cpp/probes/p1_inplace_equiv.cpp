#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using MateID = uint64_t;
enum MateValue : uint8_t { N=0, R=1, L=2 };

static MateValue get(MateID m,int p){return MateValue((m>>(2*p))&3);}
static MateID setv(MateID m,int p,MateValue v){MateID mask=MateID(3)<<(2*p);return (m&~mask)|(MateID(v)<<(2*p));}
static MateID setpair(MateID m,int p,MateValue a,MateValue b){return setv(setv(m,p,a),p-1,b);}

static void gen_rec(int pos,int h,MateID m,std::vector<MateID>&out){
    if(pos<0){if(h==0)out.push_back(m);return;}
    gen_rec(pos-1,h,setv(m,pos,N),out);
    if(h>0)gen_rec(pos-1,h-1,setv(m,pos,R),out);
    gen_rec(pos-1,h+1,setv(m,pos,L),out);
}
static std::vector<MateID> states(int W){std::vector<MateID>o;gen_rec(W-1,1,0,o);return o;}

static MateID rr_close(MateID m,int W){
    m=setpair(m,1,N,N);int q=1,s=1;
    while(s){++q;if(q>=W)throw std::runtime_error("RR without matching L");auto v=get(m,q);if(v==L)--s;else if(v==R)++s;}
    return setv(m,q,R);
}

static uint32_t addm(uint32_t a,uint32_t b,uint32_t p){uint64_t z=uint64_t(a)+b;return uint32_t(z>=p?z-p:z);}

int main(){
    constexpr uint32_t MOD=4294967291u;
    std::mt19937_64 rng(0x1618);
    for(int W=3;W<=12;++W){
        auto ss=states(W);std::unordered_map<MateID,size_t> idx;idx.reserve(ss.size()*2);
        for(size_t i=0;i<ss.size();++i)idx.emplace(ss[i],i);
        for(int trial=0;trial<8;++trial){
            std::vector<uint32_t> old(ss.size()),ref,in;
            for(auto&x:old)x=uint32_t(rng()%MOD);
            ref=old;in=old;
            // Reference out-of-place p=1 main->main transition.
            for(size_t i=0;i<ss.size();++i){
                MateID m=ss[i];auto a=get(m,1),b=get(m,0);MateID t=m;bool emit=false;
                if(a==N&&b==N){t=setpair(m,1,L,R);emit=true;}
                else if(a==N&&b==R){t=setpair(m,1,R,N);emit=true;}
                else if(a==R&&b==N){t=setpair(m,1,N,R);emit=true;}
                else if(a==R&&b==R){t=rr_close(m,W);emit=true;}
                if(emit){auto it=idx.find(t);if(it==idx.end())throw std::runtime_error("target state missing");ref[it->second]=addm(ref[it->second],old[i],MOD);}
            }
            // In-place pairs pass.  NR owns the NR<->RN orbit even if old[NR]==0.
            for(size_t i=0;i<ss.size();++i){
                MateID m=ss[i];auto a=get(m,1),b=get(m,0);
                if(a==N&&b==N){MateID t=setpair(m,1,L,R);in[idx.at(t)]=addm(in[idx.at(t)],in[i],MOD);}
                else if(a==N&&b==R){MateID t=setpair(m,1,R,N);uint32_t z=addm(in[i],in[idx.at(t)],MOD);in[i]=z;in[idx.at(t)]=z;}
            }
            // Closure pass. RR sources and NN targets are untouched by the orbit pass.
            for(size_t i=0;i<ss.size();++i){MateID m=ss[i];if(get(m,1)==R&&get(m,0)==R){MateID t=rr_close(m,W);in[idx.at(t)]=addm(in[idx.at(t)],in[i],MOD);}}
            if(in!=ref){
                for(size_t i=0;i<ss.size();++i)if(in[i]!=ref[i]){std::cerr<<"mismatch W="<<W<<" trial="<<trial<<" i="<<i<<" ref="<<ref[i]<<" got="<<in[i]<<"\n";break;}
                return 1;
            }
        }
        std::cout<<"W="<<W<<" states="<<ss.size()<<" p1_inplace=OK\n";
    }
}
