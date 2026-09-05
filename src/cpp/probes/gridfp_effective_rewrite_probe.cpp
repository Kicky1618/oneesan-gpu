#include "../../common/gridfp_transition.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace oneesan::gridfp;

static bool valid_motzkin(MateID m,int W){
    int h=1;
    for(int p=W-1;p>=0;--p){
        auto v=mget(m,p);
        if(v==R){ if(h==0)return false; --h; }
        else if(v==L) ++h;
        else if(v!=N) return false;
    }
    return h==0;
}
static std::vector<int> heights(MateID m,int W){
    std::vector<int> z;z.reserve(W+1);int h=1;z.push_back(h);
    for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;z.push_back(h);}return z;
}
static const char* pname(MateValuePair q){
    switch(q){case NN:return"NN";case NR:return"NR";case NL:return"NL";case RN:return"RN";
    case RR:return"RR";case RL:return"RL";case LN:return"LN";case LR:return"LR";case LL:return"LL";default:return"XX";}
}

static bool expected_effective(MateID m,int W,int p,MateID& out){
    MateID t=m;
    switch(mpair(m,p)){
    case NN: out=msetpair(m,p,LR); return true;
    case NR: out=msetpair(m,p,RN); return true;
    case NL: out=msetpair(m,p,LN); return true;
    case RN: out=msetpair(m,p,NR); return true;
    case LN: out=msetpair(m,p,NL); return true;
    case RL: out=msetpair(m,p,NN); return true;
    case LL:{
        t=msetpair(m,p,NN);int q=p-1,s=1;
        while(s){--q;if(q<0)return false;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}
        out=mset(t,q,L);return true;
    }
    case RR:{
        t=msetpair(m,p,NN);int q=p,s=1;
        while(s){++q;if(q>=W)return false;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}
        out=mset(t,q,R);return true;
    }
    default:return false;
    }
}

struct Totals{
    uint64_t words=0, branches=0, invalid=0, blocked=0;
    std::array<uint64_t,16> bypair{};
};

static bool check_word(MateID m,int W,Totals& z){
    ++z.words;auto hin=heights(m,W);
    for(int p=W-1;p>=1;--p){
        auto pair=mpair(m,p);auto got=include_horizontal(m,W,p);MateID exp=0;
        bool ev=expected_effective(m,W,p,exp);
        if(bool(got.valid)!=ev){
            std::cerr<<"valid mismatch W="<<W<<" p="<<p<<" pair="<<pname(pair)<<"\n";return false;
        }
        if(!got.valid){++z.invalid;continue;}
        ++z.branches;++z.bypair[unsigned(pair)];
        MateID eff=got.mate;
        if(got.blocked){
            ++z.blocked;
            if(p<=1){std::cerr<<"blocked at p1\n";return false;}
            eff=blocked_exclude(got.mate,p-1);
        }
        if(eff!=exp){
            std::cerr<<"rewrite mismatch W="<<W<<" p="<<p<<" pair="<<pname(pair)
                     <<" blocked="<<got.blocked<<"\n";return false;
        }
        if(!valid_motzkin(eff,W)){
            std::cerr<<"invalid output W="<<W<<" p="<<p<<" pair="<<pname(pair)<<"\n";return false;
        }
        auto hout=heights(eff,W);int cut=W-p;
        for(int k=0;k<=W;++k){
            int d=hout[k]-hin[k];int allow=(k==cut)?1:0;
            if(d>allow){
                std::cerr<<"cut bound fail W="<<W<<" p="<<p<<" pair="<<pname(pair)
                         <<" k="<<k<<" diff="<<d<<" allow="<<allow<<"\n";return false;
            }
            if((pair==LL||pair==RR)&&d>0){
                std::cerr<<"closure increased prefix W="<<W<<" p="<<p<<" pair="<<pname(pair)<<"\n";return false;
            }
        }
    }
    return true;
}

static bool enumerate_rec(int W,int idx,int h,MateID m,Totals& z){
    if(idx==W){if(h!=0)return true;return check_word(m,W,z);}
    int p=W-1-idx;
    if(!enumerate_rec(W,idx+1,h,m,z))return false;
    if(h>0 && !enumerate_rec(W,idx+1,h-1,m|(MateID(R)<<(2*p)),z))return false;
    if(!enumerate_rec(W,idx+1,h+1,m|(MateID(L)<<(2*p)),z))return false;
    return true;
}

int main(int argc,char**argv){
    int maxW=argc>1?std::atoi(argv[1]):16;
    if(maxW<2||maxW>28)return 2;
    Totals all;
    for(int W=2;W<=maxW;++W){
        Totals z;
        if(!enumerate_rec(W,0,1,0,z))return 1;
        std::cout<<"W="<<W<<" words="<<z.words<<" branches="<<z.branches
                 <<" blocked="<<z.blocked<<" invalid="<<z.invalid;
        for(auto q:{NN,NR,NL,RN,RR,RL,LN,LL})std::cout<<" "<<pname(q)<<"="<<z.bypair[unsigned(q)];
        std::cout<<" exact=1\n";
        all.words+=z.words;all.branches+=z.branches;all.blocked+=z.blocked;all.invalid+=z.invalid;
        for(int i=0;i<16;++i)all.bypair[i]+=z.bypair[i];
    }
    std::cout<<"gridfp_effective_rewrite maxW="<<maxW<<" words="<<all.words
             <<" branches="<<all.branches<<" blocked="<<all.blocked<<" exact=1\n";
}
