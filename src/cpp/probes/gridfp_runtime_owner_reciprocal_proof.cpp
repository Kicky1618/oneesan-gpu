#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
using Rank64 = std::uint64_t;
constexpr int MAX_W = 28;
constexpr std::array<Rank64,11> TOTAL = {
    632ULL,4451ULL,32427ULL,242413ULL,1849269ULL,14339193ULL,
    112685373ULL,895517316ULL,7184644894ULL,58113695597ULL,
    473397057701ULL
};
constexpr std::array<Rank64,11> MAGIC = {
    29187886192578405ULL,4144404420065054ULL,568869894646732ULL,
    76096348272204ULL,9975154546856ULL,1286456223423ULL,
    163701317950ULL,20598980885ULL,2567523427ULL,317425074ULL,
    38966749ULL
};

Rank64 binom(int n,int k){if(n<0||k<0||k>n)return 0;Rank64 x=1;for(int i=1;i<=k;++i)x=x*Rank64(n-k+i)/Rank64(i);return x;}
std::array<std::array<Rank64,MAX_W+2>,MAX_W+1> primitive_table(){std::array<std::array<Rank64,MAX_W+2>,MAX_W+1>p{};p[0][0]=1;for(int rem=1;rem<=MAX_W;++rem)for(int h=0;h<=MAX_W;++h)p[rem][h]=p[rem-1][h+1]+(h?p[rem-1][h-1]:0);return p;}
Rank64 group_size(const std::array<std::array<Rank64,MAX_W+2>,MAX_W+1>&p,int L,int r){Rank64 s=0;for(int l=0;l<=L;++l){int occ=r+l;if(!(occ&1))continue;s+=(binom(L,l)+binom(L-2,l-1))*p[occ][1];}return s;}
Rank64 fast_div(Rank64 x,Rank64 d,Rank64 magic){if(d==1)return x;Rank64 q=Rank64((__uint128_t(x)*magic)>>64);const __uint128_t prod=__uint128_t(q)*d;if(prod>x)--q;return q;}
}

int main(){
    const auto p=primitive_table();
    std::uint64_t groups=0, owner_cases=0;
    Rank64 max_numerator=0;
    for(int wi=0;wi<11;++wi){
        const int W=8+2*wi;const int L=W/2+1;const int O=W-L;
        Rank64 total=0;
        for(int r=0;r<=O;++r) total+=binom(O,r)*group_size(p,L,r);
        if(total!=TOTAL[wi]){std::cerr<<"total mismatch W="<<W<<" got="<<total<<" table="<<TOTAL[wi]<<'\n';return 2;}
        const Rank64 expected_magic=std::numeric_limits<Rank64>::max()/total+1;
        if(expected_magic!=MAGIC[wi]){std::cerr<<"magic mismatch W="<<W<<'\n';return 3;}
        Rank64 prefix=0;
        for(int r=0;r<=O;++r){
            const Rank64 group=group_size(p,L,r);const Rank64 count=binom(O,r);
            for(Rank64 sr=0;sr<count;++sr){
                const Rank64 base=prefix+sr*group;const Rank64 midpoint=base+group/2;++groups;
                for(int ngpu=2;ngpu<=16;++ngpu){
                    const Rank64 numerator=midpoint*Rank64(ngpu);
                    if(numerator>max_numerator)max_numerator=numerator;
                    const Rank64 exact=numerator/total;
                    const Rank64 fast=fast_div(numerator,total,MAGIC[wi]);
                    if(fast!=exact||fast>=Rank64(ngpu)){
                        std::cerr<<"owner mismatch W="<<W<<" r="<<r<<" sr="<<sr<<" ngpu="<<ngpu<<" numerator="<<numerator<<" exact="<<exact<<" fast="<<fast<<'\n';return 4;
                    }
                    ++owner_cases;
                }
            }
            prefix+=count*group;
        }
        if(prefix!=total)return 5;
    }
    if(groups!=16376ULL||owner_cases!=245640ULL||max_numerator>=Rank64(1)<<63)return 6;
    std::cout<<"gridfp-runtime-owner-reciprocal-proof OK"
             <<" W_min=8 W_max=28 W_step=2"
             <<" groups="<<groups
             <<" owner_cases="<<owner_cases
             <<" max_numerator="<<max_numerator
             <<" table_entries=11 table_bytes="<<(TOTAL.size()+MAGIC.size())*sizeof(Rank64)
             <<" quotient_error_bound=1 exact=1\n";
    return 0;
}
