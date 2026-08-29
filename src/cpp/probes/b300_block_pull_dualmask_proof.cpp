#include <cstdint>
#include <iostream>
#include <random>

using MateID=std::uint64_t;
namespace{
constexpr MateID EVEN=0x5555555555555555ULL;
std::uint32_t compact_even(MateID x){
    x&=EVEN;
    x=(x|(x>>1))&0x3333333333333333ULL;
    x=(x|(x>>2))&0x0f0f0f0f0f0f0f0fULL;
    x=(x|(x>>4))&0x00ff00ff00ff00ffULL;
    x=(x|(x>>8))&0x0000ffff0000ffffULL;
    x=(x|(x>>16))&0x00000000ffffffffULL;
    return std::uint32_t(x);
}
std::uint32_t endpoint_old(MateID m){
    MateID x=(m|(m>>1))&EVEN;
    return compact_even(x)&((std::uint32_t(1)<<28)-1u);
}
struct Masks{std::uint32_t r,l;};
Masks dual(MateID m){
    const MateID lo=m&EVEN,hi=(m>>1)&EVEN;
    return{compact_even(lo&~hi)&((std::uint32_t(1)<<28)-1u),
           compact_even(hi&~lo)&((std::uint32_t(1)<<28)-1u)};
}
int get(MateID m,int q){return int((m>>(2*q))&3ULL);}
}
int main(){
    // compact_even is an OR/shift/mask network. Verifying every input basis
    // vector proves every combination because the network distributes over OR.
    for(int q=0;q<28;++q){
        MateID x=MateID(1)<<(2*q);
        if(compact_even(x)!=(std::uint32_t(1)<<q))return 2;
    }
    // Exact per-symbol truth table for N=00,R=01,L=10,X=11. Production mates
    // contain N/R/L; X is explicitly excluded from both directional masks.
    for(int q=0;q<28;++q)for(int v=0;v<4;++v){
        MateID m=MateID(v)<<(2*q);Masks z=dual(m);std::uint32_t b=std::uint32_t(1)<<q;
        if(bool(z.r&b)!=(v==1))return 3;
        if(bool(z.l&b)!=(v==2))return 4;
    }
    // Combined-state check across the full 28-cell width catches accidental
    // carry/lane interactions if the compaction network is later edited.
    std::mt19937_64 rng(0xB300D00DULL);
    for(int t=0;t<1000000;++t){
        MateID m=0;
        for(int q=0;q<28;++q)m|=MateID(rng()%3)<<(2*q);
        Masks z=dual(m);std::uint32_t rr=0,ll=0;
        for(int q=0;q<28;++q){int v=get(m,q);if(v==1)rr|=1u<<q;else if(v==2)ll|=1u<<q;}
        if(z.r!=rr||z.l!=ll)return 5;
        if((z.r|z.l)!=endpoint_old(m))return 6;
    }
    std::cout<<"b300-block-pull-dualmask-proof OK basis28=1 symbol_truth_nrlx=1 valid_ternary_random=1000000 endpoint_union_exact=1 exact=1\n";
}
