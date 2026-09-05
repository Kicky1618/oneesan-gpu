#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_transition_reverse.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace {
using namespace oneesan::gridfp;

struct Key {
    MateID mate = 0;
    bool blocked = false;
    bool operator<(const Key& o) const {
        return blocked != o.blocked ? blocked < o.blocked : mate < o.mate;
    }
    bool operator==(const Key& o) const { return mate == o.mate && blocked == o.blocked; }
};

Key mirror_key(Key k, int W) {
    return Key{mirror_mate(k.mate, k.blocked ? W - 1 : W), k.blocked};
}

std::set<Key> low_direct_inverse(MateID d, int W) {
    std::set<Key> out;
    out.insert(Key{d, false});
    const MateValuePair pair = mpair(d, 1);
    if (pair == LR) out.insert(Key{msetpair(d, 1, NN), false});
    if (pair == RN) out.insert(Key{msetpair(d, 1, NR), false});
    if (pair == NR) out.insert(Key{msetpair(d, 1, RN), false});
    if (pair == NN) {
        int bal = 0;
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                out.insert(Key{x, false});
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }
    if (pair == NR) out.insert(Key{mshrink(d, 1), true});
    return out;
}

std::set<Key> old_high_inverse(MateID d, int W) {
    const MateID md = mirror_mate(d, W);
    const auto low = low_direct_inverse(md, W);
    std::set<Key> out;
    for (Key k : low) out.insert(mirror_key(k, W));
    return out;
}

std::set<Key> direct_high_inverse(MateID d, int W) {
    std::set<Key> out;
    out.insert(Key{d, false});
    const int p = W - 1;
    const MateValuePair pair = mpair(d, p);
    if (pair == LR) out.insert(Key{msetpair(d, p, NN), false});
    if (pair == NL) out.insert(Key{msetpair(d, p, LN), false});
    if (pair == LN) out.insert(Key{msetpair(d, p, NL), false});
    if (pair == NN) {
        int bal = 0;
        for (int q = p - 2; q >= 0; --q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == L) {
                MateID x = msetpair(d, p, LL);
                x = mset(x, q, R);
                out.insert(Key{x, false});
            }
            if (v == L) ++bal;
            else if (v == R) --bal;
            if (bal < 0) break;
        }
    }
    if (pair == LN) out.insert(Key{mshrink(d, p - 1), true});
    return out;
}

bool valid_mate(MateID m, int W) {
    int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const MateValue v = mget(m, W - 1 - pos);
        if (v == X) return false;
        if (v == R) --h; else if (v == L) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

std::vector<MateID> generate_valid(int W) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        const int rem = W - pos;
        if (h < 0 || h > rem) return;
        if (pos == W) { if (h == 0) out.push_back(m); return; }
        const int bit = W - 1 - pos;
        self(self, pos + 1, h, m);
        if (h > 0) self(self, pos + 1, h - 1, m | (MateID(R) << (2 * bit)));
        self(self, pos + 1, h + 1, m | (MateID(L) << (2 * bit)));
    };
    rec(rec, 0, 1, 0); return out;
}

using CountTable = std::array<std::array<std::uint64_t,31>,29>;
CountTable make_counts() {
    CountTable f{}; f[0][0]=1;
    for(int rem=1;rem<=28;++rem) for(int h=0;h<=29;++h){
        std::uint64_t z=f[rem-1][h];
        if(h>0) z+=f[rem-1][h-1];
        if(h<30) z+=f[rem-1][h+1];
        f[rem][h]=z;
    }
    return f;
}
MateID unrank_valid(int W,std::uint64_t rank,const CountTable& f){
    MateID m=0;int h=1;
    for(int pos=0;pos<W;++pos){
        const int rem=W-pos-1,bit=W-1-pos;
        const auto n=f[rem][h]; if(rank<n) continue; rank-=n;
        const auto r=h>0?f[rem][h-1]:0;
        if(rank<r){m|=MateID(R)<<(2*bit);--h;continue;}
        rank-=r;m|=MateID(L)<<(2*bit);++h;
    }
    return m;
}

void check(MateID d,int W,std::uint64_t& nn,std::uint64_t& ll,std::uint64_t& ln){
    if(!valid_mate(d,W)) std::exit(10);
    const auto a=old_high_inverse(d,W), b=direct_high_inverse(d,W);
    if(a!=b){std::cerr<<"mismatch W="<<W<<" mate="<<d<<'\n';std::exit(2);}
    const int p=W-1; const auto pair=mpair(d,p);
    if(pair==NN){
        ++nn; int bal=0;
        for(int q=p-2;q>=0;--q){
            const auto v=mget(d,q);
            if(bal==0&&v==L){
                MateID x=msetpair(d,p,LL);x=mset(x,q,R);
                if(!valid_mate(x,W)) std::exit(3);
                const auto z=include_horizontal_reverse(x,W,p);
                if(!z.valid||z.blocked||z.mate!=d) std::exit(4);
                ++ll;
            }
            if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;
        }
    }
    if(pair==LN){
        ++ln; const MateID bkey=mshrink(d,p-1);
        if(!valid_mate(bkey,W-1)||blocked_exclude_reverse(bkey,W,p)!=d) std::exit(5);
    }
}
}

int main(){
    std::uint64_t exhaustive=0,random=0,nn=0,ll=0,ln=0;
    for(int W=2;W<=12;++W) for(MateID d:generate_valid(W)){check(d,W,nn,ll,ln);++exhaustive;}
    const auto f=make_counts(); if(f[28][1]!=385719506620ULL) return 7;
    std::mt19937_64 rng(0x68696768696e7631ULL);
    constexpr std::uint64_t RANDOM=1000000;
    for(std::uint64_t i=0;i<RANDOM;++i){check(unrank_valid(28,rng()%f[28][1],f),28,nn,ll,ln);++random;}
    if(!nn||!ll||!ln) return 8;
    std::cout<<"gridfp-runtime-turn-direct-high-compress-inverse-proof OK"
             <<" exhaustive_width_max=12 exhaustive_cases="<<exhaustive
             <<" random_width=28 random_cases="<<random
             <<" nn_cases="<<nn<<" ll_candidates="<<ll<<" ln_cases="<<ln
             <<" local_direct=LR_NN,NL_LN,LN_NL"
             <<" nn_ll_scan_candidates_direct=1 blocked_LN_direct=1"
             <<" destination_mirror_passes=0 source_mirror_passes=0"
             <<" inverse_set_exact=1\n";
    return 0;
}
