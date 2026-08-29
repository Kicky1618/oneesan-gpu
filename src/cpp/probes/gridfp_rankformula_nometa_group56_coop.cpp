#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

static uint32_t abstract_off56(int n, int h) {
    uint32_t off = 0;
    for (int nn = 0; nn <= L; ++nn) {
        for (int hh = 0; hh < 16; ++hh) {
            if (nn == n && hh == h) return off;
            off += ballot_suffix(nn, hh);
        }
    }
    return off;
}
static uint64_t pack56(uint32_t start,uint32_t n,uint32_t h,int delta,uint32_t count) {
    if(n<h||((n-h)&1u))std::exit(20);const uint32_t lc=(n-h)>>1,aoff=abstract_off56(int(n),int(h));
    if(start>=(1u<<15)||lc>=8u||delta<-(1<<14)||delta>=(1<<14)||count==0u||count>=1024u||aoff>=8192u)std::exit(21);
    return uint64_t(start)|(uint64_t(lc)<<15)|(uint64_t(uint32_t(delta)&0x7fffu)<<18)|(uint64_t(count)<<33)|(uint64_t(aoff)<<43);
}
static uint32_t ustart(uint64_t x){return uint32_t(x)&0x7fffu;}
static uint32_t ulcount(uint64_t x){return uint32_t(x>>15)&7u;}
static int udelta(uint64_t x){uint32_t z=uint32_t(x>>18)&0x7fffu;if(z&0x4000u)z|=0xffff8000u;return int(int32_t(z));}
static uint32_t ucount(uint64_t x){return uint32_t(x>>33)&0x3ffu;}
static uint32_t uaoff(uint64_t x){return uint32_t(x>>43)&0x1fffu;}

int main(){
    constexpr uint32_t B=16u;
    Factors f=build_factors();const auto owner=low_mask_owners(f);const uint32_t LM=1u<<L;
    std::vector<int32_t> base(size_t(NG)*S*LM,-1);auto bref=[&](int g,int h,uint32_t m)->int32_t&{return base[(size_t(g)*S+size_t(h))*LM+m];};
    for(int h=0;h<=L+1;++h){std::array<uint32_t,NG> next{};for(uint32_t m=0;m<LM;++m){const int g=int(owner[m]);const uint32_t c=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(c)bref(g,h,m)=int32_t(next[g]);next[g]+=c;}}

    uint64_t codes=0,blocks=0,exact=0,successor_steps=0;uint32_t max_steps=0,max_aoff=0;
    for(int g=0;g<NG;++g){
        for(int h=0;h<=L+1;++h){
            struct G{uint64_t p;uint32_t mask;};std::vector<G> v;uint32_t total=0;
            for(uint32_t m=0;m<LM;++m){if(owner[m]!=g)continue;const auto& row=f.low_mask_h[size_t(m)*S+h];if(row.empty())continue;const uint32_t start=uint32_t(bref(g,h,m)),count=uint32_t(row.size()),n=uint32_t(__builtin_popcount(m));int delta=0;if(h+2<=L+1){const int32_t sb=bref(g,h+2,m);if(sb>=0)delta=int(sb)-int(start);}const uint64_t p=pack56(start,n,uint32_t(h),delta,count);if(ustart(p)!=start||ucount(p)!=count||uint32_t(h)+2u*ulcount(p)!=n||udelta(p)!=delta||uaoff(p)!=abstract_off56(int(n),h)||(p>>56)!=0u)return 2;max_aoff=std::max(max_aoff,uaoff(p));v.push_back(G{p,m});total+=count;}
            if(v.empty())continue;codes+=total;uint32_t first=0;const uint32_t nb=(total+B-1u)/B;blocks+=nb;
            for(uint32_t b=0;b<nb;++b){const uint32_t lo=b*B,hi=std::min(total,lo+B);while(first+1u<v.size()&&lo>=ustart(v[first].p)+ucount(v[first].p))++first;uint32_t cursor=first;uint32_t local_steps=0;for(uint32_t r=lo;r<hi;++r){while(r>=ustart(v[cursor].p)+ucount(v[cursor].p)){++cursor;++local_steps;}uint32_t expect=first;while(expect+1u<v.size()&&r>=ustart(v[expect].p)+ucount(v[expect].p))++expect;if(cursor!=expect)return 3;const uint64_t p=v[cursor].p;const uint32_t n=uint32_t(__builtin_popcount(v[cursor].mask));if(uint32_t(h)+2u*ulcount(p)!=n||uaoff(p)!=abstract_off56(int(n),h))return 4;++exact;}successor_steps+=local_steps;max_steps=std::max(max_steps,cursor-first);}
        }
    }
    if(codes!=1201917ull||exact!=codes||blocks!=75175ull||successor_steps!=65159ull||max_steps!=15u||max_aoff!=7059u)return 5;
    std::cout<<"gridfp-rankformula-nometa-group56-coop OK"
             <<" codes="<<codes<<" blocks="<<blocks<<" exact="<<exact
             <<" cooperative_successor_loads="<<successor_steps
             <<" max_steps="<<max_steps<<" max_abstract_off="<<max_aoff
             <<" block=16 leader_gi_increment_exact=1 packed56_decode_exact=1"
             <<" off_table_load_required=0\n";
    return 0;
}
