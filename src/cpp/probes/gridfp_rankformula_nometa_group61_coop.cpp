#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

static uint32_t aoff61(int n,int h){uint32_t off=0;for(int nn=0;nn<=L;++nn)for(int hh=0;hh<16;++hh){if(nn==n&&hh==h)return off;off+=ballot_suffix(nn,hh);}return off;}
static uint64_t pack61(uint32_t start,uint32_t end,uint32_t source_base,uint32_t lcount,uint32_t aoff){if(start>=32768u||end>=32768u||source_base>=32768u||lcount>=8u||aoff>=8192u||end<=start)std::exit(20);return uint64_t(start)|(uint64_t(end)<<15)|(uint64_t(source_base)<<30)|(uint64_t(lcount)<<45)|(uint64_t(aoff)<<48);}
static uint32_t us(uint64_t x){return uint32_t(x)&0x7fffu;}static uint32_t ue(uint64_t x){return uint32_t(x>>15)&0x7fffu;}static uint32_t ub(uint64_t x){return uint32_t(x>>30)&0x7fffu;}static uint32_t ul(uint64_t x){return uint32_t(x>>45)&7u;}static uint32_t ua(uint64_t x){return uint32_t(x>>48)&0x1fffu;}

int main(){
    constexpr uint32_t B=16u;Factors f=build_factors();const auto owner=low_mask_owners(f);const uint32_t LM=1u<<L;
    std::vector<int32_t> base(size_t(NG)*S*LM,-1);auto bref=[&](int g,int h,uint32_t m)->int32_t&{return base[(size_t(g)*S+size_t(h))*LM+m];};
    for(int h=0;h<=L+1;++h){std::array<uint32_t,NG> next{};for(uint32_t m=0;m<LM;++m){int g=int(owner[m]);uint32_t c=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(c)bref(g,h,m)=int32_t(next[g]);next[g]+=c;}}
    uint64_t codes=0,blocks=0,exact=0,successor=0,direct_source_checks=0;uint32_t max_steps=0,max_end=0,max_source_base=0;
    for(int g=0;g<NG;++g)for(int h=0;h<=L+1;++h){struct G{uint64_t p;uint32_t mask;};std::vector<G> v;uint32_t total=0;for(uint32_t m=0;m<LM;++m){if(owner[m]!=g)continue;const auto& row=f.low_mask_h[size_t(m)*S+h];if(row.empty())continue;uint32_t start=uint32_t(bref(g,h,m)),end=start+uint32_t(row.size()),n=uint32_t(__builtin_popcount(m)),lc=(n-uint32_t(h))>>1,sb=0;if(n<uint32_t(h)||((n-uint32_t(h))&1u))return 2;if(lc){int32_t b=bref(g,h+2,m);if(b<0)return 3;sb=uint32_t(b);++direct_source_checks;}uint64_t p=pack61(start,end,sb,lc,aoff61(int(n),h));if(us(p)!=start||ue(p)!=end||ub(p)!=sb||ul(p)!=lc||ua(p)!=aoff61(int(n),h))return 4;max_end=std::max(max_end,end);max_source_base=std::max(max_source_base,sb);v.push_back(G{p,m});total=end;}if(v.empty())continue;codes+=total;uint32_t first=0;uint32_t nb=(total+B-1u)/B;blocks+=nb;for(uint32_t b=0;b<nb;++b){uint32_t lo=b*B,hi=std::min(total,lo+B);while(first+1u<v.size()&&lo>=ue(v[first].p))++first;uint32_t cursor=first,steps=0;for(uint32_t r=lo;r<hi;++r){while(r>=ue(v[cursor].p)){++cursor;++steps;}uint32_t expect=first;while(expect+1u<v.size()&&r>=ue(v[expect].p))++expect;if(cursor!=expect)return 5;uint64_t p=v[cursor].p;uint32_t n=uint32_t(__builtin_popcount(v[cursor].mask));if(uint32_t(h)+2u*ul(p)!=n||ua(p)!=aoff61(int(n),h))return 6;if(ul(p)){int32_t want=bref(g,h+2,v[cursor].mask);if(want<0||ub(p)!=uint32_t(want))return 7;}else if(ub(p)!=0u)return 8;++exact;}successor+=steps;max_steps=std::max(max_steps,cursor-first);}}
    if(codes!=1201917ull||blocks!=75175ull||exact!=codes||successor!=65159ull||max_steps!=15u||max_end!=30114u||max_source_base!=29113u)return 9;
    std::cout<<"gridfp-rankformula-nometa-group61-coop OK codes="<<codes<<" blocks="<<blocks<<" exact="<<exact<<" cooperative_successor_loads="<<successor<<" max_steps="<<max_steps<<" max_end="<<max_end<<" max_source_base="<<max_source_base<<" direct_source_checks="<<direct_source_checks<<" block=16 leader_gi_increment_exact=1 direct_end_compare_exact=1 direct_source_base_exact=1\n";
    return 0;
}
