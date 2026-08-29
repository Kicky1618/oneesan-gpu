#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

static int bits_u32(uint32_t x){int b=0;do{++b;x>>=1;}while(x);return b;}
static uint32_t abstract_off61(int n,int h){uint32_t off=0;for(int nn=0;nn<=L;++nn)for(int hh=0;hh<16;++hh){if(nn==n&&hh==h)return off;off+=ballot_suffix(nn,hh);}return off;}
static uint64_t pack61(uint32_t start,uint32_t end,uint32_t source_base,uint32_t lcount,uint32_t aoff){
    if(start>=32768u||end>=32768u||source_base>=32768u||lcount>=8u||aoff>=8192u||end<=start)std::exit(20);
    return uint64_t(start)|(uint64_t(end)<<15)|(uint64_t(source_base)<<30)|(uint64_t(lcount)<<45)|(uint64_t(aoff)<<48);
}
int main(){
    Factors f=build_factors();const auto owner=low_mask_owners(f);const uint32_t LM=1u<<L;
    std::vector<int32_t> base(size_t(NG)*S*LM,-1);auto bref=[&](int g,int h,uint32_t m)->int32_t&{return base[(size_t(g)*S+size_t(h))*LM+m];};
    std::array<std::array<uint32_t,S>,NG> ranks{};
    for(int h=0;h<=L+1;++h){std::array<uint32_t,NG> next{};for(uint32_t m=0;m<LM;++m){const int g=int(owner[m]);const uint32_t c=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(c)bref(g,h,m)=int32_t(next[g]);next[g]+=c;}for(int g=0;g<NG;++g)ranks[g][h]=next[g];}
    uint32_t max_start=0,max_end=0,max_source_base=0,max_lcount=0,max_aoff=0,max_height_rank=0;uint64_t groups=0,exact=0,source_rows=0;
    for(int g=0;g<NG;++g)for(int h=0;h<=L+1;++h){max_height_rank=std::max(max_height_rank,ranks[g][h]);for(uint32_t m=0;m<LM;++m){if(owner[m]!=g)continue;const auto& row=f.low_mask_h[size_t(m)*S+h];if(row.empty())continue;const uint32_t start=uint32_t(bref(g,h,m)),end=start+uint32_t(row.size()),n=uint32_t(__builtin_popcount(m));if(n<uint32_t(h)||((n-uint32_t(h))&1u))return 2;const uint32_t lc=(n-uint32_t(h))>>1,aoff=abstract_off61(int(n),h);uint32_t sb=0;if(h+2<=L+1){const int32_t b=bref(g,h+2,m);if(b>=0){sb=uint32_t(b);++source_rows;}}max_start=std::max(max_start,start);max_end=std::max(max_end,end);max_source_base=std::max(max_source_base,sb);max_lcount=std::max(max_lcount,lc);max_aoff=std::max(max_aoff,aoff);const uint64_t p=pack61(start,end,sb,lc,aoff);const uint32_t us=uint32_t(p)&0x7fffu,ue=uint32_t(p>>15)&0x7fffu,ub=uint32_t(p>>30)&0x7fffu,ul=uint32_t(p>>45)&7u,ua=uint32_t(p>>48)&0x1fffu;if(us!=start||ue!=end||ub!=sb||ul!=lc||ua!=aoff||(p>>61)!=0u)return 3;++groups;++exact;}}
    if(groups!=69632ull||exact!=groups||max_start!=29113u||max_end!=30114u||max_source_base!=29113u||max_lcount!=7u||max_aoff!=7059u||max_height_rank!=30114u)return 4;
    std::cout<<"gridfp-rankformula-nometa-group61 OK groups="<<groups
             <<" max_start="<<max_start<<" start_bits="<<bits_u32(max_start)
             <<" max_end="<<max_end<<" end_bits="<<bits_u32(max_end)
             <<" max_source_base="<<max_source_base<<" source_base_bits="<<bits_u32(max_source_base)
             <<" max_lcount="<<max_lcount<<" lcount_bits="<<bits_u32(max_lcount)
             <<" max_abstract_off="<<max_aoff<<" abstract_off_bits="<<bits_u32(max_aoff)
             <<" max_height_rank="<<max_height_rank
             <<" source_rows="<<source_rows
             <<" packed_bits=61 spare_bits=3 exact="<<exact
             <<" direct_end_compare=1 direct_source_base=1 signed_delta_decode=0\n";
    return 0;
}
