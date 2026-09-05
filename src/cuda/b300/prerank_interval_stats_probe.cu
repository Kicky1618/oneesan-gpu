#define main oneesan_embedded_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_batch.cu"
#undef main
#include <algorithm>
#include <iostream>
#include <vector>

struct Stat { unsigned long long elems=0,runs=0; unsigned groups=0; std::vector<unsigned long long> avg; };
static void add(Stat& s,const GroupSpec& sp){ auto v=make_intervals(sp); s.elems+=sp.size; s.runs+=v.size(); s.groups++; s.avg.push_back(v.empty()?0:sp.size/v.size()); }
static void report(const char*tag,Stat&s){
  auto a=s.avg; std::sort(a.begin(),a.end());
  auto pct=[&](double q){return a.empty()?0ull:a[std::min<size_t>(a.size()-1,size_t(q*(a.size()-1)))];};
  std::cout<<tag<<" groups="<<s.groups<<" elems="<<s.elems<<" runs="<<s.runs<<" weighted_avg="<<(s.runs?s.elems/s.runs:0)
           <<" p0="<<pct(0)<<" p25="<<pct(.25)<<" p50="<<pct(.5)<<" p75="<<pct(.75)<<" p90="<<pct(.9)<<" p99="<<pct(.99)<<" max="<<(a.empty()?0:a.back())<<"\n";
  for(unsigned long long th: {1ull,2ull,4ull,8ull,16ull,32ull,64ull,128ull,256ull,512ull,1024ull,4096ull,16384ull,65536ull}){
    unsigned n=0; unsigned long long e=0;
    for(size_t i=0;i<a.size();++i) if(a[i]>=th){n++;}
    std::cout<<"  >="<<th<<" groups="<<n<<"/"<<a.size()<<"\n";
  }
}
int main(){ build_full_dp(); constexpr int W=TARGET_W; Stat sm,sd; const int rr[2][2]={{W-1,LOW_LUT_K+1},{LOW_LUT_K,1}}; for(auto const&r:rr){auto fp=window_candidates(W,r[0],r[1]); int nj=1<<int(fp.size()); for(int g=0;g<nj;++g){uint32_t mf,mo,bf,bo;window_masks(W,r[0],r[1],fp,uint32_t(g),mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);add(sm,ms);add(sd,ds);}} report("MAIN",sm);report("BLOCK",sd); }
