#define main oneesan_embedded_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_batch.cu"
#undef main
#include <algorithm>
#include <iostream>
#include <vector>
struct Stat { unsigned long long elems=0,runs=0; unsigned groups=0; std::vector<unsigned long long> avg; };
static void add(Stat&s,const GroupSpec&sp){auto r=interval_leaf_upper(sp);s.elems+=sp.size;s.runs+=r;s.groups++;s.avg.push_back(r?sp.size/r:0);}
static void report(const char*t,Stat&s){auto a=s.avg;std::sort(a.begin(),a.end());auto q=[&](double x){return a.empty()?0ull:a[size_t(x*(a.size()-1))];};std::cout<<t<<" groups="<<s.groups<<" weighted_avg="<<(s.runs?s.elems/s.runs:0)<<" p0="<<q(0)<<" p50="<<q(.5)<<" p90="<<q(.9)<<" p99="<<q(.99)<<" max="<<(a.empty()?0:a.back())<<"\n";for(unsigned long long th:{1ull,2ull,4ull,8ull,16ull,32ull,64ull,128ull,256ull,512ull,1024ull,4096ull,16384ull,65536ull}){unsigned n=0;for(auto x:a)n+=x>=th;std::cout<<"  >="<<th<<" "<<n<<"/"<<a.size()<<"\n";}}
int main(){build_full_dp();constexpr int W=TARGET_W;Stat m,d;const int rr[2][2]={{W-1,LOW_LUT_K+1},{LOW_LUT_K,1}};for(auto const&r:rr){auto fp=window_candidates(W,r[0],r[1]);int nj=1<<int(fp.size());for(int g=0;g<nj;++g){uint32_t mf,mo,bf,bo;window_masks(W,r[0],r[1],fp,uint32_t(g),mf,mo,bf,bo);add(m,make_spec(W,mf,mo));add(d,make_spec(W-1,bf,bo));}}report("MAIN",m);report("BLOCK",d);}
