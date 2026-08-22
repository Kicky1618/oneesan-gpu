#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
#include <vector>

static void zero(PathCounter<uint64_t>& pc){for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;}
static uint64_t finish(PathCounter<uint64_t>&pc,int start_step){int W=pc.cols;int steps_per_row=W-1;for(int s=start_step;s<W*steps_per_row;++s){int j=s%steps_per_row;pc.update(j,false);}return pc.value[pc.mc.encode(Mate(0,R))];}
int main(int argc,char**argv){msg=NONE;int W=argc>1?std::atoi(argv[1]):6;PathCounter<uint64_t> pc(W,W,false,false);zero(pc);pc.value[pc.mc.encode(Mate(W-1,R))]=1;int steps=W*(W-1);for(int s=0;s<steps;++s){int j=s%(W-1);pc.update(j,false);std::vector<std::pair<bool,Code>> nz;for(Code i=0;i<pc.mc.codeSize();++i)if(pc.value[i])nz.push_back({false,i});for(Code i=0;i<pc.wc.codeSize();++i)if(pc.deferred[i])nz.push_back({true,i});std::cout<<"after_step="<<(s+1)<<" row="<<(s/(W-1))<<" j="<<j<<" nz="<<nz.size()<<"\n";if(nz.size()>1){uint64_t sum=0;for(auto [d,idx]:nz){PathCounter<uint64_t> q(W,W,false,false);zero(q);uint64_t val=d?pc.deferred[idx]:pc.value[idx];if(d)q.deferred[idx]=val;else q.value[idx]=val;uint64_t z=finish(q,s+1);sum+=z;std::cout<<"  "<<(d?'D':'M')<<" idx="<<idx<<" val="<<val<<" completion="<<z<<"\n";}PathCounter<uint64_t> full(W,W,false,false);uint64_t exact=full.count();std::cout<<"split_sum="<<sum<<" exact="<<exact<<" half="<<(exact/2)<<"\n";break;}}
}
