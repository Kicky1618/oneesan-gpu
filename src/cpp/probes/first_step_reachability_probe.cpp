#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
#include <unordered_set>
#include <vector>
struct Sat{bool v=false;Sat()=default;Sat(uint32_t x):v(x!=0){}Sat&operator=(uint32_t x){v=x!=0;return *this;}void operator+=(Sat const&o){v=v||o.v;}bool operator==(uint32_t x)const{return v==(x!=0);}bool operator!=(uint32_t x)const{return !(*this==x);}};
static std::vector<Mate> states(MateCodec const&mc){std::vector<Mate>o(mc.codeSize());for(Code b=0;b<mc.codeSizeL();++b){auto const&t=mc.codeTable(b);for(Code i=0;i<t.size;++i)o[t.base+i]=t.mateL|t.mateR[i];}return o;}
static uint32_t occ(Mate m,int W){uint32_t z=0;for(int p=0;p<W;++p)if(m.get(p)!=N)z|=1u<<p;return z;}
static void init(PathCounter<Sat>&p){for(Code i=0;i<p.mc.codeSize();++i)p.value[i]=0;for(Code i=0;i<p.wc.codeSize();++i)p.deferred[i]=0;p.value[p.mc.encode(Mate(p.cols-1,R))]=1;}
struct Stats{size_t m=0,d=0,masks=0;};
static Stats stat(PathCounter<Sat>&p,std::vector<Mate>const&ms){std::unordered_set<uint32_t>om;Stats s;for(Code i=0;i<p.mc.codeSize();++i)if(p.value[i]!=0){++s.m;om.insert(occ(ms[i],p.cols));}for(Code i=0;i<p.wc.codeSize();++i)if(p.deferred[i]!=0)++s.d;s.masks=om.size();return s;}
int main(int argc,char**argv){msg=NONE;int W=argc>1?std::atoi(argv[1]):12;PathCounter<Sat>a(W,W,false,false),b(W,W,false,false);init(a);init(b);auto ms=states(a.mc);a.update(0,false);b.update(0,false);std::vector<Code>nz;for(Code i=0;i<b.mc.codeSize();++i)if(b.value[i]!=0)nz.push_back(i);if(nz.size()!=2){std::cerr<<"first split !=2\n";return 1;}for(Code i=0;i<b.mc.codeSize();++i)if(i!=nz[0])b.value[i]=0;for(Code i=0;i<b.wc.codeSize();++i)b.deferred[i]=0;int step=1,total=W*(W-1);for(;step<total;++step){int j=step%(W-1);a.update(j,false);b.update(j,false);if((step+1)%(W-1)==0){auto x=stat(a,ms),y=stat(b,ms);int row=(step+1)/(W-1);std::cout<<"W="<<W<<" row="<<row<<" full_main="<<x.m<<" branch_main="<<y.m<<" ratio="<<(double)y.m/x.m<<" full_masks="<<x.masks<<" branch_masks="<<y.masks<<" mask_ratio="<<(double)y.masks/x.masks<<" full_def="<<x.d<<" branch_def="<<y.d<<"\n";}}
}
