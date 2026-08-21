#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
using Code=unsigned long long;
static constexpr int MAXW=28;
struct GroupSpec{Code dp[MAXW+1][MAXW+2]{}; Code size=0;};
static GroupSpec make_spec(int width,uint32_t fixed,uint32_t occ){GroupSpec s;for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);for(int w=1;w<=width;++w){int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;for(int h=0;h<=MAXW;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h>0)x+=s.dp[w-1][h-1];if(h<MAXW+1)x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}s.size=s.dp[width][1];return s;}
static std::vector<int> candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
static void masks(int hi,int lo,const std::vector<int>&fp,uint32_t group,uint32_t&mf,uint32_t&mo,uint32_t&bf,uint32_t&bo){mf=mo=bf=bo=0;for(size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=(q<lo-1)?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
struct Plan{int hi=0,lo=0,k=-1;size_t maxbytes=0;Code maxm=0,maxd=0;std::vector<int>fp;};
static Plan plan_window(int W,int hi,int lo,size_t target,int maxbits=20){Plan z;z.hi=hi;z.lo=lo;auto c=candidates(W,hi,lo);int klim=std::min<int>(c.size(),maxbits);for(int k=0;k<=klim;++k){std::vector<int>fp(c.begin(),c.begin()+k);size_t mx=0;Code mm=0,md=0;uint64_t ng=1ull<<k;for(uint64_t g=0;g<ng;++g){uint32_t mf,mo,bf,bo;masks(hi,lo,fp,(uint32_t)g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo),ds=make_spec(W-1,bf,bo);size_t b=size_t(2*ms.size+2*ds.size)*4;if(b>mx){mx=b;mm=ms.size;md=ds.size;}if(mx>target&&k<klim)break;}if(mx<=target){z.k=k;z.maxbytes=mx;z.maxm=mm;z.maxd=md;z.fp=std::move(fp);return z;}}return z;}
static void cache_stats(int W,const Plan&p,size_t target){uint64_t ng=1ull<<p.k, fit=0;long double total=0,cached=0;for(uint64_t g=0;g<ng;++g){uint32_t mf,mo,bf,bo;masks(p.hi,p.lo,p.fp,(uint32_t)g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo),ds=make_spec(W-1,bf,bo);size_t cb=size_t(2*ms.size+2*ds.size)*4,mb=size_t(ms.size)*8;total+=ms.size;if(cb+mb<=target){++fit;cached+=ms.size;}}std::cout<<" cache_groups="<<fit<<"/"<<ng<<" cache_main_frac="<<std::fixed<<std::setprecision(5)<<double(cached/total);}
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):27;int mib=argc>2?atoi(argv[2]):4096;int maxwin=argc>3?atoi(argv[3]):14;int W=n+1;size_t target=size_t(mib)<<20;std::cout<<"n="<<n<<" target="<<mib<<"MiB max_window="<<maxwin<<"\n";int hi=W-1,wi=0;while(hi>=1){Plan p;for(int lo=std::max(1,hi-maxwin+1);lo<=hi;++lo){auto q=plan_window(W,hi,lo,target);if(q.k>=0){p=std::move(q);break;}}if(p.k<0){std::cout<<"FAIL hi="<<hi<<"\n";return 2;}std::cout<<"window"<<++wi<<" p="<<p.hi<<".."<<p.lo<<" len="<<p.hi-p.lo+1<<" k="<<p.k<<" groups="<<(1ull<<p.k)<<" maxGiB="<<std::setprecision(4)<<double(p.maxbytes)/(1ull<<30);cache_stats(W,p,target);std::cout<<" fixed=";for(int q:p.fp)std::cout<<q<<",";std::cout<<"\n";hi=p.lo-1;} }
