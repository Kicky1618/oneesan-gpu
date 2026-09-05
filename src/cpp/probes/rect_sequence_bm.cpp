#define main ggcount_embedded_main
#include "ggcount_public.cpp"
#undef main
#include <vector>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <chrono>

#ifndef BM_PRIME
#define BM_PRIME 1000000007ULL
#endif
static constexpr uint64_t P = BM_PRIME;
static uint64_t modpow(uint64_t a,uint64_t e){uint64_t r=1;while(e){if(e&1)r=(__uint128_t)r*a%P;a=(__uint128_t)a*a%P;e>>=1;}return r;}
static std::vector<uint64_t> BM(const std::vector<uint64_t>& s){
  std::vector<uint64_t> C(1,1),B(1,1); int L=0,m=1; uint64_t b=1;
  for(int n=0;n<(int)s.size();++n){
    uint64_t d=s[n]; for(int i=1;i<=L;++i)d=(d+(__uint128_t)C[i]*s[n-i])%P;
    if(!d){++m;continue;} auto T=C; uint64_t coef=(__uint128_t)d*modpow(b,P-2)%P;
    if(C.size()<B.size()+m)C.resize(B.size()+m);
    for(int j=0;j<(int)B.size();++j){uint64_t z=(__uint128_t)coef*B[j]%P;C[j+m]=(C[j+m]+P-z)%P;}
    if(2*L<=n){L=n+1-L;B=T;b=d;m=1;}else ++m;
  }
  C.erase(C.begin()); for(auto &x:C) if(x)x=P-x; return C;
}

static std::vector<uint64_t> combinePoly(const std::vector<uint64_t>& a,const std::vector<uint64_t>& b,const std::vector<uint64_t>& rec){
  int d=rec.size();std::vector<uint64_t> t(2*d-1);
  for(int i=0;i<d;++i)if(a[i])for(int j=0;j<d;++j)if(b[j])t[i+j]=(t[i+j]+(__uint128_t)a[i]*b[j])%P;
  for(int k=2*d-2;k>=d;--k)if(t[k]){uint64_t z=t[k];for(int j=1;j<=d;++j)t[k-j]=(t[k-j]+(__uint128_t)z*rec[j-1])%P;}
  t.resize(d);return t;
}
static uint64_t nthRec(const std::vector<uint64_t>& init,const std::vector<uint64_t>& rec,unsigned long long n){
  int d=rec.size();if(n<(unsigned long long)init.size())return init[n];std::vector<uint64_t> ans(d),x(d);ans[0]=1;if(d==1)x[0]=rec[0];else x[1]=1;
  while(n){if(n&1)ans=combinePoly(ans,x,rec);n>>=1;if(n)x=combinePoly(x,x,rec);}uint64_t z=0;for(int i=0;i<d;++i)z=(z+(__uint128_t)ans[i]*init[i])%P;return z;
}

static std::vector<uint64_t> seq(int cols,int H){
  modulus=P; msg=NONE; PathCounter<Modnum<uint64_t>> pc(1,cols,false,false);
  for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;
  for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;
  pc.value[pc.mc.encode(Mate(cols-1,R))]=1;
  std::vector<uint64_t>a; a.reserve(H);
  for(int row=0;row<H;++row){for(int j=0;j<cols-2;++j)pc.update(j,false);pc.update(cols-2,false);a.push_back((uint64_t)pc.value[pc.mc.encode(Mate(0,R))]);}
  return a;
}
#ifndef RECT_SEQUENCE_NO_MAIN
int main(int argc,char**argv){int lo=argc>1?atoi(argv[1]):2,hi=argc>2?atoi(argv[2]):8,H=argc>3?atoi(argv[3]):500; unsigned long long queryH=argc>4?strtoull(argv[4],nullptr,10):0;
 for(int W=lo;W<=hi;++W){auto a=seq(W,H);bool ok=true;for(int h=1;h<=std::min(H,8);++h){PathCounter<Modnum<uint64_t>> q(h,W,false,false);auto z=(uint64_t)q.count();if(z!=a[h-1]){ok=false;std::cerr<<"mismatch W="<<W<<" H="<<h<<" "<<z<<" "<<a[h-1]<<"\n";}}
 auto c=BM(a);int d=c.size();if(d<=30){std::cout<<"REC W="<<W<<":";for(auto x:c)std::cout<<" "<<x;std::cout<<"\n";}int good=0;for(int n=d;n<H;++n){uint64_t z=0;for(int i=1;i<=d;++i)z=(z+(__uint128_t)c[i-1]*a[n-i])%P;if(z==a[n])++good;}
 std::cout<<"W="<<W<<" degree="<<d<<" verified="<<good<<"/"<<(H-d)<<" rect_ok="<<ok<<" first=";for(int i=0;i<std::min(H,8);++i)std::cout<<(i?",":"")<<a[i];if(queryH){auto t0=std::chrono::steady_clock::now();auto q=nthRec(a,c,queryH-1);auto t1=std::chrono::steady_clock::now();std::cout<<" query_H="<<queryH<<" query="<<q<<" query_ms="<<std::chrono::duration<double,std::milli>(t1-t0).count();}std::cout<<"\n";
 }
}
#endif
