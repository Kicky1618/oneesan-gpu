#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
using Pairing=std::vector<int>;
static void gen(int lo,int hi,Pairing&base,std::vector<Pairing>&out){if(lo>hi){out.push_back(base);return;}for(int j=lo+1;j<=hi;j+=2){std::vector<Pairing>L,R;Pairing a(base.size(),-1),b(base.size(),-1);gen(lo+1,j-1,a,L);gen(j+1,hi,b,R);for(auto const&l:L)for(auto const&r:R){Pairing x=base;x[lo]=j;x[j]=lo;for(int k=lo+1;k<j;++k)if(l[k]>=0)x[k]=l[k];for(int k=j+1;k<=hi;++k)if(r[k]>=0)x[k]=r[k];out.push_back(std::move(x));}}}
static std::vector<Pairing> all(int m){Pairing b(2*m,-1);std::vector<Pairing>o;gen(0,2*m-1,b,o);return o;}
static bool conn(Pairing const&a,Pairing const&b){int n=a.size(),u=0,prev=-1;std::vector<uint8_t>seen(n);for(int step=0;step<n;++step){if(seen[u])return false;seen[u]=1;int v=(step&1)?b[u]:a[u];prev=u;u=v;}return u==0&&std::all_of(seen.begin(),seen.end(),[](auto x){return x;});}
static uint64_t pw(uint64_t a,uint64_t e,uint64_t p){uint64_t r=1;while(e){if(e&1)r=(__uint128_t)r*a%p;a=(__uint128_t)a*a%p;e>>=1;}return r;}
static int rank_mod(std::vector<Pairing>const&s,uint64_t p){int n=s.size();std::vector<uint64_t>A((size_t)n*n);for(int i=0;i<n;++i)for(int j=0;j<n;++j)A[(size_t)i*n+j]=conn(s[i],s[j]);int r=0;for(int c=0;c<n&&r<n;++c){int q=r;while(q<n&&!A[(size_t)q*n+c])++q;if(q==n)continue;if(q!=r)for(int j=c;j<n;++j)std::swap(A[(size_t)q*n+j],A[(size_t)r*n+j]);uint64_t inv=pw(A[(size_t)r*n+c],p-2,p);for(int j=c;j<n;++j)A[(size_t)r*n+j]=(__uint128_t)A[(size_t)r*n+j]*inv%p;for(int i=0;i<n;++i)if(i!=r&&A[(size_t)i*n+c]){uint64_t f=A[(size_t)i*n+c];for(int j=c;j<n;++j){uint64_t z=(__uint128_t)f*A[(size_t)r*n+j]%p;auto &x=A[(size_t)i*n+j];x=x>=z?x-z:x+p-z;}}++r;}return r;}
static int rank_gf2(std::vector<Pairing>const&s){int n=s.size(),words=(n+63)/64;std::vector<uint64_t>A((size_t)n*words);for(int i=0;i<n;++i)for(int j=0;j<n;++j)if(conn(s[i],s[j]))A[(size_t)i*words+j/64]|=1ull<<(j%64);int r=0;for(int c=0;c<n&&r<n;++c){int q=r;while(q<n&&!((A[(size_t)q*words+c/64]>>(c%64))&1))++q;if(q==n)continue;if(q!=r)for(int w=0;w<words;++w)std::swap(A[(size_t)q*words+w],A[(size_t)r*words+w]);for(int i=0;i<n;++i)if(i!=r&&((A[(size_t)i*words+c/64]>>(c%64))&1))for(int w=c/64;w<words;++w)A[(size_t)i*words+w]^=A[(size_t)r*words+w];++r;}return r;}
int main(int argc,char**argv){int mx=argc>1?std::atoi(argv[1]):7;uint64_t p=4294967291ull;for(int m=1;m<=mx;++m){auto s=all(m);int r2=rank_gf2(s);int rp=m<=7?rank_mod(s,p):-1;std::cout<<"m="<<m<<" catalan="<<s.size()<<" rank_gf2="<<r2<<" rank_p="<<rp<<"\n";}}
