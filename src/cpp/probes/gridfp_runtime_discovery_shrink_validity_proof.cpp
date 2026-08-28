#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
using MateID=std::uint64_t;
enum MateValue:std::uint8_t{N=0,R=1,L=2,X=3};
enum MateValuePair:std::uint8_t{NN=0x0,NR=0x1,NL=0x2,NX=0x3,RN=0x4,RR=0x5,RL=0x6,RX=0x7,LN=0x8,LR=0x9,LL=0xa,LX=0xb,XN=0xc,XR=0xd,XL=0xe,XX=0xf};
MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);} MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);} MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return(m&~z)|(MateID(v)<<(2*(p-1)));} MateID mshrink(MateID m,int k){MateID mask=(MateID(1)<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);} MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((MateID(1)<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}
bool valid(MateID m,int len){int h=1;for(int pos=0;pos<len;++pos){auto v=mget(m,len-1-pos);if(v==X)return false;if(v==R)--h;else if(v==L)++h;if(h<0)return false;}return h==0;}
void enum_rec(int len,int bit,MateID m,std::vector<MateID>&out){if(bit==len){if(valid(m,len))out.push_back(m);return;}enum_rec(len,bit+1,m,out);enum_rec(len,bit+1,m|(MateID(R)<<(2*bit)),out);enum_rec(len,bit+1,m|(MateID(L)<<(2*bit)),out);} std::vector<MateID> enumerate(int len){std::vector<MateID>o;enum_rec(len,0,0,o);return o;}
struct Gen{static constexpr int M=28;std::array<std::array<std::uint64_t,M+2>,M+1>d{};Gen(){d[0][0]=1;for(int rem=1;rem<=M;++rem)for(int h=0;h<=M;++h){d[rem][h]=d[rem-1][h]+(h?d[rem-1][h-1]:0)+d[rem-1][h+1];}}MateID sample(int W,std::mt19937_64&rng)const{MateID m=0;int h=1;for(int pos=0;pos<W;++pos){int rem=W-pos-1;auto cn=d[rem][h],cr=h?d[rem][h-1]:0,cl=d[rem][h+1],tot=cn+cr+cl,pick=rng()%tot;MateValue v=N;if(pick<cn)v=N;else if((pick-=cn)<cr){v=R;--h;}else{v=L;++h;}m|=MateID(v)<<(2*(W-1-pos));}return m;}};
bool check(MateID d,int W,std::uint64_t&a,std::uint64_t&b){if(!valid(d,W))return false;for(int p=1;p<W;++p){if(mget(d,p)==N&&(mget(d,p-1)==L||mget(d,p-1)==R)){++a;MateID z=mshrink(d,p);if(!valid(z,W-1)||mget(z,p-1)==N||minsert(z,p,N)!=d){std::cerr<<"remove-N mismatch W="<<W<<" p="<<p<<'\n';return false;}}const int q=p-1;if(q<1)continue;const auto qp=mpair(d,q);if(qp==NN||qp==LR){++b;const MateID nn=qp==NN?d:msetpair(d,q,NN);const MateID z=mshrink(nn,q);if(!valid(z,W-1)||mget(z,q-1)!=N){std::cerr<<"LR/NN shrink mismatch W="<<W<<" p="<<p<<" q="<<q<<" pair="<<int(qp)<<'\n';return false;}}}return true;}
}
int main(){std::uint64_t states=0,remove_n=0,lrnn=0;for(int W=3;W<=10;++W){for(MateID d:enumerate(W)){++states;if(!check(d,W,remove_n,lrnn))return 2;}}Gen gen;std::mt19937_64 rng(0x736872696e6bULL);constexpr std::uint64_t RANDOM=500000;for(std::uint64_t i=0;i<RANDOM;++i){int W=8+2*int(rng()%11);if(!check(gen.sample(W,rng),W,remove_n,lrnn))return 3;}std::cout<<"gridfp-runtime-discovery-shrink-validity-proof OK exhaustive_W_max=10 exhaustive_states="<<states<<" random_cases="<<RANDOM<<" remove_n_known_valid="<<remove_n<<" lr_or_nn_then_shrink_known_valid="<<lrnn<<" production_W_max=28 exact=1\n";return 0;}
