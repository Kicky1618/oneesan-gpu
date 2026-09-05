#include "../../common/gridfp_transition.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {
using namespace oneesan::gridfp;

bool valid_mate(MateID m,int len){int h=1;for(int pos=0;pos<len;++pos){auto v=mget(m,len-1-pos);if(v==X)return false;if(v==R)--h;else if(v==L)++h;if(h<0)return false;}return h==0;}
void enum_rec(int len,int bit,MateID m,std::vector<MateID>&out){if(bit==len){if(valid_mate(m,len))out.push_back(m);return;}enum_rec(len,bit+1,m,out);enum_rec(len,bit+1,m|(MateID(R)<<(2*bit)),out);enum_rec(len,bit+1,m|(MateID(L)<<(2*bit)),out);} std::vector<MateID> enumerate(int len){std::vector<MateID>o;enum_rec(len,0,0,o);return o;}

std::uint32_t endpoint_mask(MateID mate,int W){std::uint64_t x=(std::uint64_t(mate)|(std::uint64_t(mate)>>1))&0x5555555555555555ULL;x=(x|(x>>1))&0x3333333333333333ULL;x=(x|(x>>2))&0x0f0f0f0f0f0f0f0fULL;x=(x|(x>>4))&0x00ff00ff00ff00ffULL;x=(x|(x>>8))&0x0000ffff0000ffffULL;x=(x|(x>>16))&0x00000000ffffffffULL;std::uint32_t z=std::uint32_t(x);if(W<32)z&=(std::uint32_t(1)<<W)-1u;return z;}

using Candidate=std::pair<int,int>; // kind: 0 LL, 1 RR; q
std::vector<Candidate> scan_ref(MateID d,int W,int p,std::uint64_t&iterations){std::vector<Candidate>o;int bal=0;for(int q=p-2;q>=0;--q){++iterations;auto v=mget(d,q);if(bal==0&&v==L)o.emplace_back(0,q);if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}bal=0;for(int q=p+1;q<W;++q){++iterations;auto v=mget(d,q);if(bal==0&&v==R)o.emplace_back(1,q);if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}return o;}
std::vector<Candidate> scan_fast(MateID d,int W,int p,std::uint64_t&iterations){std::vector<Candidate>o;const std::uint32_t ep=endpoint_mask(d,W);std::uint32_t left=p<=1?0u:(ep&((std::uint32_t(1)<<(p-1))-1u));int bal=0;while(left){++iterations;const int q=31-__builtin_clz(left);const auto v=mget(d,q);if(bal==0&&v==L)o.emplace_back(0,q);if(v==L)++bal;else if(v==R)--bal;left^=std::uint32_t(1)<<q;if(bal<0)break;}const std::uint32_t width=(std::uint32_t(1)<<W)-1u;const std::uint32_t low=(std::uint32_t(1)<<(p+1))-1u;std::uint32_t right=ep&width&~low;bal=0;while(right){++iterations;const int q=__builtin_ffs(right)-1;const auto v=mget(d,q);if(bal==0&&v==R)o.emplace_back(1,q);if(v==R)++bal;else if(v==L)--bal;right&=right-1u;if(bal<0)break;}return o;}

struct Gen{static constexpr int M=28;std::array<std::array<std::uint64_t,M+2>,M+1>d{};Gen(){d[0][0]=1;for(int rem=1;rem<=M;++rem)for(int h=0;h<=M;++h)d[rem][h]=d[rem-1][h]+(h?d[rem-1][h-1]:0)+d[rem-1][h+1];}MateID sample(int W,std::mt19937_64&rng)const{MateID m=0;int h=1;for(int pos=0;pos<W;++pos){int rem=W-pos-1;auto cn=d[rem][h],cr=h?d[rem][h-1]:0,cl=d[rem][h+1];auto pick=rng()%(cn+cr+cl);MateValue v=N;if(pick<cn)v=N;else if((pick-=cn)<cr){v=R;--h;}else{v=L;++h;}m|=MateID(v)<<(2*(W-1-pos));}return m;}};

bool check(MateID b,int W,std::uint64_t&positions,std::uint64_t&endpoints,std::uint64_t&cases){if(!valid_mate(b,W-1))return false;for(int p=1;p<W;++p){const MateID d=minsert(b,p-1,N);if(mpair(d,p)!=NN)continue;const auto a=scan_ref(d,W,p,positions);const auto z=scan_fast(d,W,p,endpoints);++cases;if(a!=z){std::cerr<<"candidate sequence mismatch W="<<W<<" p="<<p<<'\n';return false;}}return true;}
}

int main(){std::uint64_t positions=0,endpoints=0,cases=0,blocked=0;for(int W=3;W<=10;++W)for(MateID b:enumerate(W-1)){++blocked;if(!check(b,W,positions,endpoints,cases))return 2;}Gen gen;std::mt19937_64 rng(0x656e64706f696e74ULL);constexpr std::uint64_t RANDOM=500000;for(std::uint64_t i=0;i<RANDOM;++i){int W=8+2*int(rng()%11);if(!check(gen.sample(W-1,rng),W,positions,endpoints,cases))return 3;}if(endpoints>positions||!cases)return 4;std::cout<<"gridfp-runtime-discovery-endpoint-scan-proof OK exhaustive_W_max=10 exhaustive_blocked="<<blocked<<" random_cases="<<RANDOM<<" scan_cases="<<cases<<" position_iterations="<<positions<<" endpoint_iterations="<<endpoints<<" candidate_sequence_exact=1 production_W_max=28 exact=1\n";return 0;}
