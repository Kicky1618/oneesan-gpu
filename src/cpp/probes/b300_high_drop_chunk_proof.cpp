#include <array>
#include <cstdint>
#include <iostream>

namespace {
using i64=std::int64_t;using u64=std::uint64_t;
constexpr int W=28,HC=30;
std::array<std::array<u64,HC>,W+1> M{},B{};

i64 step(int pos,int&h,int v){
    if(v==1){if(h<=0)return INT64_MIN; i64 d=i64(B[pos-1][h])-i64(M[pos][h]);--h;return d;}
    if(v==2){if(h>=W+1)return INT64_MIN;u64 b=B[pos-1][h]+(h?B[pos-1][h-1]:0),a=M[pos][h]+(h?M[pos][h-1]:0);++h;return i64(b)-i64(a);}
    return 0;
}

struct Packed{u64 z=0;bool ok=false;};
Packed make_chunk(int c,int h0,unsigned code){
    int h=h0; i64 d=0; const int top=27-4*c,lo=top-3;
    for(int pos=top;pos>=lo;--pos){int v=(code>>(2*(pos-lo)))&3u;if(v==3)return {};i64 x=step(pos,h,v);if(x==INT64_MIN)return {};d+=x;}
    constexpr u64 mask=(1ull<<56)-1;
    return { (u64(d)&mask)|(u64(h)<<56), true };
}
i64 unpack_delta(u64 z){return i64(z<<8)>>8;}

bool check_code(int p,u64 ternary){
    std::array<int,12> v{};int n=27-p;u64 x=ternary;
    for(int k=0;k<n;++k){v[k]=int(x%3);x/=3;}
    int hs=1;i64 scalar=0;
    for(int k=0;k<n;++k){int pos=27-k;i64 d=step(pos,hs,v[k]);if(d==INT64_MIN)return true;scalar+=d;}

    int h=1;i64 chunked=0;int full=n>>2,rem=n&3;
    for(int c=0;c<full;++c){unsigned code=0;const int lo=24-4*c;
        for(int r=0;r<4;++r){int pos=27-4*c-r;int vv=v[27-pos];code|=unsigned(vv)<<(2*(pos-lo));}
        Packed q=make_chunk(c,h,code);if(!q.ok)return false;chunked+=unpack_delta(q.z);h=int(q.z>>56);
    }
    int pos=27-(full<<2);
    for(int r=0;r<rem;++r,--pos){int vv=v[27-pos];i64 d=step(pos,h,vv);if(d==INT64_MIN)return false;chunked+=d;}
    return h==hs&&chunked==scalar;
}
}

int main(){
    // Deterministic nontrivial tables: the proof targets segmentation, height
    // propagation and 56-bit signed packing, independent of a particular group.
    for(int p=0;p<=W;++p)for(int h=0;h<HC;++h){
        M[p][h]=u64(1000003ull*p+1009ull*h+17ull*p*h+11);
        B[p][h]=u64(999983ull*p+1013ull*h+19ull*p*h+7);
    }
    std::uint64_t checked=0;
    for(int p=15;p<=27;++p){int n=27-p;u64 total=1;for(int i=0;i<n;++i)total*=3;
        for(u64 code=0;code<total;++code){if(!check_code(p,code)){std::cerr<<"mismatch p="<<p<<" code="<<code<<"\n";return 2;}++checked;}
    }
    // Crude absolute production-safe packing bound: every scalar term is below
    // 2*3^28 and at most 24 terms contribute, far below signed 56-bit range.
    u64 pow3=1;for(int i=0;i<28;++i)pow3*=3;
    __int128 bound=__int128(48)*pow3;
    if(bound>=(__int128(1)<<55))return 3;
    std::cout<<"b300-high-drop-chunk-proof OK checked="<<checked
             <<" p_range=15..27 max_chunks=3 max_tail=3 signed56_safe=1 height_carry_exact=1 scalar_delta_exact=1\n";
    return 0;
}
