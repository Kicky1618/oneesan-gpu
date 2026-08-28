#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

// Explicit A111960 channel labels:
// length-r words over {0,+,-,|}, exactly h separators '|', and in every gap
// between separators the number of '+' and '-' symbols is equal.
// 0 is the trivial V(0) state; +/- are the two weights of V(1).

using U64=std::uint64_t;

static U64 count_words(int r,int h){
    // DP over position, separators used, and charge in the current gap.
    int off=r;
    std::vector<U64> a((h+1)*(2*r+1)),b(a.size());
    auto ix=[&](int s,int q){return size_t(s)*(2*r+1)+(q+off);};
    a[ix(0,0)]=1;
    for(int p=0;p<r;++p){
        std::fill(b.begin(),b.end(),0);
        for(int s=0;s<=h;++s)for(int q=-r;q<=r;++q){
            U64 v=a[ix(s,q)]; if(!v)continue;
            b[ix(s,q)]+=v; // 0
            if(q<r)b[ix(s,q+1)]+=v; // +
            if(q>-r)b[ix(s,q-1)]+=v; // -
            if(s<h&&q==0)b[ix(s+1,0)]+=v; // separator
        }
        a.swap(b);
    }
    return a[ix(h,0)];
}

static U64 formula(int r,int h){
    if(h<0||h>r)return 0; int n=r-h;
    __uint128_t am1=0,a=1;
    for(int j=0;j<n;++j){
        __uint128_t rhs=__uint128_t(2*j+h+1)*a;
        if(j>0)rhs+=__uint128_t(3*(j+h))*am1;
        assert(rhs%(j+1)==0);
        __uint128_t z=rhs/(j+1); am1=a; a=z;
    }
    return U64(a);
}

int main(int argc,char**argv){
    int nmax=argc>1?std::atoi(argv[1]):18;
    for(int r=0;r<=nmax;++r){
        std::cout<<"r="<<r<<":";
        for(int h=0;h<=r;++h){
            U64 a=count_words(r,h),b=formula(r,h);
            if(a!=b){std::cerr<<" mismatch r="<<r<<" h="<<h<<" word="<<a<<" formula="<<b<<"\n";return 1;}
            std::cout<<' '<<a;
        }
        std::cout<<'\n';
    }
    std::cout<<"PASS charge-word channels = A111960\n";
}
