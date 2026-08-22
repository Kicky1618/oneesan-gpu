#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
using boost::multiprecision::cpp_int;

static inline bool checker_ok(uint32_t x,uint32_t y,int h){
    uint32_t m=(h<=1)?0u:((1u<<(h-1))-1u);
    uint32_t dx=(x^(x>>1))&m, dy=(y^(y>>1))&m;
    return (dx&dy&(x^y))==0;
}
static inline bool middle_no_isolated(uint32_t x,uint32_t y,uint32_t z,int h){
    for(int r=1;r+1<h;++r){
        int b=(y>>r)&1;
        if(((y>>(r-1))&1)!=b && ((y>>(r+1))&1)!=b && ((x>>r)&1)!=b && ((z>>r)&1)!=b) return false;
    }
    return true;
}
static double log2_cpp(cpp_int const&x){
    if(x==0)return -INFINITY;
    unsigned b=boost::multiprecision::msb(x);
    unsigned sh=b>63?b-63:0;
    uint64_t top=(x>>sh).convert_to<uint64_t>();
    return sh+std::log2((double)top);
}
int main(int argc,char**argv){
    int h=argc>1?std::atoi(argv[1]):7,w=argc>2?std::atoi(argv[2]):27;
    if(h<1||h>10||w<1){std::cerr<<"h=1..10\n";return 2;}
    int S=1<<h;
    std::vector<std::vector<uint16_t>> nxt(S);
    for(int x=0;x<S;++x)for(int y=0;y<S;++y)if(checker_ok(x,y,h))nxt[x].push_back((uint16_t)y);
    if(w==1){std::cout<<"bits="<<h<<"\n";return 0;}
    // state is ordered pair (previous,current). No isolation test on the first/last global column.
    std::vector<cpp_int> dp((size_t)S*S),ndp((size_t)S*S);
    for(int x=0;x<S;++x)for(int y:nxt[x])dp[(size_t)x*S+y]=1;
    for(int col=2;col<w;++col){
        std::fill(ndp.begin(),ndp.end(),cpp_int(0));
        for(int x=0;x<S;++x)for(int y=0;y<S;++y){auto const&v=dp[(size_t)x*S+y];if(v==0)continue;for(int z:nxt[y])if(middle_no_isolated(x,y,z,h))ndp[(size_t)y*S+z]+=v;}
        dp.swap(ndp);
    }
    cpp_int ans=0;for(auto const&v:dp)ans+=v;
    double lg=log2_cpp(ans);
    std::cout<<"h="<<h<<" w="<<w<<" log2="<<lg<<" percell="<<(lg/(h*w))<<" bitlen="<<(boost::multiprecision::msb(ans)+1)<<"\n";
}
