#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

using boost::multiprecision::cpp_int;

struct DSU {
    int p[24]; bool touch[24]{};
    explicit DSU(int n){for(int i=0;i<n;++i)p[i]=i;}
    int find(int x){while(p[x]!=x){p[x]=p[p[x]];x=p[x];}return x;}
    void unite(int a,int b){a=find(a);b=find(b);if(a==b)return;p[b]=a;touch[a]=touch[a]||touch[b];}
};

static inline bool checker_ok(uint32_t x,uint32_t y,int h){
    if(h<=1)return true;
    uint32_t m=(1u<<(h-1))-1u;
    uint32_t dx=(x^(x>>1))&m,dy=(y^(y>>1))&m;
    return (dx&dy&(x^y))==0;
}

// Layout: [color:h][label:4*h][touched:h]. Labels are canonical 0..k-1.
static uint64_t encode(uint32_t color,const std::vector<int>&lab,uint32_t touched,int h){
    uint64_t z=color,sh=h;
    for(int r=0;r<h;++r){z|=uint64_t(lab[r])<<sh;sh+=4;}
    z|=uint64_t(touched)<<sh;
    return z;
}
static void decode(uint64_t z,uint32_t&color,int lab[12],uint32_t&touched,int h){
    color=uint32_t(z&((1ull<<h)-1));uint64_t sh=h;
    for(int r=0;r<h;++r){lab[r]=int((z>>sh)&15);sh+=4;}
    touched=uint32_t(z>>sh)&((1u<<h)-1);
}

static uint64_t initial_state(uint32_t color,int h){
    std::vector<int>lab(h);int k=0;lab[0]=0;
    for(int r=1;r<h;++r){if(((color>>r)&1)!=((color>>(r-1))&1))++k;lab[r]=k;}
    uint32_t touch=(1u<<(k+1))-1u; // first column touches the real left boundary
    return encode(color,lab,touch,h);
}

static bool advance(uint64_t st,uint32_t ny,int h,uint64_t&out){
    uint32_t ox,ot;int olab[12];decode(st,ox,olab,ot,h);
    DSU d(2*h);
    // Restore old frontier connectivity.
    for(int i=0;i<h;++i)for(int j=0;j<i;++j)if(olab[i]==olab[j])d.unite(i,j);
    for(int i=0;i<h;++i)if((ot>>olab[i])&1u)d.touch[d.find(i)]=true;
    // New vertical and horizontal adjacency.
    for(int r=0;r<h;++r){
        if(((ox>>r)&1u)==((ny>>r)&1u))d.unite(r,h+r);
        if(r&&(((ny>>r)&1u)==((ny>>(r-1))&1u)))d.unite(h+r,h+r-1);
    }
    // Artificial top/bottom strip boundaries may connect to rows outside the strip.
    d.touch[d.find(h)]=true;
    d.touch[d.find(2*h-1)]=true;
    // Propagate touch after all unions.
    for(int i=0;i<2*h;++i){int r=d.find(i);if(d.touch[i])d.touch[r]=true;}
    // Any old component that vanished from the frontier is now permanently enclosed.
    bool has_new[24]{};for(int r=0;r<h;++r)has_new[d.find(h+r)]=true;
    bool seen_old[24]{};
    for(int r=0;r<h;++r){int q=d.find(r);if(seen_old[q])continue;seen_old[q]=true;if(!has_new[q]&&!d.touch[q])return false;}
    // Canonicalize new frontier roots.
    int root_to_label[24];std::fill(std::begin(root_to_label),std::end(root_to_label),-1);
    std::vector<int>lab(h);int k=0;uint32_t nt=0;
    for(int r=0;r<h;++r){int q=d.find(h+r);int&v=root_to_label[q];if(v<0)v=k++;lab[r]=v;if(d.touch[q])nt|=1u<<v;}
    out=encode(ny,lab,nt,h);return true;
}

static double log2_cpp(cpp_int const&x){unsigned b=boost::multiprecision::msb(x),sh=b>63?b-63:0;uint64_t top=(x>>sh).convert_to<uint64_t>();return sh+std::log2((double)top);}

int main(int argc,char**argv){
    int h=argc>1?std::atoi(argv[1]):7,w=argc>2?std::atoi(argv[2]):27;
    if(h<1||h>11||w<1){std::cerr<<"h=1..11\n";return 2;}
    int S=1<<h;std::vector<std::vector<uint16_t>>nxt(S);
    for(int x=0;x<S;++x)for(int y=0;y<S;++y)if(checker_ok(x,y,h))nxt[x].push_back((uint16_t)y);
    std::unordered_map<uint64_t,cpp_int>dp,ndp;dp.reserve(S*8);
    for(int x=0;x<S;++x)dp[initial_state(x,h)]+=1;
    std::cerr<<"col=1 states="<<dp.size()<<"\n";
    for(int col=1;col<w;++col){
        ndp.clear();ndp.reserve(dp.size()*2);
        for(auto const&kv:dp){uint32_t x,ot;int ol[12];decode(kv.first,x,ol,ot,h);for(uint16_t y:nxt[x]){uint64_t ns;if(advance(kv.first,y,h,ns))ndp[ns]+=kv.second;}}
        dp.swap(ndp);std::cerr<<"col="<<(col+1)<<" states="<<dp.size()<<"\n";
    }
    cpp_int ans=0;for(auto const&kv:dp)ans+=kv.second;
    double lg=log2_cpp(ans);
    std::cout<<"h="<<h<<" w="<<w<<" log2="<<lg<<" percell="<<(lg/(h*w))<<" bitlen="<<(boost::multiprecision::msb(ans)+1)<<" states="<<dp.size()<<"\n";
}
