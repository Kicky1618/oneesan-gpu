#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using u32=std::uint32_t; using u64=std::uint64_t;
static constexpr u32 P=998244353u;
static u32 mul(u32 a,u32 b){return u32((u64(a)*b)%P);} static u32 pw(u32 a,u64 e){u32 r=1;while(e){if(e&1)r=mul(r,a);a=mul(a,a);e>>=1;}return r;} static u32 inv(u32 a){return pw(a,P-2);} static u32 neg(u32 a){return a?P-a:0;}

static int dh(char c){return c=='A'?-1:c=='D'?1:0;}
static bool valid(std::string const&s){int h=0;for(char c:s){h+=dh(c);if(h<0)return false;}return h==0;}
static void gen_rec(int r,std::string&s,std::vector<std::string>&out){if((int)s.size()==r){if(valid(s))out.push_back(s);return;}for(char c:std::string("ABCD")){s.push_back(c);gen_rec(r,s,out);s.pop_back();}}
static std::vector<std::string> paths(int r){std::vector<std::string>o;std::string s;gen_rec(r,s,o);return o;}
static int height_before(std::string const&s,int k){int h=0;for(int i=0;i<k;++i)h+=dh(s[i]);return h;}

using Terms=std::vector<std::pair<std::string,u32>>;
static Terms cross_pair(char x,char y,int h){
    auto one=[](std::string z){return Terms{{z,1}};};
    std::string p;p+=x;p+=y;
    if(p=="AB"||p=="BA")return {{"AC",1},{"CA",1}};
    if(p=="AD")return one("CC");
    if(p=="BB"){
        Terms z{{"BC",1},{"CB",1},{"DA",1}};
        if(h)z.push_back({"AD",neg(mul(u32(h),inv(h+1)))});
        return z;
    }
    if(p=="BC"||p=="CB")return one("CC");
    if(p=="BD"||p=="DB")return {{"CD",1},{"DC",1}};
    if(p=="DA")return {{"CC",neg(mul(u32(h+2),inv(h+1)))}};
    return {};
}

static Terms e_on(std::string const&s,int i){
    int r=s.size(); Terms out;
    if(i==0){int k=r-1;if(s[k]=='B'){auto t=s;t[k]='C';out.push_back({t,1});}return out;}
    if(i&1){int j=(i-1)/2,k=r-1-j;if(s[k]=='C'){auto t=s;t[k]='B';out.push_back({t,1});}return out;}
    int j=i/2,k=r-1-j,h=height_before(s,k);
    for(auto const&[p,c]:cross_pair(s[k],s[k+1],h)){auto t=s;t[k]=p[0];t[k+1]=p[1];if(valid(t))out.push_back({t,c});}
    return out;
}

using Vec=std::map<std::string,u32>;
static Vec apply(Vec const&v,int i){Vec z;for(auto const&[s,x]:v)for(auto const&[t,c]:e_on(s,i)){u32 y=(z[t]+mul(x,c))%P; if(y)z[t]=y;else z.erase(t);}return z;}
static bool eq(Vec const&a,Vec const&b){return a==b;}

static u64 catalan(int n){u64 c=1;for(int k=0;k<n;++k)c=c*2*(2*k+1)/(k+2);return c;}

int main(int argc,char**argv){int maxr=argc>1?std::atoi(argv[1]):8;if(maxr>10)maxr=10;for(int r=1;r<=maxr;++r){auto ps=paths(r);u64 want=catalan(r+1);if(ps.size()!=want){std::cerr<<"dimension mismatch r="<<r<<" got="<<ps.size()<<" want="<<want<<"\n";return 1;}int n=2*r+1;for(auto const&s:ps){Vec v{{s,1}};for(int i=0;i<n-1;++i){if(!apply(apply(v,i),i).empty()){std::cerr<<"e^2 failure r="<<r<<" i="<<i<<" s="<<s<<"\n";return 2;}}for(int i=0;i<n-2;++i){auto a=apply(apply(apply(v,i),i+1),i);auto b=apply(v,i);if(!eq(a,b)){std::cerr<<"TL failure r="<<r<<" i="<<i<<" s="<<s<<"\n";return 3;}a=apply(apply(apply(v,i+1),i),i+1);b=apply(v,i+1);if(!eq(a,b)){std::cerr<<"TL reverse failure r="<<r<<" i="<<i<<" s="<<s<<"\n";return 4;}}for(int i=0;i<n-1;++i)for(int j=i+2;j<n-1;++j){auto a=apply(apply(v,i),j),b=apply(apply(v,j),i);if(!eq(a,b)){std::cerr<<"commute failure r="<<r<<" i="<<i<<" j="<<j<<" s="<<s<<"\n";return 5;}}}
        std::cout<<"r="<<r<<" n="<<n<<" dim="<<ps.size()<<" TL_relations=OK\n";
    }
    return 0;
}
