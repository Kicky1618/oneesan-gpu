#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
using namespace std; using boost::multiprecision::cpp_int;

static int K=14;
static vector<uint32_t> codes;
static unordered_map<uint32_t,uint32_t> idx;

static void gen(int depth,int x,int y,uint32_t code, vector<pair<int,int>>& pos){
    if(depth==K){ codes.push_back(code); return; }
    static const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
    for(int d=0;d<4;d++){
        int nx=x+dx[d], ny=y+dy[d]; bool ok=true;
        for(auto [px,py]:pos) if(px==nx&&py==ny){ok=false;break;}
        if(!ok) continue;
        pos.push_back({nx,ny}); gen(depth+1,nx,ny,(code<<2)|d,pos); pos.pop_back();
    }
}
static bool append_ok(uint32_t code,int d){
    static const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
    int x=0,y=0; vector<pair<int,int>> p; p.reserve(K+1); p.push_back({0,0});
    for(int i=K-1;i>=0;--i){int q=(code>>(2*i))&3; x+=dx[q];y+=dy[q];p.push_back({x,y});}
    int nx=x+dx[d],ny=y+dy[d];
    for(auto [px,py]:p)if(px==nx&&py==ny)return false;
    return true;
}
int main(int argc,char**argv){if(argc>1)K=atoi(argv[1]); if(K<1||K>15){cerr<<"K 1..15\n";return 1;}
    vector<pair<int,int>>p{{0,0}}; gen(0,0,0,0,p); cerr<<"K="<<K<<" states="<<codes.size()<<"\n";
    idx.reserve(codes.size()*13/10); idx.max_load_factor(0.8); for(uint32_t i=0;i<codes.size();++i)idx[codes[i]]=i;
    uint64_t mask=(K==16?~0u:((1ull<<(2*K))-1));
    vector<array<int32_t,4>> tr(codes.size()); vector<uint8_t> deg(codes.size());
    for(size_t i=0;i<codes.size();++i){for(int d=0;d<4;d++)if(append_ok(codes[i],d)){uint32_t nc=((uint64_t(codes[i])<<2)|d)&mask; auto it=idx.find(nc); if(it==idx.end()){cerr<<"missing state\n";return 2;}tr[i][deg[i]++]=it->second;}}
    vector<uint64_t> f(codes.size(),1),nf(codes.size()); vector<uint64_t>C(41);C[0]=1;
    for(int b=1;b<=40;b++){uint64_t mx=0; for(size_t i=0;i<codes.size();++i){uint64_t z=0;for(int j=0;j<deg[i];j++)z+=f[tr[i][j]];nf[i]=z;mx=max(mx,z);} f.swap(nf); C[b]=mx; cerr<<"b="<<b<<" C="<<mx<<"\n";}
    cpp_int bound=0; cpp_int cK=codes.size();
    for(int l=54;l<=782;l+=2){int rem=l-K;if(rem<0)continue;int q=rem/40,r=rem%40;cpp_int z=cK;for(int j=0;j<q;j++)z*=C[40];if(r)z*=C[r];bound+=z;}
    unsigned bits=boost::multiprecision::msb(bound)+1; cout<<"K="<<K<<" states="<<codes.size()<<" bound_bits="<<bits<<" C40="<<C[40]<<"\n";
}
