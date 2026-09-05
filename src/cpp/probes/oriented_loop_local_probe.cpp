#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353u;
constexpr u32 G = 3u;

u32 addp(u32 a, u32 b) { u32 c = a + b; return c >= P ? c - P : c; }
u32 mulp(u32 a, u32 b) { return static_cast<u32>((u64)a * b % P); }
u32 powp(u32 a, u64 e) { u32 r=1; while(e){ if(e&1) r=mulp(r,a); a=mulp(a,a); e>>=1;} return r; }

struct Key { u64 trits; std::int8_t left; bool operator==(Key const&o)const{return trits==o.trits&&left==o.left;} };
struct KH { size_t operator()(Key const&k)const noexcept { u64 x=k.trits*0x9e3779b97f4a7c15ULL+(u64)(k.left+2); x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31; return (size_t)x;} };

int get_trit(u64 x,int pos){ while(pos--) x/=3; return int(x%3); }
u64 set_trit(u64 x,int pos,int value){ u64 q=1; for(int i=0;i<pos;++i)q*=3; return x + (u64)(value-get_trit(x,pos))*q; }
int dec(int t){ return t==0?0:t==1?+1:-1; }
int enc(int v){ return v==0?0:v>0?1:2; }

// direction: E=0,S=1,W=2,N=3. Edge sign is + along E/S and - along W/N.
struct Local { int r,d; u32 w; };

std::vector<Local> transitions(int left,int up,bool source,bool sink,bool can_r,bool can_d,u32 z,u32 zi){
    std::vector<Local> out;
    for(int r=-1;r<=1;++r)for(int d=-1;d<=1;++d){
        if(!can_r && r) continue;
        if(!can_d && d) continue;
        int in=0,outn=0,idir=-1,odir=-1,deg=0;
        auto edge=[&](int side,int val){
            if(!val)return;
            ++deg;
            // sides: 0=L,1=R,2=U,3=D
            bool incoming=false; int dir=-1;
            if(side==0){ dir=val>0?0:2; incoming=val>0; }
            if(side==1){ dir=val>0?0:2; incoming=val<0; }
            if(side==2){ dir=val>0?1:3; incoming=val>0; }
            if(side==3){ dir=val>0?1:3; incoming=val<0; }
            if(incoming){++in;idir=dir;}else{++outn;odir=dir;}
        };
        edge(0,left); edge(1,r); edge(2,up); edge(3,d);
        u32 w=1;
        if(source){
            if(!(deg==1 && in==0 && outn==1))continue;
            // endpoint gauge z^(outgoing direction)
            w=powp(z,odir);
        }else if(sink){
            if(!(deg==1 && in==1 && outn==0))continue;
            // endpoint gauge z^(-incoming direction)
            w=powp(zi,idir);
        }else{
            if(deg==0){ out.push_back({r,d,1}); continue; }
            if(!(deg==2 && in==1 && outn==1))continue;
            int delta=(odir-idir+4)%4;
            if(delta==0)w=1;
            else if(delta==1)w=z;        // clockwise/right quarter turn
            else if(delta==3)w=zi;       // counterclockwise/left quarter turn
            else continue;
        }
        out.push_back({r,d,w});
    }
    return out;
}

u32 solve(int n, size_t& peak){
    int W=n+1;
    u32 z=powp(G,(P-1)/16), zi=powp(z,P-2);
    std::cerr<<"z="<<z<<" z4="<<powp(z,4)<<" z8="<<powp(z,8)<<"\n";
    std::unordered_map<Key,u32,KH> cur,nxt;
    cur.reserve(1024); cur[{0,0}]=1; peak=1;
    for(int y=0;y<W;++y){
        for(int x=0;x<W;++x){
            nxt.clear(); nxt.reserve(cur.size()*2+16);
            for(auto const&[k,val]:cur){
                int up=dec(get_trit(k.trits,x));
                bool source=(x==0&&y==0), sink=(x==W-1&&y==W-1);
                auto ts=transitions(k.left,up,source,sink,x+1<W,y+1<W,z,zi);
                for(auto const&t:ts){
                    Key q{set_trit(k.trits,x,enc(t.d)),(std::int8_t)t.r};
                    u32 a=mulp(val,t.w); auto it=nxt.find(q); if(it==nxt.end())nxt.emplace(q,a); else it->second=addp(it->second,a);
                }
            }
            cur.swap(nxt); peak=std::max(peak,cur.size());
        }
        // right boundary guarantees left=0
    }
    auto it=cur.find({0,0});
    return it==cur.end()?0:it->second;
}
}

int main(int argc,char**argv){
    int lo=argc>1?std::atoi(argv[1]):1, hi=argc>2?std::atoi(argv[2]):lo;
    for(int n=lo;n<=hi;++n){size_t peak=0;auto t=std::chrono::steady_clock::now();u32 a=solve(n,peak);double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::cout<<"n="<<n<<" mod="<<a<<" peak="<<peak<<" sec="<<s<<"\n";}
}
